import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Assuming user has run: !git clone https://github.com/google-research/tabfm.git && pip install -e tabfm[pytorch]
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
from tabfm import TabFMRegressor

def project_schema_constraints(x_adv, x_clean, epsilon, bounds_min, bounds_max):
    x_adv = torch.maximum(torch.minimum(x_adv, x_clean + epsilon), x_clean - epsilon)
    x_adv = torch.maximum(x_adv, bounds_min)
    x_adv = torch.minimum(x_adv, bounds_max)
    return x_adv

def run_experiment():
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available! Colab is running on CPU.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("1. Loading California Housing Dataset...")
    housing = fetch_california_housing(as_frame=True)
    X = housing.frame.drop('MedHouseVal', axis=1)
    y = housing.frame['MedHouseVal']
    
    train_size = 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=10, random_state=42)
    
    # ---------------------------------------------------------
    # TABFM PIPELINE
    # ---------------------------------------------------------
    print("2. Generating Poisoned Context on TabFM...")
    pt_model = tabfm_v1_0_0.load(model_type="regression", device=str(device), dtype=torch.float32).to(device)
    reg = TabFMRegressor(model=pt_model)
    reg.fit(X_train, y_train)
    
    tabfm_clean_preds = reg.predict(X_test)
    tabfm_clean_mse = mean_squared_error(y_test, tabfm_clean_preds)
    
    X_train_transformed = reg.X_encoder_.transform(X_train)
    bounds_min = torch.tensor(X_train_transformed.min(axis=0), dtype=torch.float32, device=device)
    bounds_max = torch.tensor(X_train_transformed.max(axis=0), dtype=torch.float32, device=device)
    
    data = reg.ensemble_generator_.transform(reg.X_encoder_.transform(X_test))
    Xs_all, ys_all, cat_masks_all, ds_all, _ = reg.ensemble_generator_.prepare_ensemble_tensors(data)
    
    Xs_tensor = torch.tensor(Xs_all, dtype=torch.float32, device=device)
    ys_tensor = F.pad(torch.tensor(ys_all, dtype=torch.float32, device=device), (0, Xs_tensor.shape[1] - ys_all.shape[1]), value=-100.0)
    cat_masks_tensor = torch.tensor(cat_masks_all, dtype=torch.bool, device=device)
    ds_tensor = torch.tensor(ds_all, dtype=torch.int32, device=device)
    
    train_size_tensor = torch.full((Xs_tensor.shape[0],), train_size, dtype=torch.int32, device=device)
    y_test_scaled = torch.tensor(reg.y_scaler_.transform(y_test.values.reshape(-1, 1)).flatten(), dtype=torch.float32, device=device)
    
    epsilon = 0.05 * (bounds_max - bounds_min)
    alpha = epsilon / 5
    num_steps = 100
    
    Xs_adv = Xs_tensor.clone().detach().requires_grad_(True)
    pt_model.eval()
    
    for step in range(num_steps):
        idx = torch.randperm(Xs_tensor.shape[0], device=device)[:2]
        out = pt_model(Xs_adv[idx], ys_tensor[idx], train_size_tensor[idx], cat_mask=cat_masks_tensor[idx], d=ds_tensor[idx])
        preds = out[:, train_size:, :].squeeze(-1).mean(dim=0)
        
        loss = -F.mse_loss(preds, y_test_scaled)
        grad = torch.autograd.grad(loss, Xs_adv)[0]
        Xs_adv_data = Xs_adv - alpha * grad.sign()
        
        context_adv = project_schema_constraints(Xs_adv_data[:, :train_size, :], Xs_tensor[:, :train_size, :], epsilon, bounds_min, bounds_max)
        Xs_adv_data[:, :train_size, :] = context_adv
        Xs_adv_data[:, train_size:, :] = Xs_tensor[:, train_size:, :]
        Xs_adv = Xs_adv_data.clone().detach().requires_grad_(True)
        torch.cuda.empty_cache()
        
    with torch.no_grad():
        out = pt_model(Xs_adv[:4], ys_tensor[:4], train_size_tensor[:4], cat_mask=cat_masks_tensor[:4], d=ds_tensor[:4])
        preds_unscaled = reg.y_scaler_.inverse_transform(out[:, train_size:, :].squeeze(-1).mean(dim=0).cpu().numpy().reshape(-1, 1)).flatten()
        tabfm_adv_mse = mean_squared_error(y_test, preds_unscaled)

    # ---------------------------------------------------------
    # XGBOOST COMPARISON PIPELINE
    # ---------------------------------------------------------
    print("\n3. Testing XGBoost Robustness to the same Poisoned Context...")
    
    # Extract Clean data for Ensemble Member 0
    X_train_clean_np = Xs_tensor[0, :train_size, :].cpu().numpy()
    y_train_clean_np = ys_tensor[0, :train_size].cpu().numpy()
    X_test_np = Xs_tensor[0, train_size:, :].cpu().numpy()
    y_test_np = y_test.values  
    
    # Train XGBoost on CLEAN context
    xgb_clean = XGBRegressor(n_estimators=100, random_state=42)
    xgb_clean.fit(X_train_clean_np, y_train_clean_np)
    xgb_clean_preds_scaled = xgb_clean.predict(X_test_np)
    xgb_clean_preds = reg.y_scaler_.inverse_transform(xgb_clean_preds_scaled.reshape(-1, 1)).flatten()
    xgb_clean_mse = mean_squared_error(y_test_np, xgb_clean_preds)
    
    # Extract POISONED data for Ensemble Member 0
    X_train_adv_np = Xs_adv[0, :train_size, :].detach().cpu().numpy()
    
    # Train XGBoost on POISONED context
    xgb_adv = XGBRegressor(n_estimators=100, random_state=42)
    xgb_adv.fit(X_train_adv_np, y_train_clean_np)
    xgb_adv_preds_scaled = xgb_adv.predict(X_test_np)
    xgb_adv_preds = reg.y_scaler_.inverse_transform(xgb_adv_preds_scaled.reshape(-1, 1)).flatten()
    xgb_adv_mse = mean_squared_error(y_test_np, xgb_adv_preds)
    
    print("\n=========================================")
    print("COMPARATIVE ROBUSTNESS RESULTS")
    print("=========================================")
    print(f"TabFM   Clean MSE: {tabfm_clean_mse:.4f} | Poisoned MSE: {tabfm_adv_mse:.4f}")
    print(f"XGBoost Clean MSE: {xgb_clean_mse:.4f} | Poisoned MSE: {xgb_adv_mse:.4f}")
    
if __name__ == "__main__":
    run_experiment()
