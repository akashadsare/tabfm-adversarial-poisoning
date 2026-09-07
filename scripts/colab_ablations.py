import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import copy

# Assuming user has run: !git clone https://github.com/google-research/tabfm.git && pip install -e tabfm[pytorch]
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
from tabfm import TabFMRegressor

def project_schema_constraints(x_adv, x_clean, epsilon, bounds_min, bounds_max):
    x_adv = torch.maximum(torch.minimum(x_adv, x_clean + epsilon), x_clean - epsilon)
    x_adv = torch.maximum(x_adv, bounds_min)
    x_adv = torch.minimum(x_adv, bounds_max)
    return x_adv

def run_pgd_attack(pt_model, reg, Xs_tensor, ys_tensor, cat_masks_tensor, ds_tensor, y_test, train_size, epsilon_multiplier, bounds_min, bounds_max, num_steps=30):
    """Runs a quick untargeted PGD attack and returns the final MSE."""
    epsilon = epsilon_multiplier * (bounds_max - bounds_min)
    alpha = epsilon / 5
    batch_size_pt = Xs_tensor.shape[0]
    train_size_tensor = torch.full((batch_size_pt,), train_size, dtype=torch.int32, device=Xs_tensor.device)
    
    y_test_scaled = torch.tensor(reg.y_scaler_.transform(y_test.values.reshape(-1, 1)).flatten(), dtype=torch.float32, device=Xs_tensor.device)
    
    Xs_adv = Xs_tensor.clone().detach().requires_grad_(True)
    pt_model.eval()
    
    for step in range(num_steps):
        idx = torch.randperm(batch_size_pt, device=Xs_tensor.device)[:2]
        
        out = pt_model(Xs_adv[idx], ys_tensor[idx], train_size_tensor[idx], cat_mask=cat_masks_tensor[idx], d=ds_tensor[idx])
        preds = out[:, train_size:, :].squeeze(-1)
        if not reg.enable_nnls:
            preds = preds.mean(dim=0)
            
        loss = -F.mse_loss(preds, y_test_scaled)
        grad = torch.autograd.grad(loss, Xs_adv)[0]
        
        Xs_adv_data = Xs_adv - alpha * grad.sign()
        
        context_adv = Xs_adv_data[:, :train_size, :]
        context_clean = Xs_tensor[:, :train_size, :]
        
        context_adv_projected = project_schema_constraints(context_adv, context_clean, epsilon, bounds_min, bounds_max)
        
        Xs_adv_data[:, :train_size, :] = context_adv_projected
        Xs_adv_data[:, train_size:, :] = Xs_tensor[:, train_size:, :]
        
        Xs_adv = Xs_adv_data.clone().detach().requires_grad_(True)
        torch.cuda.empty_cache()
        
    # Final Evaluation
    with torch.no_grad():
        out = pt_model(Xs_adv[:4], ys_tensor[:4], train_size_tensor[:4], cat_mask=cat_masks_tensor[:4], d=ds_tensor[:4])
        preds = out[:, train_size:, :].squeeze(-1).mean(dim=0)
        preds_unscaled = reg.y_scaler_.inverse_transform(preds.cpu().numpy().reshape(-1, 1)).flatten()
        final_mse = np.mean((preds_unscaled - y_test.values)**2)
        
    return final_mse

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("1. Setup & Loading Model...")
    housing = fetch_california_housing(as_frame=True)
    X, y = housing.frame.drop('MedHouseVal', axis=1), housing.frame['MedHouseVal']
    
    pt_model = tabfm_v1_0_0.load(model_type="regression", device=str(device), dtype=torch.float32).to(device)
    reg = TabFMRegressor(model=pt_model)
    
    print("\n=========================================")
    print("ABLATION 1: Epsilon Budget vs Attack Success")
    print("=========================================")
    train_size = 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=10, random_state=42)
    reg.fit(X_train, y_train)
    
    X_train_transformed = reg.X_encoder_.transform(X_train)
    bounds_min = torch.tensor(X_train_transformed.min(axis=0), dtype=torch.float32, device=device)
    bounds_max = torch.tensor(X_train_transformed.max(axis=0), dtype=torch.float32, device=device)
    
    data = reg.ensemble_generator_.transform(reg.X_encoder_.transform(X_test))
    Xs_all, ys_all, cat_masks_all, ds_all, _ = reg.ensemble_generator_.prepare_ensemble_tensors(data)
    
    Xs_tensor = torch.tensor(Xs_all, dtype=torch.float32, device=device)
    ys_tensor = F.pad(torch.tensor(ys_all, dtype=torch.float32, device=device), (0, Xs_tensor.shape[1] - ys_all.shape[1]), value=-100.0)
    cat_masks_tensor = torch.tensor(cat_masks_all, dtype=torch.bool, device=device)
    ds_tensor = torch.tensor(ds_all, dtype=torch.int32, device=device)

    epsilons = [0.01, 0.05, 0.10, 0.20]
    for eps in epsilons:
        mse = run_pgd_attack(pt_model, reg, Xs_tensor, ys_tensor, cat_masks_tensor, ds_tensor, y_test, train_size, eps, bounds_min, bounds_max)
        print(f"Epsilon {eps*100:02.0f}% | Adversarial MSE: {mse:.4f}")

    print("\n=========================================")
    print("ABLATION 2: Context Window Size vs Robustness")
    print("=========================================")
    # Fix epsilon at 5%
    eps = 0.05
    context_sizes = [20, 50, 100, 200]
    
    for c_size in context_sizes:
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y, train_size=c_size, test_size=10, random_state=42)
        reg.fit(X_train_c, y_train_c)
        
        X_train_transformed_c = reg.X_encoder_.transform(X_train_c)
        bounds_min_c = torch.tensor(X_train_transformed_c.min(axis=0), dtype=torch.float32, device=device)
        bounds_max_c = torch.tensor(X_train_transformed_c.max(axis=0), dtype=torch.float32, device=device)
        
        data_c = reg.ensemble_generator_.transform(reg.X_encoder_.transform(X_test_c))
        Xs_all_c, ys_all_c, cat_masks_all_c, ds_all_c, _ = reg.ensemble_generator_.prepare_ensemble_tensors(data_c)
        
        Xs_tensor_c = torch.tensor(Xs_all_c, dtype=torch.float32, device=device)
        ys_tensor_c = F.pad(torch.tensor(ys_all_c, dtype=torch.float32, device=device), (0, Xs_tensor_c.shape[1] - ys_all_c.shape[1]), value=-100.0)
        cat_masks_tensor_c = torch.tensor(cat_masks_all_c, dtype=torch.bool, device=device)
        ds_tensor_c = torch.tensor(ds_all_c, dtype=torch.int32, device=device)
        
        mse = run_pgd_attack(pt_model, reg, Xs_tensor_c, ys_tensor_c, cat_masks_tensor_c, ds_tensor_c, y_test_c, c_size, eps, bounds_min_c, bounds_max_c, num_steps=20)
        print(f"Context Size {c_size:3d} | Adversarial MSE: {mse:.4f}")

if __name__ == "__main__":
    run_experiment()
