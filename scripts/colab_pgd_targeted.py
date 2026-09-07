import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

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
    
    print("2. Loading TabFM PyTorch Model to GPU (Float32)...")
    pt_model = tabfm_v1_0_0.load(model_type="regression", device=str(device), dtype=torch.float32)
    pt_model = pt_model.to(device)
    
    reg = TabFMRegressor(model=pt_model)
    
    print("3. Evaluating Baseline Zero-Shot Performance...")
    reg.fit(X_train, y_train)
    baseline_preds = reg.predict(X_test)
    baseline_mse = np.mean((baseline_preds - y_test.values)**2)
    print(f"--> Baseline MSE against True Labels: {baseline_mse:.4f}")

    print("4. Preparing Transformed Tensors for White-Box Attack...")
    X_train_transformed = reg.X_encoder_.transform(X_train)
    bounds_min = torch.tensor(X_train_transformed.min(axis=0), dtype=torch.float32, device=device)
    bounds_max = torch.tensor(X_train_transformed.max(axis=0), dtype=torch.float32, device=device)
    
    X_test_transformed = reg.X_encoder_.transform(X_test)
    data = reg.ensemble_generator_.transform(X_test_transformed)
    Xs_all, ys_all, cat_masks_all, ds_all, _ = reg.ensemble_generator_.prepare_ensemble_tensors(data)
    
    Xs_tensor = torch.tensor(Xs_all, dtype=torch.float32, device=device)
    ys_tensor = torch.tensor(ys_all, dtype=torch.float32, device=device)
    cat_masks_tensor = torch.tensor(cat_masks_all, dtype=torch.bool, device=device)
    ds_tensor = torch.tensor(ds_all, dtype=torch.int32, device=device)
    
    batch_size_pt = Xs_tensor.shape[0]
    train_size_tensor = torch.full((batch_size_pt,), train_size, dtype=torch.int32, device=device)
    
    # TARGET DEFINITION: Force model to predict $5,000,000 (represented as 5.0 in the dataset)
    target_val = 5.0
    y_target = np.full(y_test.shape, target_val)
    # Scale the target to match model's internal representation
    y_target_scaled = torch.tensor(reg.y_scaler_.transform(y_target.reshape(-1, 1)).flatten(), dtype=torch.float32, device=device)
    
    epsilon = 0.05 * (bounds_max - bounds_min)
    alpha = epsilon / 5
    num_steps = 100
    batch_size_ensemble = 2 
    num_ensemble = Xs_tensor.shape[0]
    
    Xs_adv = Xs_tensor.clone().detach().requires_grad_(True)
    
    print(f"\n5. Running PGD Targeted Attack (Target = {target_val})...")
    pt_model.eval()
    for step in range(num_steps):
        idx = torch.randperm(num_ensemble, device=device)[:batch_size_ensemble]
        
        Xs_adv_batch = Xs_adv[idx]
        ys_batch = ys_tensor[idx]
        cat_masks_batch = cat_masks_tensor[idx]
        ds_batch = ds_tensor[idx]
        train_size_batch = train_size_tensor[idx]
        
        out = pt_model(Xs_adv_batch, ys_batch, train_size_batch, cat_mask=cat_masks_batch, d=ds_batch)
        out = out[:, train_size:, :]
        preds = out.squeeze(-1)
        
        if not reg.enable_nnls:
            preds = preds.mean(dim=0)
            
        # TARGETED OBJECTIVE: MINIMIZE the error against our fake target label
        loss = F.mse_loss(preds, y_target_scaled)
        grad = torch.autograd.grad(loss, Xs_adv)[0]
        
        # PGD Step: Standard descent (subtract gradient) because we are MINIMIZING loss
        Xs_adv_data = Xs_adv - alpha * grad.sign()
        
        context_adv = Xs_adv_data[:, :train_size, :]
        context_clean = Xs_tensor[:, :train_size, :]
        
        context_adv_projected = project_schema_constraints(context_adv, context_clean, epsilon, bounds_min, bounds_max)
        
        Xs_adv_data[:, :train_size, :] = context_adv_projected
        Xs_adv_data[:, train_size:, :] = Xs_tensor[:, train_size:, :]
        
        Xs_adv = Xs_adv_data.clone().detach().requires_grad_(True)
        
        if step % 10 == 0:
            preds_unscaled = reg.y_scaler_.inverse_transform(preds.detach().cpu().numpy().reshape(-1, 1)).flatten()
            target_mse = np.mean((preds_unscaled - target_val)**2)
            true_mse = np.mean((preds_unscaled - y_test.values)**2)
            print(f"Step {step:03d} | MSE to Target (5.0): {target_mse:.4f} | MSE to True Labels: {true_mse:.4f}")
            
        torch.cuda.empty_cache()
            
    print("\n--- Attack Complete ---")
    
if __name__ == "__main__":
    run_experiment()
