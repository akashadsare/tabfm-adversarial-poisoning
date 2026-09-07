import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error

# Assuming user has run: !git clone https://github.com/google-research/tabfm.git && pip install -e tabfm[pytorch]
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
from tabfm import TabFMRegressor

def project_schema_constraints(x_adv, x_clean, epsilon, bounds_min, bounds_max):
    x_adv = torch.maximum(torch.minimum(x_adv, x_clean + epsilon), x_clean - epsilon)
    x_adv = torch.maximum(x_adv, bounds_min)
    x_adv = torch.minimum(x_adv, bounds_max)
    return x_adv

def knn_sanitization(X_poisoned, k=3):
    """
    KNN Sanitization Defense:
    Replaces each row in the context with the average of its K nearest neighbors.
    This aims to 'smooth out' high-frequency adversarial noise.
    """
    sanitized = np.zeros_like(X_poisoned)
    knn = NearestNeighbors(n_neighbors=k+1)
    knn.fit(X_poisoned)
    distances, indices = knn.kneighbors(X_poisoned)
    
    for i in range(len(X_poisoned)):
        # Average the K nearest neighbors (excluding the point itself)
        neighbors = X_poisoned[indices[i, 1:]]
        sanitized[i] = np.mean(neighbors, axis=0)
        
    return sanitized

def run_experiment():
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available! Colab is running on CPU.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("1. Loading California Housing Dataset & Setup...")
    housing = fetch_california_housing(as_frame=True)
    X = housing.frame.drop('MedHouseVal', axis=1)
    y = housing.frame['MedHouseVal']
    
    train_size = 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=10, random_state=42)
    
    pt_model = tabfm_v1_0_0.load(model_type="regression", device=str(device), dtype=torch.float32).to(device)
    reg = TabFMRegressor(model=pt_model)
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
    
    train_size_tensor = torch.full((Xs_tensor.shape[0],), train_size, dtype=torch.int32, device=device)
    y_test_scaled = torch.tensor(reg.y_scaler_.transform(y_test.values.reshape(-1, 1)).flatten(), dtype=torch.float32, device=device)
    
    epsilon = 0.05 * (bounds_max - bounds_min)
    alpha = epsilon / 5
    
    print("\n2. Fast-Tracking Adversarial Attack (50 Steps)...")
    Xs_adv = Xs_tensor.clone().detach().requires_grad_(True)
    pt_model.eval()
    
    for step in range(50):
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

    print("\n3. Testing KNN Sanitization Defense...")
    # Evaluate Clean Baseline
    with torch.no_grad():
        out_clean = pt_model(Xs_tensor[:4], ys_tensor[:4], train_size_tensor[:4], cat_mask=cat_masks_tensor[:4], d=ds_tensor[:4])
        preds_clean = out_clean[:, train_size:, :].squeeze(-1).mean(dim=0)
        mse_clean = np.mean((reg.y_scaler_.inverse_transform(preds_clean.cpu().numpy().reshape(-1, 1)).flatten() - y_test.values)**2)
        
        # Evaluate Poisoned Baseline
        out_adv = pt_model(Xs_adv[:4], ys_tensor[:4], train_size_tensor[:4], cat_mask=cat_masks_tensor[:4], d=ds_tensor[:4])
        preds_adv = out_adv[:, train_size:, :].squeeze(-1).mean(dim=0)
        mse_adv = np.mean((reg.y_scaler_.inverse_transform(preds_adv.cpu().numpy().reshape(-1, 1)).flatten() - y_test.values)**2)

    print(f"Clean MSE:    {mse_clean:.4f}")
    print(f"Poisoned MSE: {mse_adv:.4f}")
    
    # Apply KNN Defense (K=3, K=5, K=10)
    for k in [3, 5, 10]:
        Xs_sanitized = Xs_adv.clone().detach()
        # Sanitize context for each ensemble member independently
        for i in range(Xs_adv.shape[0]):
            adv_context_np = Xs_adv[i, :train_size, :].detach().cpu().numpy()
            sanitized_context = knn_sanitization(adv_context_np, k=k)
            Xs_sanitized[i, :train_size, :] = torch.tensor(sanitized_context, dtype=torch.float32, device=device)
            
        with torch.no_grad():
            out_sanitized = pt_model(Xs_sanitized[:4], ys_tensor[:4], train_size_tensor[:4], cat_mask=cat_masks_tensor[:4], d=ds_tensor[:4])
            preds_sanitized = out_sanitized[:, train_size:, :].squeeze(-1).mean(dim=0)
            mse_sanitized = np.mean((reg.y_scaler_.inverse_transform(preds_sanitized.cpu().numpy().reshape(-1, 1)).flatten() - y_test.values)**2)
            
        print(f"Sanitized MSE (K={k}): {mse_sanitized:.4f}")

if __name__ == "__main__":
    run_experiment()
