import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Assuming user has run: !git clone https://github.com/google-research/tabfm.git && pip install -e tabfm[pytorch]
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
from tabfm import TabFMRegressor

def project_schema_constraints(x_adv, x_clean, epsilon, bounds_min, bounds_max):
    x_adv = torch.maximum(torch.minimum(x_adv, x_clean + epsilon), x_clean - epsilon)
    x_adv = torch.maximum(x_adv, bounds_min)
    x_adv = torch.minimum(x_adv, bounds_max)
    return x_adv

def nadaraya_watson_surrogate(X_context, y_context, X_query, bandwidth=1.0):
    """
    A differentiable non-parametric surrogate (Kernel Regression).
    Like TabFM, it predicts query labels directly from the context window at inference time.
    """
    # Compute pairwise Euclidean distances between queries and context
    # X_query: [N_q, F], X_context: [N_c, F]
    dist = torch.cdist(X_query, X_context, p=2)
    
    # Gaussian Kernel weights
    weights = torch.exp(- (dist ** 2) / (2 * bandwidth ** 2))
    
    # Normalize weights
    weights_normalized = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    
    # Predict as weighted sum of context labels
    # weights: [N_q, N_c], y_context: [N_c] -> preds: [N_q]
    preds = torch.matmul(weights_normalized, y_context)
    return preds

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
    
    print("2. Generating Black-Box Poisoned Context via Kernel Regression Surrogate...")
    # We standard scale for the surrogate to make Euclidean distance meaningful
    X_mean, X_std = X_train.mean(), X_train.std()
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std
    
    X_ctx_t = torch.tensor(X_train_scaled.values, dtype=torch.float32, device=device)
    y_ctx_t = torch.tensor(y_train.values, dtype=torch.float32, device=device)
    X_qry_t = torch.tensor(X_test_scaled.values, dtype=torch.float32, device=device)
    y_qry_t = torch.tensor(y_test.values, dtype=torch.float32, device=device)
    
    bounds_min = X_ctx_t.min(dim=0)[0]
    bounds_max = X_ctx_t.max(dim=0)[0]
    
    epsilon = 0.05 * (bounds_max - bounds_min)
    alpha = epsilon / 5
    
    X_adv_ctx = X_ctx_t.clone().detach().requires_grad_(True)
    
    # PGD to maximize MSE on the Kernel Regression surrogate
    for step in range(100):
        preds = nadaraya_watson_surrogate(X_adv_ctx, y_ctx_t, X_qry_t, bandwidth=2.0)
        loss = -F.mse_loss(preds, y_qry_t)
        
        grad = torch.autograd.grad(loss, X_adv_ctx)[0]
        X_adv_data = X_adv_ctx - alpha * grad.sign()
        
        X_adv_data = project_schema_constraints(X_adv_data, X_ctx_t, epsilon, bounds_min, bounds_max)
        X_adv_ctx = X_adv_data.clone().detach().requires_grad_(True)
        
    surrogate_clean_mse = F.mse_loss(nadaraya_watson_surrogate(X_ctx_t, y_ctx_t, X_qry_t, 2.0), y_qry_t).item()
    surrogate_adv_mse = F.mse_loss(nadaraya_watson_surrogate(X_adv_ctx, y_ctx_t, X_qry_t, 2.0), y_qry_t).item()
    print(f"Surrogate (Kernel Reg) Clean MSE: {surrogate_clean_mse:.4f} | Poisoned MSE: {surrogate_adv_mse:.4f}")

    print("\n3. Transferring Black-Box Poisoned Context to TabFM...")
    # Unscale the adversarial data back to the original pandas space for TabFM's encoder
    X_train_adv_df = pd.DataFrame(X_adv_ctx.detach().cpu().numpy() * X_std.values + X_mean.values, columns=X.columns)
    
    pt_model = tabfm_v1_0_0.load(model_type="regression", device=str(device), dtype=torch.float32).to(device)
    reg = TabFMRegressor(model=pt_model)
    
    # Evaluate Clean TabFM
    reg.fit(X_train, y_train)
    clean_preds = reg.predict(X_test)
    tabfm_clean_mse = np.mean((clean_preds - y_test.values)**2)
    print(f"TabFM Clean MSE: {tabfm_clean_mse:.4f}")
    
    # Evaluate Poisoned TabFM
    reg.fit(X_train_adv_df, y_train)  # Pass the Black-Box poisoned context
    adv_preds = reg.predict(X_test)
    tabfm_adv_mse = np.mean((adv_preds - y_test.values)**2)
    
    print(f"TabFM Transfer (Poisoned) MSE: {tabfm_adv_mse:.4f}")
    
    if tabfm_adv_mse > tabfm_clean_mse:
        print("\nSUCCESS: The Black-Box Transfer Attack successfully degraded TabFM's accuracy!")
    
if __name__ == "__main__":
    run_experiment()
