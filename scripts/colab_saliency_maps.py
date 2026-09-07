import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
from tabfm import TabFMRegressor

def project_schema_constraints(x_adv, x_clean, epsilon, bounds_min, bounds_max):
    x_adv = torch.maximum(torch.minimum(x_adv, x_clean + epsilon), x_clean - epsilon)
    x_adv = torch.maximum(x_adv, bounds_min)
    x_adv = torch.minimum(x_adv, bounds_max)
    return x_adv

def plot_saliency_heatmap(clean_saliency, adv_saliency, feature_names, out_path="saliency_comparison.png"):
    """
    Plots a side-by-side heatmap of Feature x Context Row importance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # We take the mean absolute gradient across the sequence length (Context Rows)
    # Shape of saliency: [Context_Size, Features]
    
    sns.heatmap(clean_saliency, cmap="Blues", ax=axes[0], cbar_kws={'label': 'Gradient Magnitude'})
    axes[0].set_title("CLEAN Context Saliency (Model Focus)")
    axes[0].set_xlabel("Feature Index")
    axes[0].set_ylabel("Context Row Index")
    axes[0].set_xticklabels(feature_names, rotation=45, ha='right')
    
    sns.heatmap(adv_saliency, cmap="Reds", ax=axes[1], cbar_kws={'label': 'Gradient Magnitude'})
    axes[1].set_title("POISONED Context Saliency (Attack Focus)")
    axes[1].set_xlabel("Feature Index")
    axes[1].set_ylabel("Context Row Index")
    axes[1].set_xticklabels(feature_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"--> Saved Saliency Heatmap to {out_path}")

def run_experiment():
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available! Colab is running on CPU.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("1. Loading California Housing Dataset...")
    housing = fetch_california_housing(as_frame=True)
    X = housing.frame.drop('MedHouseVal', axis=1)
    y = housing.frame['MedHouseVal']
    feature_names = list(X.columns)
    
    # Use a small context size (e.g., 20) so the heatmap is readable
    train_size = 20
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
    
    print("2. Extracting CLEAN Saliency Maps (Input x Gradient)...")
    # We want to see how much the loss depends on the clean context features
    Xs_clean = Xs_tensor.clone().detach().requires_grad_(True)
    pt_model.eval()
    
    out = pt_model(Xs_clean[:1], ys_tensor[:1], train_size_tensor[:1], cat_mask=cat_masks_tensor[:1], d=ds_tensor[:1])
    preds = out[:, train_size:, :].squeeze(-1).mean(dim=0)
    loss = F.mse_loss(preds, y_test_scaled)
    
    grad_clean = torch.autograd.grad(loss, Xs_clean)[0]
    # Saliency = |Gradient| for the context rows of the first ensemble member
    # Shape: [Context_Size, Features]
    saliency_clean_np = torch.abs(grad_clean[0, :train_size, :]).detach().cpu().numpy()
    
    print("3. Generating Poisoned Context (50 Steps)...")
    Xs_adv = Xs_tensor.clone().detach().requires_grad_(True)
    for step in range(50):
        out = pt_model(Xs_adv[:2], ys_tensor[:2], train_size_tensor[:2], cat_mask=cat_masks_tensor[:2], d=ds_tensor[:2])
        preds = out[:, train_size:, :].squeeze(-1).mean(dim=0)
        
        loss = -F.mse_loss(preds, y_test_scaled)
        grad = torch.autograd.grad(loss, Xs_adv)[0]
        
        Xs_adv_data = Xs_adv - alpha * grad.sign()
        context_adv = project_schema_constraints(Xs_adv_data[:, :train_size, :], Xs_tensor[:, :train_size, :], epsilon, bounds_min, bounds_max)
        Xs_adv_data[:, :train_size, :] = context_adv
        Xs_adv_data[:, train_size:, :] = Xs_tensor[:, train_size:, :]
        
        Xs_adv = Xs_adv_data.clone().detach().requires_grad_(True)
        torch.cuda.empty_cache()
        
    print("4. Extracting POISONED Saliency Maps...")
    # Calculate gradients on the adversarial context
    out_adv = pt_model(Xs_adv[:1], ys_tensor[:1], train_size_tensor[:1], cat_mask=cat_masks_tensor[:1], d=ds_tensor[:1])
    preds_adv = out_adv[:, train_size:, :].squeeze(-1).mean(dim=0)
    loss_adv = F.mse_loss(preds_adv, y_test_scaled)
    
    grad_adv = torch.autograd.grad(loss_adv, Xs_adv)[0]
    saliency_adv_np = torch.abs(grad_adv[0, :train_size, :]).detach().cpu().numpy()
    
    print("5. Plotting and Saving Heatmaps...")
    plot_saliency_heatmap(saliency_clean_np, saliency_adv_np, feature_names)
    print("Done! You can view 'saliency_comparison.png' in the Colab file browser.")

if __name__ == "__main__":
    run_experiment()
