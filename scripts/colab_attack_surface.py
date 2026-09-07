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

def run_grid_search():
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available! Colab is running on CPU.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("1. Loading California Housing Dataset & TabFM Model...")
    housing = fetch_california_housing(as_frame=True)
    X = housing.frame.drop('MedHouseVal', axis=1)
    y = housing.frame['MedHouseVal']
    
    # Load model ONCE to prevent OOM
    pt_model = tabfm_v1_0_0.load(model_type="regression", device=str(device), dtype=torch.float32).to(device)
    reg = TabFMRegressor(model=pt_model)
    
    epsilons = [0.01, 0.03, 0.05, 0.10]
    context_sizes = [10, 20, 50, 100]
    
    # Matrix to store MSE results (Rows = Context Size, Cols = Epsilon)
    results_matrix = np.zeros((len(context_sizes), len(epsilons)))
    
    for i, c_size in enumerate(context_sizes):
        # We need to re-fit the regressor for each context size to properly set up the ensemble tensors
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=c_size, test_size=10, random_state=42)
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
        
        train_size_tensor = torch.full((Xs_tensor.shape[0],), c_size, dtype=torch.int32, device=device)
        y_test_scaled = torch.tensor(reg.y_scaler_.transform(y_test.values.reshape(-1, 1)).flatten(), dtype=torch.float32, device=device)
        
        for j, eps_pct in enumerate(epsilons):
            print(f"--> Sweeping Context Size = {c_size}, Epsilon = {eps_pct*100:.0f}%")
            epsilon = eps_pct * (bounds_max - bounds_min)
            alpha = epsilon / 5
            
            Xs_adv = Xs_tensor.clone().detach().requires_grad_(True)
            pt_model.eval()
            
            for step in range(30):
                # Micro-batch of 2 to save VRAM
                out = pt_model(Xs_adv[:2], ys_tensor[:2], train_size_tensor[:2], cat_mask=cat_masks_tensor[:2], d=ds_tensor[:2])
                preds = out[:, c_size:, :].squeeze(-1).mean(dim=0)
                loss = -F.mse_loss(preds, y_test_scaled)
                
                grad = torch.autograd.grad(loss, Xs_adv)[0]
                Xs_adv_data = Xs_adv - alpha * grad.sign()
                
                context_adv = project_schema_constraints(Xs_adv_data[:, :c_size, :], Xs_tensor[:, :c_size, :], epsilon, bounds_min, bounds_max)
                Xs_adv_data[:, :c_size, :] = context_adv
                Xs_adv_data[:, c_size:, :] = Xs_tensor[:, c_size:, :]
                
                Xs_adv = Xs_adv_data.clone().detach().requires_grad_(True)
                torch.cuda.empty_cache()
                
            # Evaluate
            with torch.no_grad():
                out_eval = pt_model(Xs_adv[:2], ys_tensor[:2], train_size_tensor[:2], cat_mask=cat_masks_tensor[:2], d=ds_tensor[:2])
                preds_eval = out_eval[:, c_size:, :].squeeze(-1).mean(dim=0)
                final_mse = F.mse_loss(preds_eval, y_test_scaled).item()
                
            results_matrix[i, j] = final_mse
            
            # Clean up loop tensors
            del Xs_adv, out_eval, preds_eval
            torch.cuda.empty_cache()
            
    print("\n2. Plotting 2D Attack Surface Heatmap...")
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(results_matrix, annot=True, fmt=".3f", cmap="Reds", 
                     xticklabels=[f"{e*100:.0f}%" for e in epsilons],
                     yticklabels=context_sizes,
                     cbar_kws={'label': 'Adversarial MSE'})
    
    plt.title("Attack Surface: Epsilon vs. Context Size")
    plt.xlabel(r"L-$\infty$ Epsilon Budget (\% of Feature Range)")
    plt.ylabel("Context Window Size (Rows)")
    plt.tight_layout()
    plt.savefig("attack_surface.png", dpi=300)
    plt.close()
    
    print("Done! View 'attack_surface.png' in the Colab file browser.")

if __name__ == "__main__":
    run_grid_search()
