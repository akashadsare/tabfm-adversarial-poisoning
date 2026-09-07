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

def evaluate_subsampling(pt_model, reg, Xs_eval, ys_eval, cat_masks, ds, original_train_size, drop_rates, y_test, target_val=None):
    """
    Evaluates the model by randomly dropping a percentage of the context window.
    Because tabular rows are order-invariant (Set Transformer), dropping the last N rows 
    of the context is mathematically identical to a random drop (assuming pre-shuffled data).
    """
    results = {}
    batch_size = Xs_eval.shape[0]
    device = Xs_eval.device
    
    for drop_rate in drop_rates:
        keep_size = int(original_train_size * (1.0 - drop_rate))
        
        # We construct a new sequence: [Subsampled Context] + [Query]
        context_part = Xs_eval[:, :keep_size, :]
        query_part = Xs_eval[:, original_train_size:, :]
        Xs_subsampled = torch.cat([context_part, query_part], dim=1)
        
        ys_context = ys_eval[:, :keep_size]
        ys_query = ys_eval[:, original_train_size:]
        ys_subsampled = torch.cat([ys_context, ys_query], dim=1)
        
        # New train size tensor
        train_size_tensor = torch.full((batch_size,), keep_size, dtype=torch.int32, device=device)
        
        # Mini-batching evaluation to prevent OOM
        eval_batch = 4
        all_preds = []
        
        with torch.no_grad():
            for i in range(0, batch_size, eval_batch):
                out = pt_model(
                    Xs_subsampled[i:i+eval_batch], 
                    ys_subsampled[i:i+eval_batch], 
                    train_size_tensor[i:i+eval_batch], 
                    cat_mask=cat_masks[i:i+eval_batch], 
                    d=ds[i:i+eval_batch]
                )
                out = out[:, keep_size:, :]
                all_preds.append(out.squeeze(-1))
                
        preds = torch.cat(all_preds, dim=0)
        
        if not reg.enable_nnls:
            preds = preds.mean(dim=0)
            
        preds_unscaled = reg.y_scaler_.inverse_transform(preds.cpu().numpy().reshape(-1, 1)).flatten()
        true_mse = np.mean((preds_unscaled - y_test.values)**2)
        
        if target_val is not None:
            target_mse = np.mean((preds_unscaled - target_val)**2)
            results[drop_rate] = {'True_MSE': true_mse, 'Target_MSE': target_mse}
            print(f"Drop Rate {drop_rate*100:02.0f}% | True MSE: {true_mse:.4f} | Target MSE: {target_mse:.4f}")
        else:
            results[drop_rate] = {'True_MSE': true_mse}
            print(f"Drop Rate {drop_rate*100:02.0f}% | True MSE: {true_mse:.4f}")
            
    return results

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("1. Setup & Loading Model...")
    housing = fetch_california_housing(as_frame=True)
    X, y = housing.frame.drop('MedHouseVal', axis=1), housing.frame['MedHouseVal']
    
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
    ys_tensor = torch.tensor(ys_all, dtype=torch.float32, device=device)
    cat_masks_tensor = torch.tensor(cat_masks_all, dtype=torch.bool, device=device)
    ds_tensor = torch.tensor(ds_all, dtype=torch.int32, device=device)
    
    seq_len = Xs_tensor.shape[1]
    if ys_tensor.shape[1] < seq_len:
        ys_tensor = F.pad(ys_tensor, (0, seq_len - ys_tensor.shape[1]), value=-100.0)
    
    train_size_tensor = torch.full((Xs_tensor.shape[0],), train_size, dtype=torch.int32, device=device)
    
    target_val = 5.0
    y_target_scaled = torch.tensor(reg.y_scaler_.transform(np.full(y_test.shape, target_val).reshape(-1, 1)).flatten(), dtype=torch.float32, device=device)
    
    epsilon = 0.05 * (bounds_max - bounds_min)
    alpha = epsilon / 5
    
    print("\n2. Fast-Tracking Adversarial Attack (50 Steps) to generate poisoned context...")
    Xs_adv = Xs_tensor.clone().detach().requires_grad_(True)
    pt_model.eval()
    
    for step in range(50):
        idx = torch.randperm(Xs_tensor.shape[0], device=device)[:2]
        out = pt_model(Xs_adv[idx], ys_tensor[idx], train_size_tensor[idx], cat_mask=cat_masks_tensor[idx], d=ds_tensor[idx])
        
        preds = out[:, train_size:, :].squeeze(-1)
        if not reg.enable_nnls:
            preds = preds.mean(dim=0)
            
        loss = F.mse_loss(preds, y_target_scaled)
        grad = torch.autograd.grad(loss, Xs_adv)[0]
        
        Xs_adv_data = Xs_adv - alpha * grad.sign()
        
        context_adv = Xs_adv_data[:, :train_size, :]
        context_clean = Xs_tensor[:, :train_size, :]
        
        context_adv_projected = project_schema_constraints(context_adv, context_clean, epsilon, bounds_min, bounds_max)
        
        Xs_adv_data[:, :train_size, :] = context_adv_projected
        Xs_adv_data[:, train_size:, :] = Xs_tensor[:, train_size:, :]
        Xs_adv = Xs_adv_data.clone().detach().requires_grad_(True)
        torch.cuda.empty_cache()

    print("\n3. Testing Defense: Context Subsampling on CLEAN Data")
    drop_rates = [0.0, 0.1, 0.3, 0.5, 0.75]
    evaluate_subsampling(pt_model, reg, Xs_tensor, ys_tensor, cat_masks_tensor, ds_tensor, train_size, drop_rates, y_test)
    
    print("\n4. Testing Defense: Context Subsampling on POISONED Data")
    evaluate_subsampling(pt_model, reg, Xs_adv, ys_tensor, cat_masks_tensor, ds_tensor, train_size, drop_rates, y_test, target_val)

if __name__ == "__main__":
    run_experiment()
