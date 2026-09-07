import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def project_schema_constraints(x_adv, x_clean, epsilon, bounds_min, bounds_max, cat_masks_tensor=None):
    x_adv_proj = torch.maximum(torch.minimum(x_adv, x_clean + epsilon), x_clean - epsilon)
    x_adv_proj = torch.maximum(x_adv_proj, bounds_min)
    x_adv_proj = torch.minimum(x_adv_proj, bounds_max)
    if cat_masks_tensor is not None:
        seq_len = x_adv.shape[1]
        cat_masks_expanded = cat_masks_tensor.unsqueeze(1).expand(-1, seq_len, -1)
        x_adv_proj = torch.where(cat_masks_expanded, x_clean, x_adv_proj)
    return x_adv_proj

def get_smoothed_prediction(pt_model, reg_model, Xs_context, ys_context, Xs_query, ys_query, train_size, cat_masks, ds, n_samples=50, sigma=0.5):
    """
    Monte Carlo Randomized Smoothing.
    Injects isotropic Gaussian noise into the context matrix, querying the model N times,
    and returns the majority vote class.
    """
    device = Xs_context.device
    votes = []
    
    pt_model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            # Generate Gaussian Noise for the context rows only
            noise = torch.randn_like(Xs_context) * sigma
            
            # Categorical constraints: Do not add noise to categorical columns
            seq_len = Xs_context.shape[1]
            cat_masks_expanded = cat_masks.unsqueeze(1).expand(-1, seq_len, -1)
            noise = torch.where(cat_masks_expanded, torch.zeros_like(noise), noise)
            
            noisy_context = Xs_context + noise
            
            # Reconstruct the full sequence: [Noisy Context, Clean Query]
            Xs_full = torch.cat([noisy_context, Xs_query], dim=1)
            ys_full = torch.cat([ys_context, ys_query], dim=1)
            
            # Predict
            train_size_tensor = torch.full((Xs_full.shape[0],), train_size, dtype=torch.int32, device=device)
            out = pt_model(Xs_full, ys_full, train_size_tensor, cat_mask=cat_masks, d=ds)
            
            # Extract logits for the query row
            logits = out[:, train_size:, :reg_model.n_classes_].squeeze(-1).mean(dim=0)
            pred = logits.argmax(dim=1)
            votes.append(pred)
            
    # Stack votes and take the mode (majority vote)
    votes = torch.stack(votes, dim=0) # [n_samples, query_size]
    smoothed_preds, _ = torch.mode(votes, dim=0)
    return smoothed_preds

def run_certified_robustness():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Certified Robustness via Randomized Smoothing ({device}) ---")
    
    # We use Breast Cancer because Certified Robustness is best defined for Classification
    from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
    from tabfm import TabFMClassifier
    
    cancer = load_breast_cancer(as_frame=True)
    X = cancer.frame.drop('target', axis=1)
    y = pd.Series(LabelEncoder().fit_transform(cancer.frame['target']))
    
    train_size = 50
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=10, random_state=42)
    
    pt_model = tabfm_v1_0_0.load(model_type='classification', device=str(device), dtype=torch.float32).to(device)
    reg = TabFMClassifier(model=pt_model)
    reg.fit(X_train, y_train)
    
    X_train_transformed = reg.X_encoder_.transform(X_train)
    bounds_min = torch.tensor(X_train_transformed.min(axis=0), dtype=torch.float32, device=device)
    bounds_max = torch.tensor(X_train_transformed.max(axis=0), dtype=torch.float32, device=device)
    
    X_test_transformed = reg.X_encoder_.transform(X_test)
    data = reg.ensemble_generator_.transform(X_test_transformed)
    Xs_all, ys_all, cat_masks_all, ds_all, _ = reg.ensemble_generator_.prepare_ensemble_tensors(data)
    
    Xs_tensor = torch.tensor(Xs_all, dtype=torch.float32, device=device)
    cat_masks_tensor = torch.tensor(cat_masks_all, dtype=torch.bool, device=device)
    ds_tensor = torch.tensor(ds_all, dtype=torch.int32, device=device)
    ys_tensor = F.pad(torch.tensor(ys_all, dtype=torch.float32, device=device), (0, Xs_tensor.shape[1] - ys_all.shape[1]), value=-100.0)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long, device=device)
    train_size_tensor = torch.full((Xs_tensor.shape[0],), train_size, dtype=torch.int32, device=device)
    
    # 1. Baseline Evaluation
    with torch.no_grad():
        out = pt_model(Xs_tensor[:2], ys_tensor[:2], train_size_tensor[:2], cat_mask=cat_masks_tensor[:2], d=ds_tensor[:2])
        logits = out[:, train_size:, :reg.n_classes_].squeeze(-1).mean(dim=0)
        clean_acc = (logits.argmax(dim=1) == y_test_tensor).float().mean().item()
    print(f"Standard Model (Clean Context) Accuracy: {clean_acc*100:.1f}%")
    
    # 2. Poison the Context
    epsilon = 0.05 * (bounds_max - bounds_min)
    alpha = epsilon / 5
    Xs_adv = Xs_tensor.clone().detach().requires_grad_(True)
    pt_model.eval()
    
    print("Generating Poisoned Context (30 steps)...")
    for step in range(30):
        out = pt_model(Xs_adv[:2], ys_tensor[:2], train_size_tensor[:2], cat_mask=cat_masks_tensor[:2], d=ds_tensor[:2])
        logits = out[:, train_size:, :reg.n_classes_].squeeze(-1).mean(dim=0)
        loss = -F.cross_entropy(logits, y_test_tensor)
        
        grad = torch.autograd.grad(loss, Xs_adv)[0]
        Xs_adv_data = Xs_adv - alpha * grad.sign()
        
        context_adv = project_schema_constraints(Xs_adv_data[:, :train_size, :], Xs_tensor[:, :train_size, :], epsilon, bounds_min, bounds_max, cat_masks_tensor)
        Xs_adv_data[:, :train_size, :] = context_adv
        Xs_adv_data[:, train_size:, :] = Xs_tensor[:, train_size:, :]
        Xs_adv = Xs_adv_data.clone().detach().requires_grad_(True)
        torch.cuda.empty_cache()
        
    with torch.no_grad():
        out = pt_model(Xs_adv[:2], ys_tensor[:2], train_size_tensor[:2], cat_mask=cat_masks_tensor[:2], d=ds_tensor[:2])
        logits = out[:, train_size:, :reg.n_classes_].squeeze(-1).mean(dim=0)
        adv_acc = (logits.argmax(dim=1) == y_test_tensor).float().mean().item()
    print(f"Standard Model (Poisoned Context) Accuracy: {adv_acc*100:.1f}%")
    
    # 3. Randomized Smoothing Defense
    print(f"\nApplying Randomized Smoothing Defense (N=50 Monte Carlo passes, sigma=0.5)...")
    Xs_adv_context = Xs_adv[:2, :train_size, :]
    ys_context = ys_tensor[:2, :train_size]
    Xs_query = Xs_tensor[:2, train_size:, :] # Query remains clean
    ys_query = ys_tensor[:2, train_size:]
    
    smoothed_preds = get_smoothed_prediction(
        pt_model, reg, Xs_adv_context, ys_context, Xs_query, ys_query, 
        train_size, cat_masks_tensor[:2], ds_tensor[:2], n_samples=50, sigma=0.5
    )
    
    smoothed_acc = (smoothed_preds == y_test_tensor).float().mean().item()
    print(f"SMOOTHED Model (Poisoned Context) Accuracy: {smoothed_acc*100:.1f}%")

if __name__ == "__main__":
    run_certified_robustness()
