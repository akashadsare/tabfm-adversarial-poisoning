import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
from tabfm import TabFMClassifier

def project_schema_constraints_mixed(x_adv, x_clean, epsilon, bounds_min, bounds_max, cat_masks):
    """
    Projects continuous variables to L-infinity and schema bounds.
    Leaves categorical variables completely untouched since they are non-differentiable 
    integers in TabFM's input space.
    """
    # 1. L-infinity projection
    x_adv_proj = torch.maximum(torch.minimum(x_adv, x_clean + epsilon), x_clean - epsilon)
    
    # 2. Schema bounding
    x_adv_proj = torch.maximum(x_adv_proj, bounds_min)
    x_adv_proj = torch.minimum(x_adv_proj, bounds_max)
    
    # 3. Restore categorical columns to their exact clean values
    # cat_masks is [Batch, Features]. We expand it to match [Batch, SeqLen, Features]
    seq_len = x_adv.shape[1]
    cat_masks_expanded = cat_masks.unsqueeze(1).expand(-1, seq_len, -1)
    
    # Where cat_mask is True (Categorical), use x_clean. Where False (Continuous), use x_adv_proj.
    x_adv_final = torch.where(cat_masks_expanded, x_clean, x_adv_proj)
    
    return x_adv_final

def run_experiment():
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available! Colab is running on CPU.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("1. Loading Adult Census Dataset (Mixed Categorical/Continuous)...")
    # Fetch adult dataset (classification)
    adult = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
    df = adult.frame.dropna() # Drop NaNs for simplicity in this baseline
    
    # Downsample for quick Colab iteration
    df = df.sample(n=1000, random_state=42)
    
    X = df.drop('class', axis=1)
    y_raw = df['class']
    
    # Encode target to 0/1 integers
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw))
    
    train_size = 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=10, random_state=42)
    
    print("2. Loading TabFM PyTorch Model to GPU (Float32)...")
    pt_model = tabfm_v1_0_0.load(model_type="classification", device=str(device), dtype=torch.float32)
    pt_model = pt_model.to(device)
    
    reg = TabFMClassifier(model=pt_model)
    
    print("3. Evaluating Baseline Zero-Shot Performance...")
    reg.fit(X_train, y_train)
    baseline_preds = reg.predict_proba(X_test)
    
    # Calculate baseline Cross-Entropy Loss and Accuracy
    # Predict returns [Samples, Classes]
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long, device=device)
    baseline_preds_tensor = torch.tensor(baseline_preds, dtype=torch.float32, device=device)
    baseline_loss = F.cross_entropy(baseline_preds_tensor, y_test_tensor).item()
    baseline_acc = (baseline_preds_tensor.argmax(dim=1) == y_test_tensor).float().mean().item()
    
    print(f"--> Baseline CE Loss: {baseline_loss:.4f} | Accuracy: {baseline_acc*100:.2f}%")

    print("4. Preparing Mixed Tensors for White-Box Attack...")
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
    
    seq_len = Xs_tensor.shape[1]
    if ys_tensor.shape[1] < seq_len:
        ys_tensor = F.pad(ys_tensor, (0, seq_len - ys_tensor.shape[1]), value=-100.0)
    
    batch_size_pt = Xs_tensor.shape[0]
    train_size_tensor = torch.full((batch_size_pt,), train_size, dtype=torch.int32, device=device)
    
    # UNTARGETED ATTACK: Maximize Cross Entropy against True Labels
    epsilon = 0.05 * (bounds_max - bounds_min)
    alpha = epsilon / 5
    num_steps = 100
    batch_size_ensemble = 2 
    num_ensemble = Xs_tensor.shape[0]
    
    Xs_adv = Xs_tensor.clone().detach().requires_grad_(True)
    
    print(f"\n5. Running PGD Continuous Subspace Attack on Mixed Data...")
    pt_model.eval()
    for step in range(num_steps):
        idx = torch.randperm(num_ensemble, device=device)[:batch_size_ensemble]
        
        Xs_adv_batch = Xs_adv[idx]
        ys_batch = ys_tensor[idx]
        cat_masks_batch = cat_masks_tensor[idx]
        ds_batch = ds_tensor[idx]
        train_size_batch = train_size_tensor[idx]
        
        out = pt_model(Xs_adv_batch, ys_batch, train_size_batch, cat_mask=cat_masks_batch, d=ds_batch)
        
        # Classification output processing
        out = out[:, train_size:, :reg.n_classes_]
        logits = out.squeeze(-1)
        
        # Average ensemble logits (LogSumExp is better, but mean is an acceptable approximation for PGD)
        logits_mean = logits.mean(dim=0)
        
        # Untargeted: Maximize CE Loss (Minimize negative CE)
        loss = -F.cross_entropy(logits_mean, y_test_tensor)
        
        grad = torch.autograd.grad(loss, Xs_adv)[0]
        
        # The gradient for categorical columns will be 0 internally due to .long() casting in TabFM.
        # But we also enforce it strictly in our projection.
        Xs_adv_data = Xs_adv - alpha * grad.sign()
        
        context_adv = Xs_adv_data[:, :train_size, :]
        context_clean = Xs_tensor[:, :train_size, :]
        
        context_adv_projected = project_schema_constraints_mixed(
            context_adv, context_clean, epsilon, bounds_min, bounds_max, cat_masks_tensor
        )
        
        Xs_adv_data[:, :train_size, :] = context_adv_projected
        Xs_adv_data[:, train_size:, :] = Xs_tensor[:, train_size:, :]
        
        Xs_adv = Xs_adv_data.clone().detach().requires_grad_(True)
        
        if step % 10 == 0:
            current_loss = F.cross_entropy(logits_mean, y_test_tensor).item()
            current_acc = (logits_mean.argmax(dim=1) == y_test_tensor).float().mean().item()
            print(f"Step {step:03d} | CE Loss: {current_loss:.4f} | Accuracy: {current_acc*100:.2f}%")
            
        torch.cuda.empty_cache()
            
    print("\n--- Attack Complete ---")
    
if __name__ == "__main__":
    run_experiment()
