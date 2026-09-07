import os
import sys
import subprocess
import json

# This script generates a worker script, then runs it using OS-level subprocesses
# to completely bypass Jupyter Notebook's multiprocessing/CUDA initialization bugs.

worker_code = """
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
from tabfm import TabFMRegressor, TabFMClassifier

def project_schema_constraints(x_adv, x_clean, epsilon, bounds_min, bounds_max, cat_masks_tensor=None):
    x_adv_proj = torch.maximum(torch.minimum(x_adv, x_clean + epsilon), x_clean - epsilon)
    x_adv_proj = torch.maximum(x_adv_proj, bounds_min)
    x_adv_proj = torch.minimum(x_adv_proj, bounds_max)
    if cat_masks_tensor is not None:
        seq_len = x_adv.shape[1]
        cat_masks_expanded = cat_masks_tensor.unsqueeze(1).expand(-1, seq_len, -1)
        x_adv_proj = torch.where(cat_masks_expanded, x_clean, x_adv_proj)
    return x_adv_proj

dataset_name = sys.argv[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\\n--- Benchmarking {dataset_name} on {device} ---")

if dataset_name == "California Housing":
    housing = fetch_california_housing(as_frame=True)
    X = housing.frame.drop('MedHouseVal', axis=1)
    y = housing.frame['MedHouseVal']
    task_type = 'regression'
    task_label = 'Regression'
elif dataset_name == "Breast Cancer":
    cancer = load_breast_cancer(as_frame=True)
    X = cancer.frame.drop('target', axis=1)
    y = pd.Series(LabelEncoder().fit_transform(cancer.frame['target']))
    task_type = 'classification'
    task_label = 'Classification (Acc)'
elif dataset_name == "Diabetes":
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.frame.drop('target', axis=1)
    y = diabetes.frame['target']
    task_type = 'regression'
    task_label = 'Regression'

train_size = 50
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=10, random_state=42)

pt_model = tabfm_v1_0_0.load(model_type=task_type, device=str(device), dtype=torch.float32).to(device)
if task_type == 'regression':
    reg = TabFMRegressor(model=pt_model)
else:
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

if task_type == 'regression':
    ys_tensor = F.pad(torch.tensor(ys_all, dtype=torch.float32, device=device), (0, Xs_tensor.shape[1] - ys_all.shape[1]), value=-100.0)
    y_test_tensor = torch.tensor(reg.y_scaler_.transform(y_test.values.reshape(-1, 1)).flatten(), dtype=torch.float32, device=device)
else:
    ys_tensor = F.pad(torch.tensor(ys_all, dtype=torch.float32, device=device), (0, Xs_tensor.shape[1] - ys_all.shape[1]), value=-100.0)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long, device=device)

train_size_tensor = torch.full((Xs_tensor.shape[0],), train_size, dtype=torch.int32, device=device)
epsilon = 0.05 * (bounds_max - bounds_min)
alpha = epsilon / 5

with torch.no_grad():
    out = pt_model(Xs_tensor[:2], ys_tensor[:2], train_size_tensor[:2], cat_mask=cat_masks_tensor[:2], d=ds_tensor[:2])
    if task_type == 'regression':
        preds = out[:, train_size:, :].squeeze(-1).mean(dim=0)
        clean_loss = F.mse_loss(preds, y_test_tensor).item()
    else:
        logits = out[:, train_size:, :reg.n_classes_].squeeze(-1).mean(dim=0)
        clean_loss = (logits.argmax(dim=1) == y_test_tensor).float().mean().item()
        
Xs_adv = Xs_tensor.clone().detach().requires_grad_(True)
pt_model.eval()
for step in range(30):
    out = pt_model(Xs_adv[:2], ys_tensor[:2], train_size_tensor[:2], cat_mask=cat_masks_tensor[:2], d=ds_tensor[:2])
    if task_type == 'regression':
        preds = out[:, train_size:, :].squeeze(-1).mean(dim=0)
        loss = -F.mse_loss(preds, y_test_tensor)
    else:
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
    if task_type == 'regression':
        preds = out[:, train_size:, :].squeeze(-1).mean(dim=0)
        adv_loss = F.mse_loss(preds, y_test_tensor).item()
        c_clean_str = f"{clean_loss:.4f}"
        c_adv_str = f"{adv_loss:.4f}"
    else:
        logits = out[:, train_size:, :reg.n_classes_].squeeze(-1).mean(dim=0)
        adv_loss = (logits.argmax(dim=1) == y_test_tensor).float().mean().item()
        c_clean_str = f"{clean_loss*100:.1f}\\%"
        c_adv_str = f"{adv_loss*100:.1f}\\%"
        
result = (dataset_name, task_label, c_clean_str, c_adv_str)
with open("worker_result.json", "w") as f:
    json.dump(result, f)
"""

if __name__ == "__main__":
    with open("worker.py", "w") as f:
        f.write(worker_code)
        
    datasets = ["California Housing", "Breast Cancer", "Diabetes"]
    results = []
    
    for d in datasets:
        # Run worker as a completely isolated OS process
        # This bypasses Jupyter's multiprocessing bug and guarantees 100% VRAM is freed upon exit
        cmd = [sys.executable, "worker.py", d]
        proc = subprocess.run(cmd, capture_output=False) # allow it to print to Colab console
        
        if proc.returncode == 0:
            with open("worker_result.json", "r") as f:
                results.append(json.load(f))
        else:
            print(f"Error processing {d}")
            
    print("\n\n=== GENERATED LATEX TABLE ===")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Massive Multi-Dataset Vulnerability Benchmark ($\epsilon=0.05$)}")
    print(r"\begin{tabular}{llcc}")
    print(r"\toprule")
    print(r"\textbf{Dataset} & \textbf{Task} & \textbf{Clean Perf.} & \textbf{Poisoned Perf.} \\")
    print(r"\midrule")
    for r in results:
        print(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\label{tab:benchmark}")
    print(r"\end{table}")
