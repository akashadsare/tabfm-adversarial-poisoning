import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

def project_schema_constraints(x_adv, x_clean, epsilon, bounds_min, bounds_max):
    """
    CRITIC FIX 1: Enforce tabular schema constraints using feature-specific bounds,
    preventing issues like setting negative longitudes to 0.
    """
    # 1. L-infinity projection
    x_adv = torch.maximum(torch.minimum(x_adv, x_clean + epsilon), x_clean - epsilon)
    
    # 2. Schema projection (feature-specific bounds)
    x_adv = torch.maximum(x_adv, bounds_min)
    x_adv = torch.minimum(x_adv, bounds_max)
    
    return x_adv

def pgd_context_poisoning_untargeted(model, X_train_clean, y_train, X_test, y_test_true, 
                                     bounds_min, bounds_max, num_steps=50, epsilon=0.1, alpha=0.01):
    """
    White-box UNTARGETED PGD attack on the In-Context rows (X_train).
    Goal: Modify X_train to maximize the error on X_test compared to the true labels.
    """
    # Clone the clean context rows
    X_train_adv = X_train_clean.clone().detach().requires_grad_(True)
    
    # Ensure test samples don't get gradients
    X_test = X_test.clone().detach()
    y_test_true = y_test_true.clone().detach()
    
    model.eval() # Model weights are frozen
    
    print("Starting Untargeted PGD Attack on Context...")
    for step in range(num_steps):
        # Forward pass: TabFM takes [Context, Target]
        # (Pseudo-code: we will use the actual TabFM forward signature here)
        predictions = model.forward_inference(
            context_x=X_train_adv, 
            context_y=y_train, 
            query_x=X_test
        )
        
        # CRITIC FIX 3: This is an untargeted attack, so we want to maximize the error 
        # relative to the true labels. We minimize negative MSE.
        loss = -F.mse_loss(predictions, y_test_true)
        
        # Calculate gradients
        grad = torch.autograd.grad(loss, X_train_adv)[0]
        
        # CRITIC FIX 2: Classic PGD update (no Adam optimizer momentum desync)
        # We step in the direction of the NEGATIVE gradient of the NEGATIVE loss 
        # (which maximizes the actual loss)
        X_train_adv = X_train_adv - alpha * grad.sign() 
        
        # Project back to valid tabular constraints
        X_train_adv = project_schema_constraints(
            X_train_adv, X_train_clean, epsilon, bounds_min, bounds_max
        )
        
        # Re-attach gradient tracking for the next iteration
        X_train_adv = X_train_adv.clone().detach().requires_grad_(True)
            
        if step % 10 == 0:
            # Print the actual positive MSE loss for readability
            real_loss = F.mse_loss(predictions, y_test_true)
            print(f"Step {step} | Adversarial Error (MSE): {real_loss.item():.4f}")
            
    return X_train_adv.detach()

if __name__ == "__main__":
    # Pseudo-execution pipeline
    print("1. Loading California Housing")
    print("2. Calculating valid bounds_min and bounds_max from dataset")
    print("3. Loading TabFM PyTorch Model")
    print("4. Executing Untargeted PGD Context Poisoning Attack")
