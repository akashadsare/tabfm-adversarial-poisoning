import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory for figures
out_dir = "/home/ghost/Desktop/Research/tabfm_paper/figures"
os.makedirs(out_dir, exist_ok=True)

# Set style for academic papers
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.autolayout': True
})

def plot_targeted_attack_progression():
    """Plots the MSE progression over PGD steps."""
    steps = np.arange(0, 100, 10)
    
    # Data from Colab logs
    mse_target = [8.3105, 5.7703, 5.7130, 5.5465, 7.0738, 5.1649, 4.5613, 6.9750, 4.6925, 4.0576]
    mse_true = [0.7114, 2.1494, 1.9602, 2.0756, 2.0895, 1.8937, 2.1360, 1.9069, 2.0677, 2.2740]
    
    plt.figure(figsize=(6, 4))
    plt.plot(steps, mse_target, marker='o', label='MSE to Target (5.0)', color='red', linewidth=2)
    plt.plot(steps, mse_true, marker='s', label='MSE to True Labels', color='blue', linewidth=2)
    
    plt.xlabel('PGD Steps')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Targeted Context Poisoning Progression')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(os.path.join(out_dir, 'attack_progression.pdf'), format='pdf', dpi=300)
    plt.close()

def plot_defense_subsampling():
    """Plots the effect of context subsampling on Clean vs Poisoned data."""
    drop_rates = [0, 10, 30, 50, 75]
    
    # Data from Colab logs
    clean_true_mse = [0.7380, 0.7526, 0.4927, 0.5534, 1.2053]
    poisoned_target_mse = [5.4530, 5.8524, 6.3731, 7.6831, 11.0201]
    
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Context Drop Rate (%)')
    ax1.set_ylabel('Clean Context (True MSE)', color=color1)
    ax1.plot(drop_rates, clean_true_mse, marker='s', color=color1, linewidth=2, label='Clean (True MSE)')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Highlight the 'sweet spot' for defense
    ax1.axvspan(25, 55, alpha=0.2, color='green', label='Optimal Defense Zone')
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Poisoned Context (Target MSE)', color=color2)
    ax2.plot(drop_rates, poisoned_target_mse, marker='o', color=color2, linewidth=2, label='Poisoned (Target MSE)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Effect of Context Subsampling Defense')
    fig.tight_layout()
    
    plt.savefig(os.path.join(out_dir, 'defense_subsampling.pdf'), format='pdf', dpi=300)
    plt.close()

def plot_ablations():
    """Plots Epsilon Budget and Context Size ablations."""
    # Epsilon Ablation
    epsilons = [1, 5, 10, 20]
    eps_mse = [2.0950, 2.0616, 2.0438, 2.0267]
    baseline_mse = 0.7380
    
    plt.figure(figsize=(6, 4))
    plt.plot(epsilons, eps_mse, marker='o', color='purple', linewidth=2, label='Adversarial MSE')
    plt.axhline(y=baseline_mse, color='black', linestyle='--', label='Clean Baseline MSE')
    plt.xlabel('Epsilon Budget (%)')
    plt.ylabel('Mean Squared Error')
    plt.title('Ablation: Epsilon vs. Vulnerability')
    plt.xticks(epsilons)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(out_dir, 'ablation_epsilon.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Context Size Ablation
    context_sizes = [20, 50, 100, 200]
    ctx_mse = [1.9724, 1.8328, 2.1454, 1.8524]
    
    plt.figure(figsize=(6, 4))
    plt.plot(context_sizes, ctx_mse, marker='s', color='darkorange', linewidth=2, label='Adversarial MSE')
    plt.axhline(y=baseline_mse, color='black', linestyle='--', label='Clean Baseline MSE')
    plt.xlabel('Context Window Size (Rows)')
    plt.ylabel('Mean Squared Error')
    plt.title('Ablation: Context Size vs. Vulnerability')
    plt.xticks(context_sizes)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(out_dir, 'ablation_context.pdf'), format='pdf', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_targeted_attack_progression()
    plot_defense_subsampling()
    plot_ablations()
    print(f"Figures successfully generated in: {out_dir}")
