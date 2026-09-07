# Adversarial Context Poisoning in Tabular Foundation Models

**Author:** Akash Adsare (Sinhgad Institute of Technology Lonavala)  
**Paper:** [Adversarial Context Poisoning in Tabular Foundation Models: Vulnerabilities and Zero-Cost Defenses](main.pdf)

## Overview
This repository contains the official code, LaTeX manuscript, and experimental data for exploring adversarial vulnerabilities in Tabular Foundation Models (TabFMs). Specifically, this research demonstrates how In-Context Learning (ICL) in models like **TabFM** and **TabPFN** can be completely shattered by injecting imperceptible $L_\infty$ bounded noise strictly into the continuous feature subspace of the context window.

### Key Contributions
1. **Continuous Subspace Poisoning:** A novel PGD-based adversarial attack that preserves categorical schemas while hijacking the Set Transformer's attention mechanism.
2. **Stochastic Ensemble Micro-Batching:** A CUDA optimization algorithm that allows for end-to-end differentiable attacks on massive 32-way ensemble Foundation Models using less than 15GB of VRAM.
3. **Zero-Cost Defenses (Context Subsampling):** We prove that randomly dropping 30-50% of the in-context rows completely breaks the adversarial gradient alignment, outperforming standard KNN sanitization.
4. **Theoretical Limitations:** We mathematically demonstrate that standard Certified Robustness (Randomized Smoothing via Monte Carlo Gaussian noise) completely fails on Tabular Foundation Models due to their extreme sensitivity to isotropic noise.

## Repository Structure
- `main.pdf`: The final compiled IEEE-format research paper.
- `main.tex`: Root LaTeX source code.
- `sections/`: Contains the modular LaTeX sections (Introduction, Methodology, Threat Model, Experiments, Defenses).
- `figures/`: Contains the generated graphs, saliency maps, and attack surface heatmaps.

## Experimental Results
Our Multi-Dataset Benchmark across standard UCI Datasets proves that TabFM is hyper-fragile to adversarial context shifts as small as $\epsilon = 5\%$. 

| Dataset | Task | Clean Perf. | Poisoned Perf. |
|---------|------|-------------|----------------|
| California Housing | Regression (MSE) | 0.5227 | 1.9691 |
| Breast Cancer | Classification (Acc) | 70.0% | 0.0% |
| Diabetes | Regression (MSE) | 0.3223 | 0.8920 |

### Explainability (Saliency Maps)
By extracting the gradient-based saliency maps from the Set Transformer, we visualize the exact mechanism of the adversarial collapse. The poisoned context completely hijacks the attention weights, forcing the model to fixate on localized, high-frequency adversarial artifacts rather than the true data manifold.
*(See `figures/saliency_comparison.png`)*

### 2D Attack Surface
We generated a comprehensive Hyperparameter Attack Surface Heatmap by sweeping the noise budget against the Context Window Size. Counterintuitively, scaling the prompt context up to 100 rows provides zero defensive benefit. The model collapses even at $\epsilon=1\%$.
*(See `figures/attack_surface.png`)*

## Citation
If you find this research useful, please cite our paper.

```bibtex
@article{adsare2026tabfm,
  title={Adversarial Context Poisoning in Tabular Foundation Models: Vulnerabilities and Zero-Cost Defenses},
  author={Adsare, Akash},
  journal={arXiv preprint},
  year={2026}
}
```
