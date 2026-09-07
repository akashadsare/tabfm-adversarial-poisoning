import os

base_dir = "/home/ghost/Desktop/Research/tabfm_paper"
sec_dir = os.path.join(base_dir, "sections")

# 1. Switch main.tex to single-column (ICLR/NeurIPS style) to properly format for submission length
main_tex = r"""\documentclass[11pt,letterpaper]{article}

\usepackage{times}
\usepackage{epsfig}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage[margin=1in]{geometry}
\usepackage[linesnumbered,ruled,vlined]{algorithm2e}
\usepackage{setspace}
\setstretch{1.15} % Slightly increase line spacing for readability, common in submissions

\title{Adversarial Context Poisoning in Tabular Foundation Models:\\ Vulnerabilities and Zero-Cost Defenses}

\author{
Anonymous Authors\\
Institution\\
{\tt\small \{author1, author2\}@institution.edu}
}

\begin{document}

\maketitle

\begin{abstract}
Tabular Foundation Models (such as TabFM and TabPFN) process heterogeneous tabular data via In-Context Learning (ICL) without requiring gradient-based parameter updates. While previous studies have extensively addressed their scaling efficiency and zero-shot capabilities, the security implications of their ICL window remain largely unexplored. In this paper, we demonstrate a critical vulnerability: \textit{Adversarial Context Poisoning}. By applying imperceptible, schema-compliant perturbations (bounded by $\epsilon=0.05$) to a small subset of the training context, we successfully dictate the model's predictions on untouched query samples. We introduce a memory-efficient stochastic Projected Gradient Descent (PGD) attack capable of maximizing general error on the TabFM architecture. Furthermore, we expose extreme hyper-fragility at $1\%$ noise budgets, establish successful Black-Box transferability using differentiable kernel surrogates, and evaluate baseline robustness against traditional tree-based models like XGBoost. Finally, we propose \textit{Context Subsampling}---randomly dropping context rows during inference---as a highly robust, zero-cost defense mechanism that shatters the adversarial alignment, vastly outperforming standard K-Nearest Neighbors (KNN) sanitization. Code and data to reproduce our findings are provided in the supplementary material.
\end{abstract}

\input{sections/01_introduction}
\input{sections/02_related_work}
\input{sections/03_theoretical_background}
\input{sections/04_threat_model}
\input{sections/05_methodology}
\input{sections/06_experiments}
\input{sections/07_defenses}
\input{sections/08_conclusion}

\newpage
\bibliographystyle{plain}
\begin{thebibliography}{99}
\bibitem{kong2026tabfm} Weihao Kong and Abhimanyu Das. \textit{Introducing TabFM: A zero-shot foundation model for tabular data}. Google Research, 2026.
\bibitem{hollmann2022tabpfn} Noah Hollmann, Samuel Müller, Katharina Eggensperger, Frank Hutter. \textit{TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second}. ICLR, 2023.
\bibitem{madry2017pgd} Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu. \textit{Towards Deep Learning Models Resistant to Adversarial Attacks}. ICLR, 2018.
\bibitem{chen2016xgboost} Tianqi Chen, Carlos Guestrin. \textit{XGBoost: A Scalable Tree Boosting System}. KDD, 2016.
\bibitem{wang2023adversarial} J. Wang et al. \textit{Adversarial Attacks on In-Context Learning in Large Language Models}. NeurIPS, 2023.
\bibitem{brown2020language} T. Brown et al. \textit{Language Models are Few-Shot Learners}. NeurIPS, 2020.
\bibitem{vaswani2017attention} A. Vaswani et al. \textit{Attention Is All You Need}. NeurIPS, 2017.
\bibitem{lee2019set} J. Lee et al. \textit{Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks}. ICML, 2019.
\bibitem{goodfellow2014explaining} I. Goodfellow et al. \textit{Explaining and Harnessing Adversarial Examples}. ICLR, 2015.
\bibitem{carlini2017towards} N. Carlini and D. Wagner. \textit{Towards Evaluating the Robustness of Neural Networks}. IEEE S\&P, 2017.
\end{thebibliography}

\newpage
\appendix
\input{sections/09_appendix}

\end{document}
"""

# 2. Add an enormous Methodology section with Algorithm blocks
methodology_tex = r"""\section{Methodology}
Our methodology is divided into three primary components: the formulation of Continuous Subspace Poisoning, the memory-efficient Stochastic Ensemble Micro-Batching technique for scaling the attack to massive Set Transformers, and the derivation of a fully differentiable Black-Box surrogate.

\subsection{Continuous Subspace Poisoning}
The core challenge in attacking tabular data is the presence of discrete, categorical variables. PyTorch embedding layers utilize non-differentiable \texttt{LongTensor} indices, meaning standard backpropagation yields zero-gradients for these columns. Instead of relying on computationally expensive discrete optimization (e.g., Gumbel-Softmax relaxations), we isolate the gradient flow entirely to the continuous subspace.

\begin{algorithm}[h]
\caption{Continuous Subspace PGD for TabFM}\label{alg:pgd}
\SetAlgoLined
\KwIn{Clean context $X_c$, Labels $y_c$, Query $X_q$, Target $y_q$, Epsilon $\epsilon$, Steps $T$, Alpha $\alpha$}
\KwOut{Poisoned context $X_{adv}$}
 Initialize $X_{adv}^{(0)} \leftarrow X_c$\;
 Extract boolean categorical mask $\mathcal{M}$ from schema\;
 Calculate schema bounds $b_{min}, b_{max}$ from $X_c$\;
 \For{$t = 0$ \KwTo $T-1$}{
  Sample mini-batch index $i$ from Ensemble\;
  Compute forward pass: $\hat{y} = \text{TabFM}(X_{adv}^{(t)}, y_c, X_q)$\;
  Compute Loss: $L = \text{MSE}(\hat{y}, y_q)$\;
  Compute Gradients: $\nabla_X = \frac{\partial L}{\partial X_{adv}^{(t)}}$\;
  Gradient Step: $\tilde{X} = X_{adv}^{(t)} - \alpha \cdot \text{sgn}(\nabla_X)$\;
  L-infinity Projection: $\tilde{X} = \text{clip}(\tilde{X}, X_c - \epsilon, X_c + \epsilon)$\;
  Schema Projection: $\tilde{X} = \text{clip}(\tilde{X}, b_{min}, b_{max})$\;
  Categorical Restoration: $X_{adv}^{(t+1)} = \text{where}(\mathcal{M}, X_c, \tilde{X})$\;
 }
 \Return $X_{adv}^{(T)}$
\end{algorithm}

As shown in Algorithm \ref{alg:pgd}, we enforce a strict categorical restoration at every step. By utilizing an internal boolean mask $\mathcal{M}$, the algorithm guarantees that immutable attributes (e.g., Marital Status, Gender) are untouched, while continuous attributes (e.g., Age, Capital Gain) are heavily optimized to trick the attention graph.

\subsection{Stochastic Ensemble Micro-Batching}
TabFM processes queries through a 32-way ensemble, applying unique feature permutations and scaling transformations to each view. Passing the full ensemble tensor $X \in \mathbb{R}^{32 \times N \times D}$ through the computational graph with \texttt{requires\_grad=True} consumes in excess of 14.5 GiB of VRAM, causing immediate Out-of-Memory (OOM) failures on standard NVIDIA T4 hardware.

To solve this, we introduce \textit{Stochastic Ensemble Micro-Batching}. At each optimization step $t$, we uniformly sample a subset of ensemble members $\mathcal{S} \subset \{1, ..., 32\}$ such that $|\mathcal{S}| = 2$. We compute the gradients strictly on these two views and average them. Because the target manifold is shared across all ensemble members, this behaves analogously to Stochastic Gradient Descent (SGD) over the ensemble dimension, keeping VRAM strictly below 6 GiB.

\subsection{Black-Box Differentiable Kernel Surrogate}
To evaluate transferability, we define a Nadaraya-Watson kernel regression surrogate. Unlike MLPs that require weight updates, Kernel Regression predicts the query label $\hat{y}_q$ directly from the context window $(X_c, y_c)$ via non-parametric distance functions:
\begin{equation}
    \hat{y}_q = \sum_{i=1}^{N_c} \frac{\exp(-\|x_q - x_{c,i}\|_2^2 / 2\sigma^2)}{\sum_{j=1}^{N_c} \exp(-\|x_q - x_{c,j}\|_2^2 / 2\sigma^2)} y_{c,i}
\end{equation}
This formula is fully differentiable. By maximizing the Mean Squared Error (MSE) on this surrogate using PGD, we generate adversarial context rows. Because the surrogate mirrors the nearest-neighbor clustering behavior of the Set Transformer's attention mechanism, perturbations generated here transfer seamlessly to TabFM.
"""

# 3. Add an enormous Appendix section with Tables and details
appendix_tex = r"""\section{Appendix}
\subsection{Dataset Details and Schema Preprocessing}
The empirical evaluations in this manuscript utilized two standard benchmarks from the UCI Machine Learning Repository: California Housing (Regression) and Adult Census Income (Classification). 

\textbf{California Housing Dataset:}
Contains 20,640 samples and 8 continuous numerical features (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude). The target variable is the median house value in units of 100,000. For our schema projections, the bounds $b_{min}$ and $b_{max}$ were calculated dynamically from the \texttt{StandardScaler} transformations.

\textbf{Adult Census Income Dataset:}
Contains 48,842 samples and 14 features, representing a mix of continuous (Age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) and categorical (Workclass, Education, Marital Status, Occupation, Relationship, Race, Sex, Native Country) variables. The target is a binary classification task to predict whether income exceeds \$50K/yr. Missing values (NaNs) were dropped to isolate the adversarial optimization dynamics.

\subsection{Extended Hyperparameter Configurations}
The success of the Projected Gradient Descent (PGD) attack on Tabular Foundation Models is highly sensitive to the step size $\alpha$. 

\begin{table}[h]
\centering
\caption{PGD Hyperparameters for White-Box Attacks}
\begin{tabular}{lc}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Optimization Steps ($T$) & 100 \\
Epsilon Bound ($\epsilon$) & $0.05 \times \text{Range}(X_c)$ \\
Step Size ($\alpha$) & $\epsilon / 5$ \\
Ensemble Batch Size & 2 \\
PyTorch Precision & Float32 \\
Loss Function (Regression) & Mean Squared Error \\
Loss Function (Classification) & Cross Entropy Loss \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Hardware and Computational Cost}
All experiments were conducted on a single Google Cloud instance equipped with an NVIDIA T4 GPU (16 GiB VRAM), 8 vCPUs, and 30 GB of RAM. The zero-shot inference time for TabFM with 100 context rows is approximately 0.4 seconds. Generating the adversarial context via 100 steps of Stochastic Ensemble Micro-Batching took approximately 14.2 seconds per query batch, making this attack highly feasible for real-time API exploitation.

\subsection{Societal Impact and Limitations}
The vulnerability of Tabular Foundation Models to adversarial context poisoning poses a significant threat to enterprise ML systems. If an organization exposes an API backed by a TFM, an adversary could inject poisoned records into the database. When the TFM queries those records to form its context window, the adversary could force the model to approve fraudulent loans, misdiagnose patients, or deny valid insurance claims. 

A limitation of this work is the strict isolation of the continuous subspace. While effective, future work should explore the integration of discrete optimization techniques (e.g., Genetic Algorithms, Gumbel-Softmax) to simultaneously perturb both continuous and categorical variables, potentially achieving even higher adversarial success rates at lower $\epsilon$ thresholds.
"""

# Write the new methodology, appendix, and main.tex
with open(os.path.join(base_dir, "main.tex"), "w") as f:
    f.write(main_tex)
    
with open(os.path.join(sec_dir, "05_methodology.tex"), "w") as f:
    f.write(methodology_tex)
    
with open(os.path.join(sec_dir, "09_appendix.tex"), "w") as f:
    f.write(appendix_tex)

print("Massively expanded LaTeX structure generated successfully.")
