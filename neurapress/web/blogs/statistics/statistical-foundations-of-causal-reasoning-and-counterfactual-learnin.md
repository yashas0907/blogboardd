# Statistical Foundations of Causal Reasoning and Counterfactual Learning for Foundation Models  

*Published by the Statistics for AI Community*  

---  

## Introduction  

Foundation models—large‐scale language, vision, and multimodal networks—have transformed what AI systems can do, but their sheer size also obscures the *why* behind their predictions. Classical statistical learning explains **correlation** but not **causation**; yet many practical concerns (bias detection, robustness to distribution shift, interpretability) are fundamentally causal.  

This tutorial walks through the statistical underpinnings that enable **causal discovery**, **counterfactual data augmentation**, and **causal inference for bias mitigation** in the context of foundation models. We also discuss **sample‑complexity** and **generalization bounds** that guide how much data is needed when we fine‑tune models under causal constraints.  

The material assumes familiarity with basic probability, deep learning, and the notion of interventions in Pearl’s do‑calculus. Throughout, we highlight concrete algorithms, a reproducible code sketch, and recent peer‑reviewed work that grounds the theory in practice.

---  

## 1. Causal Discovery from Model Activations  

### 1.1 Why look at activations?  

Activations (the hidden‑state vectors produced by each layer) can be treated as **high‑dimensional observational variables**. If a subset of neurons consistently mediates the effect of an input feature on the output, a causal graph over activations may reveal *mechanistic pathways* inside the black box.  

Formally, let  

\[
\mathbf{X} = (X_1,\dots,X_p) \quad\text{= input tokens or image patches},\\
\mathbf{H}^{(l)} = (H^{(l)}_1,\dots,H^{(l)}_{d_l}) \quad\text{= activations of layer }l,\\
Y \quad\text{= model output (logits, class, etc.).}
\]

We aim to learn a directed acyclic graph (DAG) \(\mathcal{G}\) over \(\{\mathbf{X},\mathbf{H}^{(l)},Y\}\) such that  

\[
P(Y \mid \operatorname{do}(\mathbf{X}=x)) = P(Y \mid \mathbf{X}=x,\operatorname{Pa}(Y;\mathcal{G})),
\]  

where \(\operatorname{Pa}(Y;\mathcal{G})\) denotes the parents of \(Y\) in the graph.

### 1.2 Statistical assumptions  

| Assumption | Description | Typical verification |
|------------|-------------|----------------------|
| **Causal Markov Condition** | Each node is independent of its non‑descendants given its parents. | Check conditional independencies via kernel‑based CI tests (e.g., HSIC). |
| **Faithfulness** | All conditional independencies in the data are entailed by the DAG. | Look for near‑zero partial correlations; if violated, consider latent confounders. |
| **No hidden confounders (or accounted for)** | All common causes of observed variables are measured. | Use auxiliary variables (e.g., attention maps) or latent‑variable methods (e.g., FCI). |

### 1.3 Algorithmic pipeline  

1. **Collect activation dataset**  
   * Sample a large, diverse set of inputs \(\{x^{(i)}\}_{i=1}^n\).  
   * Record \(\mathbf{H}^{(l)}\) for a chosen layer (or multiple layers).  

2. **Preprocess**  
   * Center and optionally whiten activations.  
   * Reduce dimensionality (PCA, random projection) to keep the CI tests tractable.  

3. **Apply a causal discovery method**  
   * Score‑based (e.g., GES, NOTEARS) or constraint‑based (PC, FCI).  
   * For differentiable models, NOTEARS‑style continuous optimization works well because the loss can be expressed as a smooth function of the weighted adjacency matrix \(W\).  

4. **Validate**  
   * Perform **intervention experiments**: zero‑out a candidate parent neuron and observe the effect on downstream activations or output.  
   * Compare discovered edges with known architectural connections (e.g., residual links) to sanity‑check.

### 1.4 Example: Discovering a causal subgraph in a BERT encoder  

Below is a minimal, end‑to‑end PyTorch sketch that (i) extracts activations from the 8‑th transformer layer, (ii) reduces dimensionality, and (iii) runs NOTEARS to obtain a weighted adjacency matrix.  

```python
import torch
from transformers import BertModel, BertTokenizer
import numpy as np
from sklearn.decomposition import PCA
from causal_nonlinear import notears_linear  # from the causal-learn repo

# 1️⃣ Load model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.eval()

# 2️⃣ Sample inputs
sentences = [
    "The cat sat on the mat.",
    "A lawyer argued the case.",
    "The sun rises in the east.",
    # … (thousands more)
]
inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)

# 3️⃣ Forward pass – capture activations of layer 8
with torch.no_grad():
    outputs = model(**inputs)
    # hidden_states is a tuple (layer0,…,layer12); layer8 has shape (batch, seq_len, dim)
    h8 = outputs.hidden_states[8]               # (B, L, D)
    # Collapse sequence dimension by averaging token representations
    h8_mean = h8.mean(dim=1).cpu().numpy()      # (B, D)

# 4️⃣ Dimensionality reduction (retain 95 % variance)
pca = PCA(n_components=0.95, svd_solver='full')
H_reduced = pca.fit_transform(h8_mean)          # (B, d) with d << D

# 5️⃣ Run NOTEARS (linear version for illustration)
W_est = notears_linear(H_reduced, lambda1=0.01)   # shape (d, d)

# 6️⃣ Inspect strongest edges
threshold = np.percentile(np.abs(W_est), 99)
strong_edges = np.argwhere(np.abs(W_est) > threshold)
print("Top causal edges (node_i → node_j):")
for i, j in strong_edges:
    print(f"{i} → {j}   weight={W_est[i, j]:.3f}")
```

*Key points*  

* The `notears_linear` function solves a continuous optimization problem with an acyclicity constraint \(\mathrm{tr}\big(e^{W \circ W}\big) - d = 0\).  
* The resulting matrix \(W_{\text{est}}\) can be thresholded to obtain a sparse DAG that approximates causal influence among reduced‑dimensional activation components.  

---  

## 2. Counterfactual Data Augmentation and Inference  

### 2.1 Counterfactual reasoning in foundation models  

A **counterfactual** asks: *What would the model output have been if some latent variable had taken a different value?* In Pearl’s notation, we compute  

\[
P(Y \mid \operatorname{do}(X=x'),\; \mathbf{Z}=z),
\]  

where \(X\) is the intervened variable (e.g., gender token) and \(\mathbf{Z}\) are observed covariates. Counterfactuals are powerful for two reasons  

* **Data augmentation**: synthetic examples that fill gaps in under‑represented subpopulations.  
* **Inference**: estimating causal effects (e.g., average treatment effect) directly from the model’s internal representation.

### 2.2 Formal pipeline  

| Step | Description | Statistical tool |
|------|-------------|------------------|
| **(a) Identify intervenable variables** | Use the causal graph from Section 1 to locate *parent* nodes of the output that correspond to high‑level concepts (e.g., gender, sentiment). | Graph‑based variable selection. |
| **(b) Define structural equations** | Approximate the mapping \(f_X\) from latent code to activation using a shallow regressor (e.g., linear probe). | Regression, representation learning. |
| **(c) Perform do‑intervention** | Replace the activation of the target node with the value generated by the structural equation under a new setting \(x'\). | Soft intervention: \(H^{\text{new}} = H^{\text{old}} + \Delta\). |
| **(d) Decode the counterfactual** | Run the downstream layers with the intervened activation to obtain a new output. | Forward pass with frozen weights. |
| **(e) Augment training set** | Pair the original input with the counterfactual output as an additional labeled example. | Standard supervised learning. |

### 2.3 A concrete example: Gender‑balanced sentiment classification  

Suppose we have a sentiment classifier built on top of a pretrained RoBERTa model. We want to ensure that sentiment predictions are **invariant** to gendered language.  

```python
import torch.nn as nn
from causal_nonlinear import linear_probe  # hypothetical helper

# Assume `model` is the frozen RoBERTa encoder from the previous section
# and `classifier` is a linear head trained on sentiment data.

# 1️⃣ Train a probe that predicts the gender embedding from layer‑8 activations
gender_labels = torch.tensor([0, 1, 0, 1, ...])   # 0 = male, 1 = female (from metadata)
probe = linear_probe(input_dim=H_reduced.shape[1], output_dim=1)
optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

for epoch in range(5):
    pred = probe(torch.from_numpy(H_reduced))
    loss = nn.BCEWithLogitsLoss()(pred.squeeze(), gender_labels.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 2️⃣ Generate counterfactual activations by flipping the gender probe output
with torch.no_grad():
    gender_pred = torch.sigmoid(probe(torch.from_numpy(H_reduced))).squeeze()
    # Flip the predicted gender probability (0→1, 1→0) and compute the required delta
    delta = (1.0 - gender_pred) - gender_pred   # simple linear approximation
    H_cf = H_reduced + delta.unsqueeze(1).numpy() * probe.weight.detach().numpy()

# 3️⃣ Decode counterfactual logits
def decode_from_reduced(H_red):
    # Inverse PCA → full‑dim activation → forward through remaining layers
    H_full = pca.inverse_transform(H_red)
    # Insert back into the model at layer 8 (requires custom forward hook)
    # For brevity, assume `model_forward_with_custom_h8` does this.
    return model_forward_with_custom_h8(H_full)

logits_cf = decode_from_reduced(H_cf)

# 4️⃣ Build augmented dataset
aug_inputs = inputs   # original tokenized inputs
aug_labels = torch.argmax(logits_cf, dim=-1)   # pseudo‑labels from counterfactuals
```

*Why this works*  

* The probe isolates a **latent gender factor** inside the activation space.  
* By **softly intervening** on that factor (adding `delta`), we generate a version of the same sentence where gender is flipped while preserving all other semantics.  
* The resulting pseudo‑labels can be mixed with the original training data, encouraging the classifier to ignore gender cues.

### 2.4 Counterfactual inference for treatment effects  

When the downstream task is a **policy decision** (e.g., recommending a job posting), we may wish to estimate the **individual treatment effect (ITE)** of a protected attribute. Using the causal graph, the ITE can be expressed as  

\[
\text{ITE}(x) = \mathbb{E}[Y \mid \operatorname{do}(A=1), X=x] -
               \mathbb{E}[Y \mid \operatorname{do}(A=0), X=x],
\]  

where \(A\) is the attribute (e.g., gender). The same intervention machinery described above yields the two expectations, which can be averaged over a test set to obtain the **average treatment effect (ATE)**.

---  

## 3. Causal Inference for Bias Detection and Mitigation  

###