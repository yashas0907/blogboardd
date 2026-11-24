# Statistical Foundations of Explainability, Interpretability, and Attribution for Foundation Models  

*Published by the Statistical AI Review*  

---

## Introduction  

Foundation models—large‑scale language and vision transformers—are increasingly deployed in high‑stakes applications. Their sheer size and opaque internal dynamics raise pressing questions: **How can we rigorously quantify which inputs drive a model’s output?** **How certain are we about those attributions?** **Do the explanations themselves leak sensitive information or hide spurious patterns?**  

Answering these questions demands more than visual heatmaps; it requires **statistical theory** that treats explanations as estimators, confidence‑bearing objects, and hypothesis tests. In this tutorial we weave together four strands of recent research:

1. **Causal‑inference‑based attribution** for token‑level explanations.  
2. **Post‑hoc confidence intervals** for feature‑importance scores via bootstrap and Bayesian posterior sampling.  
3. **Hypothesis‑testing frameworks** for detecting spurious correlations in saliency maps, including a full permutation‑test pipeline.  
4. **Trade‑offs** among interpretability fidelity, privacy leakage, and downstream utility, expressed through formal statistical risk bounds.  

Each section presents the underlying statistical model, practical estimation strategies for massive foundation models, and illustrative results that highlight strengths and limitations. The goal is to equip researchers and practitioners with tools that are **provably sound**, **scalable**, and **transparent**.

---

## 1. Causal‑Inference‑Based Attribution for Token‑Level Explanations  

### 1.1 From Correlation to Causation  

Traditional attribution methods (e.g., gradient‑based saliency, LIME) treat the model as a black‑box function \(f\) and measure **associative** influence of input tokens. Causal inference reframes the problem: we ask *what would the model output be if a particular token were intervened upon?*  

Formally, let a text input be a sequence \(X = (x_1,\dots,x_T)\). For a target token \(x_t\), define the **do‑intervention** \(do(x_t = \tilde{x}_t)\) that replaces \(x_t\) with a counterfactual token \(\tilde{x}_t\) while leaving the rest of the context unchanged. The **causal effect** on a scalar output \(Y = f(X)\) (e.g., probability of a class) is  

\[
\text{CE}_{t} = \mathbb{E}[Y \mid do(x_t = \tilde{x}_t)] - \mathbb{E}[Y \mid do(x_t = x_t)] .
\]

When the model is deterministic, the expectation collapses to a single forward pass, but stochastic components (e.g., dropout, sampling from a language model) preserve the need for averaging.

### 1.2 Estimating Causal Effects with the Potential Outcomes Framework  

Directly evaluating the do‑operator is infeasible for large vocabularies. Instead we adopt the **potential outcomes** approach (Rubin, 1974) and treat each token replacement as a treatment. Let \(A_t\) be a binary indicator of whether token \(t\) is replaced (treatment = 1) or left untouched (control = 0). The observed outcome is  

\[
Y = A_t \, Y(1) + (1-A_t) \, Y(0),
\]

where \(Y(1)\) and \(Y(0)\) are the potential outputs under treatment and control, respectively. The **average treatment effect (ATE)** for token \(t\) is  

\[
\text{ATE}_t = \mathbb{E}[Y(1) - Y(0)] .
\]

Because we cannot observe both potentials simultaneously, we estimate ATE using **inverse‑propensity weighting (IPW)** or **doubly‑robust (DR)** estimators:

* **Propensity model** \(e_t = \Pr(A_t=1 \mid X_{-t})\) is learned with a lightweight classifier that predicts how likely a token would be replaced under a random perturbation scheme (e.g., masked language modeling).  
* **Outcome model** \(\hat{m}_t(x) = \mathbb{E}[Y \mid X = x, A_t = a]\) is approximated by a surrogate network trained on a modest set of intervention data.

The IPW estimator  

\[
\widehat{\text{ATE}}^{\text{IPW}}_t = \frac{1}{N}\sum_{i=1}^{N}\Bigl(\frac{A_{ti}Y_i}{e_{ti}} - \frac{(1-A_{ti})Y_i}{1-e_{ti}}\Bigr)
\]

is unbiased under correct propensity specification. The DR estimator adds a correction term that remains consistent if *either* the propensity *or* outcome model is correctly specified (Bang & Robins, 2005).  

### 1.3 Counterfactual Token Removal in Practice  

A concrete workflow for a transformer‑based language model:

| Step | Description |
|------|-------------|
| 1 | Sample a set of sentences from the target domain. |
| 2 | For each token \(x_t\), generate a **masked** version \(\tilde{X}^{(t)}\) where the token is replaced by the [MASK] token (or a random synonym). |
| 3 | Run the model on both the original and masked inputs, recording the output distribution (e.g., logits for the target class). |
| 4 | Fit propensity scores using a logistic regression on token‑frequency features. |
| 5 | Compute DR ATEs for every token. |
| 6 | Visualize the signed ATEs as a heatmap aligned with the text. |

Empirically, causal ATEs often differ markedly from raw gradients: function words that receive high gradient magnitude may show negligible causal impact, while rare content words exhibit strong ATEs despite modest gradients (Janzing et al., 2020).

---

## 2. Post‑hoc Confidence Intervals for Feature Importance Scores  

### 2.1 Why Uncertainty Matters  

Feature‑importance scores (gradients, Integrated Gradients, SHAP values) are point estimates. In downstream decision‑making—e.g., medical report generation—knowing the **statistical uncertainty** of each attribution is essential. Two complementary approaches are widely used:

1. **Frequentist bootstrap** (Efron & Tibshirani, 1993).  
2. **Bayesian posterior sampling** over model parameters (Neal, 1996; Zhang et al., 2022).

### 2.2 Bootstrap Resampling of Model Predictions  

Given a fixed trained foundation model \(f_\theta\) and a dataset \(\{X_i\}_{i=1}^n\), we generate **B** bootstrap replicates:

1. Sample with replacement a set \(\mathcal{I}^{(b)} \subset \{1,\dots,n\}\).  
2. Compute the attribution vector \(\mathbf{a}^{(b)} = \text{Attr}(f_\theta, X_{\mathcal{I}^{(b)}})\) (e.g., Integrated Gradients per token).  
3. For each feature \(j\) (token position or image patch) collect \(\{a^{(b)}_j\}_{b=1}^B\) and form a **percentile confidence interval** \([a_j^{\alpha/2}, a_j^{1-\alpha/2}]\).  

Because the model parameters remain unchanged, this procedure captures **sampling variability** due to the data distribution, not model uncertainty. For large‑scale models, we can exploit **Monte‑Carlo dropout** (Gal & Ghahramani, 2016) to approximate bootstrap draws at a lower computational cost.

### 2.3 Bayesian Posterior Sampling over Model Weights  

When model uncertainty is non‑negligible (e.g., fine‑tuning on a small corpus), we place a prior \(p(\theta)\) over the weights and obtain the posterior \(p(\theta \mid \mathcal{D})\). Approximate sampling techniques include:

* **Hamiltonian Monte Carlo (HMC)** on a low‑dimensional adapter layer (e.g., LoRA).  
* **Stochastic Gradient Langevin Dynamics (SGLD)** applied to the full model (Welling & Teh, 2011).  

For each posterior draw \(\theta^{(s)}\), compute the attribution \(\mathbf{a}^{(s)} = \text{Attr}(f_{\theta^{(s)}}, X)\). The resulting **posterior credible interval** for feature \(j\) is the \((1-\alpha)\) highest‑density region of \(\{a^{(s)}_j\}_{s=1}^S\).

### 2.4 Practical Workflow for Foundation Models  

1. **Select a lightweight adapter** (e.g., 1 % of parameters) to keep sampling tractable.  
2. **Run HMC** for 2 000 iterations, discarding the first 500 as burn‑in, and thin every 5 steps → \(S = 300\) posterior samples.  
3. **Compute Integrated Gradients** for each sample on a held‑out sentence.  
4. **Summarize** with median attribution and 95 % credible intervals.  

Figure 1 (not shown) typically reveals that high‑magnitude attributions for rare tokens have **wide intervals**, signalling that the model’s confidence about those explanations is limited—a crucial insight for risk‑aware deployment.

---

## 3. Hypothesis‑Testing Frameworks for Detecting Spurious Correlations in Saliency Maps  

### 3.1 Defining the Null Hypothesis  

Given a saliency map \(\mathbf{S}\) (e.g., gradient magnitude per token), we want to test whether the observed alignment with a **ground‑truth relevance set** \(\mathcal{R}\) could arise by chance. The null hypothesis \(H_0\) is:

> *The saliency scores are independent of the true relevance; any overlap between high‑scoring tokens and \(\mathcal{R}\) is due to random assignment.*

A common test statistic is the **area under the precision‑recall curve (AUPRC)** when treating saliency scores as a ranking of tokens.

### 3.2 Permutation Test Procedure  

The permutation test constructs the null distribution by randomly shuffling relevance labels while preserving the saliency map:

1. **Compute observed statistic** \(T_{\text{obs}} = \text{AUPRC}(\mathbf{S}, \mathcal{R})\).  
2. **For \(b = 1,\dots,B\):**  
   * Randomly permute the binary relevance vector \(\mathcal{R}^{(b)}\).  
   * Compute \(T^{(b)} = \text{AUPRC}(\mathbf{S}, \mathcal{R}^{(b)})\).  
3. **Estimate p‑value**  
   \[
   \hat{p} = \frac{1}{B+1}\Bigl(1 + \sum_{b=1}^{B