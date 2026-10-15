# Statistical Foundations of Privacy‑Preserving Training and Inference for Foundation Models  

*An in‑depth tutorial on the theory, trade‑offs, and practical considerations that enable large language models (LLMs) to be trained and served while offering provable privacy guarantees.*

---

## 1. Introduction  

Foundation models—particularly large language models with billions of parameters—have become the workhorse of modern AI applications. Their impressive capabilities, however, raise **privacy** concerns because the training data often contain sensitive text (e.g., medical records, personal communications, proprietary code).  

Statistical privacy research provides a rigorous toolkit for answering the question: *How can we train or adapt such models without leaking information about any individual datum?* This tutorial surveys the **statistical underpinnings** that make privacy‑preserving training and inference possible, focusing on:

1. **Differential privacy (DP) mechanisms** and the utility‑privacy trade‑offs that arise for LLMs.  
2. **Statistical analysis of privacy leakage** during fine‑tuning and prompt adaptation.  
3. **Secure aggregation and federated learning** with formal DP guarantees.  
4. **Scaling laws and sample‑complexity** for privacy‑preserving training and inference.

The goal is to equip researchers and engineers with a clear mental model of the *probabilistic guarantees* that underlie modern privacy‑preserving AI pipelines, and to highlight open challenges at the intersection of **statistics**, **machine learning**, and **privacy law**.

---

## 2. Differential Privacy for Large Language Models  

### 2.1 Core Definition  

A randomized algorithm \(\mathcal{M}\) satisfies \((\varepsilon,\delta)\)-**Differential Privacy** if for any two neighboring datasets \(D,D'\) (differing in a single record) and any measurable set \(S\),

\[
\Pr[\mathcal{M}(D)\in S]\le e^{\varepsilon}\Pr[\mathcal{M}(D')\in S]+\delta .
\]

The parameters \(\varepsilon\) (privacy loss) and \(\delta\) (failure probability) quantify the *statistical indistinguishability* of the algorithm’s output when a single individual's data is added or removed.

### 2.2 DP‑SGD for Transformer Training  

The canonical method for training deep networks under DP is **DP‑Stochastic Gradient Descent (DP‑SGD)** (Abadi et al., 2016). The algorithm modifies each mini‑batch update as follows:

1. **Per‑example clipping**: For each example \(x_i\) compute the gradient \(g_i\) and clip its \(\ell_2\)-norm to a bound \(C\).  
2. **Noise addition**: Aggregate the clipped gradients and add Gaussian noise \(\mathcal{N}(0,\sigma^2 C^2 I)\).  
3. **Update**: Apply the noisy, clipped gradient to the model parameters.

Statistically, the noise scale \(\sigma\) determines the **privacy budget** consumed per iteration. Advanced composition theorems (Kairouz et al., 2021) and **privacy amplification by subsampling** (Balle et al., 2020) allow us to bound the total \((\varepsilon,\delta)\) after many steps.

### 2.3 Utility‑Privacy Trade‑offs for LLMs  

Training LLMs under DP incurs a **utility penalty** that manifests as higher perplexity or lower downstream task performance. The trade‑off is governed by three statistical levers:

| Lever | Effect on Privacy | Effect on Utility |
|------|-------------------|-------------------|
| **Clipping bound \(C\)** | Smaller \(C\) reduces sensitivity → lower \(\varepsilon\) | Over‑clipping can bias gradients, hurting convergence |
| **Noise scale \(\sigma\)** | Larger \(\sigma\) → stronger DP (smaller \(\varepsilon\)) | Excessive noise degrades model quality |
| **Batch size / sampling rate** | Lower sampling rate amplifies privacy (subsampling) | Smaller batches increase variance → slower learning |

Empirical studies (e.g., Liu et al., 2023; Bianchi et al., 2024) show that **parameter‑efficient fine‑tuning** (e.g., LoRA, adapter layers) dramatically reduces the number of trainable parameters, thereby lowering the effective sensitivity and enabling *tight* \((\varepsilon,\delta)\) budgets (e.g., \(\varepsilon\approx 3\) for 7‑B models) with modest utility loss.

### 2.4 Formal Guarantees for Inference  

Beyond training, **DP inference** can be achieved via **output perturbation** (adding noise to logits) or **private query answering** (e.g., the **PATE** framework, Papernot et al., 2018). The statistical analysis mirrors DP‑SGD: the added noise must be calibrated to the *global sensitivity* of the prediction function, which for a transformer is bounded by the softmax temperature and the Lipschitz constant of the final linear layer.

---

## 3. Statistical Analysis of Privacy Leakage During Fine‑Tuning & Prompt Adaptation  

### 3.1 Membership Inference as a Statistical Test  

**Membership inference attacks** (Shokri et al., 2017) test whether a specific record was present in the training set. Statistically, the attacker performs a hypothesis test:

- \(H_0\): Record not in training data.  
- \(H_1\): Record in training data.

The test statistic often involves the model’s loss on the target record. Under DP, the **privacy loss random variable** bounds the likelihood ratio between the two hypotheses, guaranteeing that the **type‑I** and **type‑II** error rates cannot be driven arbitrarily low (Dwork & Roth, 2014). Consequently, a model trained with \((\varepsilon,\delta)\)-DP satisfies:

\[
\Pr[\text{success} \mid H_1] \le e^{\varepsilon}\Pr[\text{success} \mid H_0] + \delta .
\]

### 3.2 Influence Functions and Gradient Leakage  

Fine‑tuning modifies a subset of parameters \(\theta\) based on a small dataset \(\mathcal{D}_{\text{ft}}\). **Influence functions** (Koh & Liang, 2017) approximate the change in loss on a test point \(z\) caused by upweighting a training point \(z_i\):

\[
\mathcal{I}_{\uparrow,z_i}(z) = -\nabla_\theta \ell(z;\theta)^\top H_\theta^{-1}\nabla_\theta \ell(z_i;\theta).
\]

Statistically, the magnitude of \(\mathcal{I}\) quantifies **information leakage**: large influence indicates that a single fine‑tuning example can be recovered from model behavior. DP‑SGD’s clipping and noise effectively regularize the influence, shrinking the norm of the inverse Hessian term and providing a bound on the maximal influence (Bassily et al., 2020).

### 3.3 Prompt‑Based Adaptation  

Prompt engineering (e.g., prefix‑tuning, soft prompts) treats the prompt as a set of learnable embeddings \(\mathbf{p}\). The **sensitivity** of the model output to a single fine‑tuning example is often lower than full‑parameter fine‑tuning because the prompt dimension \(d_p\) is small (typically a few hundred). Recent statistical analyses (Zhu et al., 2023) show that the **Rademacher complexity** of the prompt‑only hypothesis class scales as \(\mathcal{O}(\sqrt{d_p/n})\), implying that for a fixed privacy budget the *sample complexity* required to achieve a target utility is dramatically reduced.

---

## 4. Secure Aggregation and Federated Learning with Formal Privacy Guarantees  

### 4.1 Federated Learning Overview  

In **Federated Learning (FL)**, a central server orchestrates local model updates \(\Delta_i\) from \(K\) clients, aggregates them, and broadcasts the new global model. The canonical algorithm **FedAvg** (McMahan et al., 2017) computes:

\[
\Delta = \frac{1}{K}\sum_{i=1}^{K} \Delta_i .
\]

### 4.2 Secure Aggregation  

**Secure aggregation** (Bonawitz et al., 2017) cryptographically ensures that the server learns only the sum \(\Delta\) and not any individual \(\Delta_i\). From a statistical standpoint, secure aggregation **removes** the *direct* leakage channel but does **not** guarantee DP because an adversary could still infer information from the *distribution* of updates.

### 4.3 Combining Secure Aggregation with DP  

The standard approach is to apply **DP noise** to the aggregated update:

\[
\tilde{\Delta}= \Delta + \mathcal{N}(0,\sigma^2 C^2 I).
\]

Because the noise is added **after** secure aggregation, the privacy analysis follows the same composition rules as DP‑SGD, but with *client‑level* sensitivity. The **client‑level privacy** parameter \(\varepsilon_{\text{client}}\) is typically much tighter than *example‑level* privacy, reflecting the fact that each client may hold many data points.

### 4.4 Statistical Sample Complexity in FL  

For a target excess risk \(\alpha\), the **sample complexity** under client‑level DP scales as (Bassily et al., 2020):

\[
n = \tilde{O}\!\left(\frac{\sqrt{K}\, \log(1/\delta)}{\varepsilon}\cdot\frac{1}{\alpha^2}\right).
\]

Thus, increasing the number of participating clients \(K\) **reduces** the required per‑client data to achieve a fixed utility under a given privacy budget—a phenomenon sometimes called **privacy amplification by federation**.

---

## 5. Scaling Laws and Sample‑Complexity for Privacy‑Preserving Training and Inference  

### 5.1 Classical Scaling Laws  

Empirical work on foundation models (Kaplan et al., 2020; Hoffmann et al., 2022) identified power‑law relationships between model size \(N\), dataset size \(D\), compute \(C\), and performance \(L\) (e.g., loss):

\[
L(N,D) \approx \left(\frac{N}{N_0}\right)^{-\alpha_N} + \left(\frac{D}{D_0}\right)^{-\alpha_D}.
\]

These **scaling laws** assume *unconstrained* training. When privacy constraints are imposed, the effective dataset size is reduced by the *privacy‑induced noise*.

### 5.2 Privacy‑Aware Scaling  

Recent theoretical work (Gopi et al., 2023) extends scaling laws to DP training by introducing a **privacy penalty factor** \(\phi(\varepsilon,\delta)\) that inflates the effective loss:

\[
L_{\text{DP}}(N,D;\varepsilon,\delta) \approx \phi(\varepsilon,\delta)\cdot L(N,D).
\]

Empirically, \(\phi\) behaves roughly as:

\[
\phi(\varepsilon,\delta) \approx 1 + \frac{c}{\varepsilon^\beta},
\]

with \(c,\beta>0\) determined by model architecture and training regime. This formulation predicts that **larger models can tolerate stronger privacy (smaller \(\varepsilon\))** because the baseline loss \(L(N,D)\) is already low; the relative penalty diminishes.

### 5.3 Sample Complexity under DP  

For a target excess loss \(\alpha\) and a fixed model size, the **DP sample complexity** obeys (Bassily et al., 2020):

\[
D_{\text{DP}} = \tilde{O}\!\left(\frac{\sqrt{N}}{\alpha^2}\cdot\frac{\log(1/\delta)}{\varepsilon}\right).
\]

Comparing with the non‑private bound \(D = \tilde{O}(\sqrt{N}/\alpha^2)\), we see a multiplicative factor of \(\log(1/\delta)/\varepsilon\). Consequently, **privacy‑preserving training demands more data**, but the factor grows only linearly