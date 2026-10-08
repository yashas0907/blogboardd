# Energy‑Based Models and Score‑Based Generative Modeling  
*A comprehensive tutorial*

---

## Introduction  

Energy‑Based Models (EBMs) provide a unifying view of many modern generative approaches. By defining an **unnormalized density** through an energy function \(E_\theta(\mathbf{x})\), EBMs sidestep the need for an explicit partition function during training. When combined with **score‑matching** techniques, they give rise to the family of **score‑based generative models** (SGMs) that have set the state‑of‑the‑art in high‑fidelity image synthesis, scientific data generation, and inverse‑problem solving.

This tutorial walks through:

1. The theoretical foundations of EBMs and score matching.  
2. Large‑scale training strategies, including contrastive and diffusion‑based objectives.  
3. Efficient sampling and acceleration tricks (Langevin dynamics, annealed sampling, distillation).  
4. A complete end‑to‑end pipeline for image synthesis.  
5. Real‑world applications and practical advice.

The goal is to equip you with a **hands‑on understanding** that can be turned into production‑ready code.

---

## 1. Energy‑Based Models: Foundations  

### 1.1 Definition  

An EBM assigns an **energy** to each data point \(\mathbf{x}\in\mathbb{R}^d\) via a neural network \(E_\theta(\mathbf{x})\). The corresponding probability density is  

\[
p_\theta(\mathbf{x}) = \frac{\exp\!\big(-E_\theta(\mathbf{x})\big)}{Z(\theta)},\qquad 
Z(\theta)=\int \exp\!\big(-E_\theta(\mathbf{x})\big)\,d\mathbf{x}.
\]

\(Z(\theta)\) is the **partition function**; it is typically intractable, which motivates alternative training objectives.

### 1.2 Maximum‑Likelihood Gradient  

The log‑likelihood gradient decomposes into a **data term** and a **model term**:

\[
\nabla_\theta \log p_\theta(\mathbf{x}) 
= -\nabla_\theta E_\theta(\mathbf{x}) 
+ \mathbb{E}_{\tilde{\mathbf{x}}\sim p_\theta}\!\big[\nabla_\theta E_\theta(\tilde{\mathbf{x}})\big].
\]

The expectation under the model requires samples from \(p_\theta\), leading to **Monte‑Carlo approximations** such as Langevin dynamics.

### 1.3 Score Matching  

Instead of learning the energy directly, **score matching** (Hyvärinen, 2005) learns the **score function**  

\[
\mathbf{s}_\theta(\mathbf{x}) = \nabla_{\mathbf{x}} \log p_\theta(\mathbf{x}) = -\nabla_{\mathbf{x}}E_\theta(\mathbf{x}),
\]

by minimizing the expected squared error between the model score and the true data score:

\[
\mathcal{L}_{\text{SM}}(\theta)=\frac12\mathbb{E}_{p_{\text{data}}}\!\big\|\mathbf{s}_\theta(\mathbf{x})-\nabla_{\mathbf{x}}\log p_{\text{data}}(\mathbf{x})\big\|_2^2.
\]

The **denoising score matching** (Vincent, 2011) reformulates this loss using perturbed data \(\tilde{\mathbf{x}} = \mathbf{x} + \sigma\mathbf{z}\) (with \(\mathbf{z}\sim\mathcal{N}(0,\mathbf{I})\)):

\[
\mathcal{L}_{\text{DSM}}(\theta)=\frac12\mathbb{E}_{p_{\text{data}},\mathbf{z}}\!\big\|\mathbf{s}_\theta(\tilde{\mathbf{x}})+\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma^2}\big\|_2^2.
\]

By training over a **continuum of noise levels** \(\sigma\in[\sigma_{\min},\sigma_{\max}]\), the model learns a **family of scores** that can be used for sampling via reverse diffusion.

---

## 2. Contrastive Training of EBMs  

Contrastive methods approximate the model expectation in the likelihood gradient by short Markov chains.

| Method | Core Idea | Typical Chain Length | Notable Traits |
|--------|-----------|----------------------|----------------|
| **Contrastive Divergence (CD‑k)** | Initialize at data, run \(k\) Gibbs/Langevin steps, treat endpoint as negative sample. | 1–10 | Fast but biased; works well for shallow models. |
| **Persistent CD (PCD)** | Maintain a replay buffer of “negative particles” that evolve across minibatches. | 1–5 per update | Reduces bias, improves mixing for deeper nets. |
| **Stochastic Gradient Langevin Dynamics (SGLD)** | Add Gaussian noise to gradient updates of \(\mathbf{x}\) while descending the energy. | 10–100 (often annealed) | Provides a principled approximation to the Langevin diffusion; widely used for high‑dimensional images. |

### 2.1 Large‑Scale Tricks  

Training EBMs on ImageNet‑scale data demands additional engineering:

* **Spectral Normalization** (Miyato et al., 2018) to bound the Lipschitz constant of \(E_\theta\).  
* **Batch Normalization** or **Group Normalization** inside the energy network for stable gradients.  
* **Multi‑Scale Architectures** (e.g., hierarchical ConvNets) that share parameters across resolutions.  
* **Replay Buffers** of size \(10^5\)–\(10^6\) to keep a diverse set of negative samples.  

---

## 3. Diffusion‑Based Objectives and Score‑Based Generative Modeling  

Diffusion models view data generation as the **reverse of a forward noising process**. The connection to score matching is formalized through **stochastic differential equations (SDEs)**.

### 3.1 Forward Diffusion  

A continuous‑time Ornstein‑Uhlenbeck (OU) process adds Gaussian noise:

\[
d\mathbf{x}_t = -\beta(t)\mathbf{x}_t\,dt + \sqrt{2\beta(t)}\,d\mathbf{w}_t,\qquad t\in[0,1],
\]

with \(\beta(t)\) controlling the noise schedule. At \(t=1\) the distribution approaches a standard normal.

### 3.2 Reverse SDE  

The reverse dynamics (Anderson, 1982) are governed by the **score of the perturbed data**:

\[
d\mathbf{x}_t = \big[-\beta(t)\mathbf{x}_t - \beta(t)\,\mathbf{s}_\theta(\mathbf{x}_t,t)\big]dt + \sqrt{2\beta(t)}\,d\overline{\mathbf{w}}_t.
\]

Training \(\mathbf{s}_\theta\) with DSM across a set of discrete noise levels \(\{\sigma_i\}\) yields a **time‑conditional score network**.

### 3.3 Popular Diffusion Variants  

| Model | Key Publication | Main Innovation |
|-------|-----------------|-----------------|
| **DDPM** (Ho et al., 2020) | Discrete‑time diffusion with variance schedule; simple MSE loss on predicted noise. |
| **NCSN / NCSNv2** (Song & Ermon, 2019/2020) | Direct score estimation; variance‑exploding (VE) and variance‑preserving (VP) SDEs. |
| **EDM (Elucidated Diffusion Models)** (Karras et al., 2022) | Adaptive noise schedule, improved sampler (Heun’s method), state‑of‑the‑art FID on ImageNet‑256. |

All these models share a **single training objective** that can be written as a weighted sum of DSM losses over the chosen noise levels.

---

## 4. Efficient Sampling and Acceleration  

Sampling from a learned score network is the most computationally intensive step. Below are the most widely adopted accelerations.

### 4.1 Langevin Dynamics  

**Unadjusted Langevin Algorithm (ULA)** for a fixed noise level \(\sigma\):

\[
\mathbf{x}_{k+1} = \mathbf{x}_k + \frac{\epsilon}{2}\,\mathbf{s}_\theta(\mathbf{x}_k,\sigma) + \sqrt{\epsilon}\,\mathbf{z}_k,\quad \mathbf{z}_k\sim\mathcal{N}(0,\mathbf{I}).
\]

**Metropolis‑adjusted Langevin Algorithm (MALA)** adds an acceptance step, improving asymptotic correctness at the cost of extra evaluations.

### 4.2 Annealed Sampling  

**Annealed Langevin Dynamics (ALD)** runs ULA over a decreasing sequence of noise levels \(\sigma_1>\dots>\sigma_L\). For each level \(i\) we perform \(N_i\) steps:

```
for i = 1 … L:
    for n = 1 … N_i:
        x ← x + (ε_i/2) * sθ(x, σ_i) + √ε_i * N(0, I)
```

The schedule \(\{\sigma_i\}\) is often **geometric**: \(\sigma_i = \sigma_{\max} (\sigma_{\min}/\sigma_{\max})^{i/L}\).

### 4.3 Adaptive Step Sizes & Preconditioning  

* **Per‑level step size** \(\epsilon_i = \alpha \sigma_i^2\) (α≈0.1) stabilizes dynamics.  
* **Preconditioned Langevin** multiplies the score by a diagonal matrix derived from the empirical covariance of the noisy data.  
* **Nesterov momentum** (Song et al., 2021) accelerates convergence, especially for high‑resolution images.

### 4.4 Distillation to Deterministic Samplers  

Training a **student network** to imitate the multi‑step sampler (knowledge distillation) yields a **single‑step generator**. Notable works:

* **DDIM** (Song et al., 2020) shows that the reverse diffusion ODE can be discretized with far fewer steps while preserving quality.  
* **Distilled Diffusion Models** (Salimans & Ho, 2022) train a deterministic UNet to predict the final image directly from noise, achieving >30× speed‑up.

### 4.5 Hyper‑Parameter Table (Complete)  

| Symbol | Meaning | Typical Value (ImageNet‑256) | Comments |
|--------|---------|------------------------------|----------|
| \(\sigma_{\max}\) | Largest noise level (initial diffusion) | 50.0 | Sets the variance of the Gaussian prior. |
| \(\sigma_{\min}\) | Smallest noise level (final refinement) | 0.01 | Controls final image fidelity. |
| \(L\) | Number of noise levels (geometric schedule) | 1000 | Larger \(L\) yields smoother trajectories. |
| \(\epsilon_i\) | Langevin step size at level \(i\) | \(\epsilon_i = 0.