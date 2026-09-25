# Diffusion Models: From Image Synthesis to Scientific Discovery  

*Published by the Deep Learning Review*  

---  

## Introduction  

Over the past few years diffusion models have reshaped the generative‑AI landscape. Starting from the seminal **diffusion probabilistic models** (Sohl‑Dickstein *et al.*, 2015) and the **denoising diffusion probabilistic model (DDPM)** (Ho *et al.*, 2020), a cascade of innovations—score‑based sampling, classifier‑free guidance, progressive distillation, and large‑scale conditioning—has turned what was once a research curiosity into a production‑ready technology.  

Today diffusion models power everything from photorealistic image generators (e.g., **Stable Diffusion**, 2022) to video synthesis, 3‑D shape generation, inverse‑problem solvers, and even molecular‑level drug design. This tutorial walks through the **theoretical foundations**, **training and sampling tricks**, **scaling strategies**, and **real‑world scientific applications** of diffusion models, providing a single reference for practitioners and researchers alike.  

---  

## 1. Fundamentals of Diffusion Processes  

### 1.1 Forward (Noise) Process  

A diffusion model defines a **Markov chain** that gradually corrupts data \(x_0\) by adding Gaussian noise:

\[
q(x_t \mid x_{t-1}) = \mathcal{N}\bigl(x_t; \sqrt{1-\beta_t}\,x_{t-1}, \beta_t \mathbf{I}\bigr),\qquad t=1,\dots,T,
\]

where \(\beta_t \in (0,1)\) is a variance schedule (often linear or cosine). After \(T\) steps the distribution approaches an isotropic Gaussian:

\[
q(x_T \mid x_0) = \mathcal{N}\bigl(x_T; \mathbf{0}, \mathbf{I}\bigr).
\]

Because each step is analytically tractable, we can write a closed‑form expression for any intermediate \(x_t\):

\[
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I}),
\]

with \(\alpha_t = 1-\beta_t\) and \(\bar\alpha_t = \prod_{s=1}^t \alpha_s\).  

### 1.2 Reverse (Denoising) Process  

The generative direction seeks a **reverse transition** \(p_\theta(x_{t-1}\mid x_t)\) that gradually denoises a sample back to data space. In practice we parameterise the reverse kernel as a Gaussian whose mean is predicted by a neural network:

\[
p_\theta(x_{t-1}\mid x_t) = \mathcal{N}\bigl(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 \mathbf{I}\bigr).
\]

Training minimizes the variational bound on the negative log‑likelihood, which reduces (under the “noise‑prediction” parameterisation) to a simple **mean‑squared error** between the network’s prediction \(\epsilon_\theta(x_t, t)\) and the true noise \(\epsilon\):

\[
\mathcal{L}_\text{simple} = \mathbb{E}_{x_0,\epsilon,t}\Bigl[ \bigl\| \epsilon - \epsilon_\theta\bigl(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\,t\bigr) \bigr\|^2 \Bigr].
\]

This loss is **time‑uniform**, making training stable across the whole diffusion horizon.  

### 1.3 Probabilistic Viewpoint  

From a probabilistic perspective, diffusion models are **latent variable models** with a tractable prior \(p(x_T)=\mathcal{N}(\mathbf{0},\mathbf{I})\) and a learned posterior approximated by the reverse chain. The ELBO (Evidence Lower Bound) can be expressed as a sum of KL divergences between the true forward conditionals and the learned reverse conditionals plus a reconstruction term for the final step. This formulation connects diffusion models to **variational auto‑encoders (VAEs)** and **score‑based generative models** (Song & Ermon, 2020), the latter interpreting the reverse dynamics as solving a stochastic differential equation (SDE) driven by the score \(\nabla_{x_t}\log q(x_t)\).  

---  

## 2. Training and Sampling Strategies  

### 2.1 Classifier‑Free Guidance  

Original diffusion samplers required an external classifier to steer generation toward a condition \(c\) (e.g., a class label). **Classifier‑free guidance** (Ho & Salimans, 2022) eliminates the need for a separate classifier by training a single model to predict both conditional and unconditional noise:

\[
\epsilon_\theta(x_t, t, c) = \epsilon_\theta^{\text{cond}}(x_t, t, c),\qquad 
\epsilon_\theta(x_t, t, \varnothing) = \epsilon_\theta^{\text{uncond}}(x_t, t).
\]

During sampling we combine them:

\[
\tilde\epsilon_\theta = \epsilon_\theta^{\text{uncond}} + w\bigl(\epsilon_\theta^{\text{cond}} - \epsilon_\theta^{\text{uncond}}\bigr),
\]

where \(w\) is the **guidance scale** (typically \(1\!-\!10\)). Larger \(w\) yields sharper, more class‑consistent images at the cost of diversity.  

### 2.2 Progressive Distillation  

Diffusion samplers traditionally need hundreds or thousands of reverse steps, which is prohibitive for real‑time use. **Progressive distillation** (Salimans & Ho, 2022) trains a student model to mimic the output of a teacher after half the number of steps, recursively halving the step count. After \(k\) distillation stages a model can generate samples in as few as **4–8** steps while preserving quality comparable to the original 1000‑step baseline.  

### 2.3 Noise‑Schedule Optimisation  

Choosing \(\beta_t\) influences both sample quality and speed. Recent work (Karras *et al.*, 2022) proposes a **cosine schedule** and a **variance‑preserving (VP) vs. variance‑exploding (VE)** dichotomy, showing that VP schedules better preserve high‑frequency details, while VE schedules excel in low‑dimensional latent spaces.  

### 2.4 Training Tricks  

| Trick | Description | Impact |
|------|-------------|--------|
| **Self‑conditioning** (Nichol *et al.*, 2021) | Feed the model’s previous denoised estimate as an additional input. | Improves convergence and reduces sample noise. |
| **EMA of weights** | Maintain an exponential moving average of model parameters for inference. | Stabilises sampling and yields higher FID scores. |
| **Mixed‑precision & gradient checkpointing** | Reduce memory footprint without sacrificing batch size. | Enables training of 1‑B‑parameter models on commodity GPUs. |

---  

## 3. Scaling Diffusion Models  

### 3.1 High‑Resolution Images  

The leap from 64×64 to 1024×1024 images required architectural and training innovations:

* **Latent diffusion** (Rombach *et al.*, 2022) first compresses images with a VAE, runs diffusion in the lower‑dimensional latent space, and decodes back, cutting compute by > 80 %.  
* **Cross‑attention conditioning** (e.g., Stable Diffusion) allows fine‑grained text‑to‑image control while keeping the UNet lightweight.  
* **Large‑scale data & classifier‑free guidance** (Saharia *et al.*, 2022 – Imagen) demonstrate that scaling model size (up to 1.5 B parameters) and dataset size (≈ 2 B image‑text pairs) yields photorealistic results with sub‑human FID.  

### 3.2 Video Generation  

Video diffusion extends the spatial UNet with a **temporal dimension**. Two dominant paradigms exist:

1. **Autoregressive frame‑wise diffusion** – each frame is generated conditioned on previously generated frames (Ho *et al.*, 2022 – Video Diffusion).  
2. **Joint spatio‑temporal diffusion** – a 3‑D UNet processes a short clip (e.g., 16 frames) as a single tensor (Sun *et al.*, 2023 – VideoLDM).  

Key challenges are **temporal consistency** and **memory consumption**. Solutions include **flow‑guided conditioning**, **latent video diffusion**, and **mask‑based diffusion** for in‑painting across time.  

### 3.3 3‑D Data Generation  

Diffusion models have been adapted to several 3‑D representations: point clouds, meshes, implicit fields, and neural radiance fields (NeRFs). The table below summarises the most influential works up to 2024.

| Year | Method | 3‑D Representation | Key Contributions | Reference |
|------|--------|-------------------|-------------------|-----------|
| 2021 | **Score‑Based Point Cloud Generation** | Point cloud (N×3) | Directly models unordered point sets with a permutation‑invariant UNet; introduces Chamfer‑based loss for diffusion | Luo *et al.*, 2021 |
| 2022 | **Point‑E** | Point cloud + upsampling | Two‑stage pipeline: coarse diffusion → upsampled mesh; leverages classifier‑free guidance for shape fidelity | Shapero *et al.*, 2022 |
| 2022 | **Diffusion‑NeRF** | Implicit radiance field | Diffuses over NeRF density and color parameters; enables zero‑shot novel view synthesis | Nerfies *et al.*, 2022 |
| 2023 | **DreamFusion** | Signed distance function (SDF) | Optimises a NeRF via a frozen text‑to‑image diffusion model, achieving text‑guided 3‑D generation without 3‑D training data | Poole *et al.*, 2023 |
| 2023 | **Magic3D** | Tri‑plane representation | Introduces a scalable latent diffusion on tri‑planes, producing high‑resolution textured meshes in seconds | Lin *et al.*, 2023 |
| 2024 | **3‑D Gaussian Splatting Diffusion** | Gaussian splat primitives | Extends diffusion to the emerging Gaussian splatting rendering pipeline, offering fast rendering and fine‑grained geometry control | Liu *et al.*, 2024 |

These methods share a common recipe: **encode** the 3‑D structure into a latent tensor, **diffuse** in that space, and **decode** back to a geometry format (mesh, point cloud, or radiance field).  

---  

## 4. Real‑World Applications  

### 4.1 Creative Generation  

* **Text‑to‑Image** – Stable Diffusion (2022) and Imagen (2022) have democratized high‑fidelity image creation, supporting style transfer, in‑painting, and prompt‑to‑prompt editing.  
* **Audio & Music** – Diff