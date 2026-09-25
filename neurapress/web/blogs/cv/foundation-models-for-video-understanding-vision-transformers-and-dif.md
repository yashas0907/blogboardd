# Foundation Models for Video Understanding: Vision Transformers and Diffusion for Temporal Modeling  

*Published by the Computer Vision Community Blog*  

---

## Introduction  

Video understanding has moved from hand‑crafted pipelines to **foundation models** that learn generic spatiotemporal representations from massive, web‑scale corpora. Two architectural families now dominate the field:

1. **Spatiotemporal Vision Transformers (ViTs)** that extend the self‑attention paradigm from images to video clips.  
2. **Diffusion‑based generative models** that treat video synthesis as a sequential denoising process, enabling high‑fidelity generation and powerful downstream feature learning.

This tutorial walks through the most influential architectures, the data‑centric pretraining strategies that make them possible, and practical techniques for **efficient fine‑tuning** and **real‑time deployment**. By the end you will understand how to select, adapt, and serve a video foundation model for tasks ranging from action recognition to video‑to‑text generation.

---

## 1. Spatiotemporal Vision Transformer Architectures  

### 1.1. From Image to Video: Core Design Choices  

| Design Dimension | Image‑ViT | Video‑ViT | Typical Choices |
|------------------|-----------|-----------|-----------------|
| **Tokenization** | Patch embedding (e.g., 16×16) | 3‑D tubelet embedding (e.g., 2×16×16) or frame‑wise patches + temporal encoding | *Tubelet ViT* (Arnab et al., 2021), *TimeSformer* (Bertasius et al., 2021) |
| **Temporal Modeling** | None (static) | **Joint** (space‑time attention) or **Factorized** (spatial then temporal) | Joint: **ViViT** (Arnab et al., 2021); Factorized: **TimeSformer**, **MViT** (Li et al., 2022) |
| **Computation Scaling** | Quadratic in number of patches | Quadratic in space‑time tokens → expensive | Factorized attention, hierarchical pooling, or sparse attention to keep FLOPs tractable |

### 1.2. Representative Architectures  

| Model | Year | Temporal Strategy | Notable Contributions |
|-------|------|-------------------|-----------------------|
| **ViViT** | 2021 | Joint space‑time attention (full) | First pure‑ViT video model, demonstrated scalability with large pretraining (Kinetics‑700) |
| **TimeSformer** | 2021 | Factorized attention (spatial → temporal) | Showed that separating dimensions reduces memory while preserving accuracy |
| **MViT (Multiscale Vision Transformer)** | 2022 | Hierarchical pooling across space‑time | Efficient multiscale representation; strong on AVA action detection |
| **VideoMAE** | 2022 | Masked autoencoding on video tubes | Self‑supervised pretraining that rivals supervised models on Kinetics‑400 |
| **CoCa‑Video** | 2023 | Dual‑encoder (image‑video + text) with cross‑modal attention | Unified video‑text foundation model for retrieval and captioning |

#### Key Architectural Insights  

* **Tubelet vs. Frame‑wise Tokens** – Tubelet embeddings capture short‑range motion directly, reducing the number of tokens. Frame‑wise patches keep temporal resolution higher but need explicit positional encodings (e.g., sinusoidal or learned frame embeddings).  
* **Factorized Attention** – Splitting attention into spatial and temporal stages cuts the quadratic cost from *O((T·N)²)* to *O(T·N² + N·T²)*, where *T* is frames and *N* patches per frame.  
* **Hierarchical Design** – Down‑sampling in both space and time (as in MViT) yields a pyramid of tokens, enabling long‑range context without exploding memory.  

---

## 2. Video Diffusion Models for High‑Fidelity Generation  

Diffusion models have become the de‑facto standard for image synthesis; extending them to video introduces new challenges: temporal consistency, memory efficiency, and conditioning on multimodal signals.

### 2.1. Fundamentals of Video Diffusion  

A diffusion model learns to reverse a forward **Gaussian noise schedule** applied to a video tensor *V ∈ ℝ^{T×H×W×C}*. At each denoising step *t* the network predicts the noise residual *ε̂* conditioned on:

* **Spatial context** – captured by 3‑D convolutions or spatiotemporal transformers.  
* **Temporal context** – enforced through causal attention, recurrent blocks, or explicit motion priors.  

Training minimizes the expected *ℓ₂* loss between predicted and true noise, as in the original DDPM formulation (Ho et al., 2020).

### 2.2. Notable Video Diffusion Works  

| Model | Year | Core Innovation |
|-------|------|-----------------|
| **Video Diffusion Models (VDM)** | 2022 | 3‑D UNet with attention over space‑time; demonstrates 256‑frame synthesis |
| **Imagen Video** | 2022 | Cascaded diffusion (low‑res → high‑res) with text conditioning; state‑of‑the‑art video generation |
| **Make‑A‑Video** | 2022 | Large‑scale pretraining on *WebVid‑2M*; zero‑shot text‑to‑video synthesis |
| **Stable Diffusion Video** | 2023 | Latent diffusion applied to video latent space; reduces memory by >10× |
| **VideoGPT** | 2022 | Autoregressive transformer decoder for video generation; early diffusion alternative |

### 2.3. Temporal Consistency Techniques  

1. **Flow‑Guided Noise Prediction** – Use optical flow between adjacent frames to align latent features before denoising (Ho et al., 2022).  
2. **Temporal Attention Masks** – Restrict attention to a sliding window of frames, preserving causality while allowing information flow (Ho et al., 2022).  
3. **Latent Interpolation** – Generate a short keyframe sequence, then interpolate latent codes with a diffusion prior to fill intermediate frames (Liu et al., 2023).

---

## 3. Large‑Scale Pretraining on Web‑Scale Video Corpora  

### 3.1. Data Sources  

| Corpus | Size | Typical Content | Primary Use |
|--------|------|----------------|-------------|
| **HowTo100M** | 136 M video‑text pairs | Instructional YouTube clips | Multimodal pretraining |
| **WebVid‑2M** | 2 M short videos + captions | Diverse internet videos | Text‑to‑video diffusion |
| **Instagram‑1B** | ~1 B short-form videos | Social media content | Large‑scale self‑supervision |
| **Kinetics‑700** | 650 k clips | Human actions | Supervised video classification |
| **AVA** | 430 k annotated frames | Spatio‑temporal action detection | Fine‑grained localization |

### 3.2. Self‑Supervised Objectives  

* **Masked Video Modeling (MVM)** – Randomly mask spatiotemporal tubes and reconstruct them (VideoMAE).  
* **Contrastive Multimodal Learning** – Align video embeddings with paired text or audio (CLIP‑Video, ALIGN‑Video).  
* **Predictive Coding** – Forecast future frames or motion vectors from past context (VIMPAC).  

These objectives enable models to learn **generic motion primitives** and **semantic concepts** without exhaustive annotation.

### 3.3. Scaling Laws  

Recent analyses (Zhou et al., 2023) show that **performance improves logarithmically** with the number of video tokens seen during pretraining, provided the model capacity (depth × width) scales accordingly. Consequently, modern video foundation models typically feature **>1 B parameters** and are trained on **hundreds of thousands of GPU‑hours**.

---

## 4. Efficient Fine‑Tuning and Deployment Strategies  

Fine‑tuning a billion‑parameter video foundation model from scratch is impractical for most teams. The community has converged on a toolbox of **parameter‑efficient adaptation** methods, combined with **hardware‑aware optimizations** for real‑time inference.

### 4.1. Parameter‑Efficient Fine‑Tuning  

| Method | Core Idea | Typical Overhead |
|--------|-----------|------------------|
| **Adapters** (Houlsby et al., 2019) | Insert lightweight bottleneck modules (↓ k) after each transformer block; only adapters are trained. | +0.1 % of total parameters |
| **Prompt Tuning** (Mao et al., 2022) | Learn a small set of learnable tokens prepended to the input sequence; the backbone remains frozen. | < 0.01 % of parameters |
| **LoRA (Low‑Rank Adaptation)** (Hu et al., 2021) | Decompose weight updates into low‑rank matrices ΔW = A·Bᵀ; inject into attention/query/value projections. | 0.1–0.5 % of parameters |
| **BitFit** (Ben Zaken et al., 2022) | Fine‑tune only bias terms across the network. | ~0.02 % of parameters |
| **AdapterFusion** (Pfeiffer et al., 2021) | Fuse multiple task‑specific adapters via a learned gating network, enabling multi‑task sharing. | Slightly higher than single adapters |

#### Practical Tips  

* **Layer Selection** – For video tasks, adapters placed in the **temporal attention layers** often yield the biggest gains because motion dynamics are task‑specific.  
* **Rank Choice for LoRA** – A rank of 8–16 balances expressivity and memory; larger ranks are useful for high‑resolution generation.  
* **Training Schedule** – Use a **lower learning rate** (1e‑5 to 5e‑5) for adapters/LoRA and a **shorter schedule** (≤ 10 epochs) when the backbone is frozen.

### 4.2. Knowledge Distillation for Speed  

Distilling a large video foundation model into a **compact student** (e.g., a 12‑layer ViT or a lightweight 3‑D CNN) preserves most of the performance while reducing latency. Common recipes:

* **Logits Distillation** – Minimize KL divergence between teacher and student class probabilities.  
* **Feature Mimicry** – Align intermediate spatiotemporal token embeddings using L2 loss.  
* **Temporal Consistency Loss** – Encourage the student to reproduce the teacher’s motion vectors or flow predictions.

Distilled students can run at **>30 fps** on a single RTX 4090 or **>15 fps** on an edge GPU (e.g., NVIDIA Jetson Orin).

### 4.3. Quantization & Pruning  

| Technique | Effect | Recommended Tool