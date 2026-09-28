# Foundation Models for Video Understanding  
**Scaling Vision Transformers to Spatiotemporal Data**

---

## Introduction

The rise of **Vision Transformers (ViTs)** has reshaped the landscape of image‑based computer vision, but extending their success to the temporal domain has proven more challenging. Video understanding requires models that can reason over both space and time while handling multimodal signals such as audio and text. Recent work has begun to treat video as a *foundation model*—a large, pretrained representation that can be *prompted* and *fine‑tuned* for a wide variety of downstream tasks.  

This tutorial walks through the key components of building such foundation models for video:

1. **Large‑scale pretraining** on diverse video corpora with multimodal signals.  
2. **Promptable video queries** that combine natural language and temporal descriptors.  
3. **Efficient fine‑tuning** for action detection, video captioning, and video‑text retrieval.  
4. **Real‑time streaming inference** on edge and edge‑cloud platforms for live analytics.  

We will discuss state‑of‑the‑art architectures, training objectives, and practical deployment strategies, all grounded in peer‑reviewed research.

---

## 1. Large‑Scale Pretraining on Diverse Video Corpora

### 1.1. Video Datasets and Multimodal Signals

| Dataset | Size | Modality | Notes |
|---------|------|----------|-------|
| **Kinetics‑400/600/700** | 300k–700k clips | RGB | Action‑centric |
| **Something‑Something V2** | 220k clips | RGB | Fine‑grained temporal actions |
| **AVA** | 57k clips | RGB + Audio | Spatiotemporal annotations |
| **Epic Kitchens** | 55k clips | RGB + Audio + Text | Fine‑grained cooking actions |
| **YouCook2** | 2k videos | RGB + Audio + Text | Recipe‑based |
| **WebVid‑2.5M** | 2.5M clips | RGB + Audio + Text | Large‑scale web‑derived |
| **Laion‑Video** | 400M clips | RGB + Audio + Text | Open‑source, noisy |

These datasets provide a rich mixture of visual, auditory, and textual signals. For foundation models, the goal is to learn a *joint embedding* that captures correlations across modalities.  

### 1.2. Architectures for Spatiotemporal Transformers

| Architecture | Year | Key Idea |
|--------------|------|----------|
| **TimeSformer** | 2021 | Factorized spatiotemporal attention (spatial then temporal) |
| **ViViT** | 2021 | 3D patch embeddings with ViT backbone |
| **MViT** | 2021 | Multiscale vision transformer with hierarchical feature maps |
| **Swin‑Video** | 2022 | Shifted windows for efficient local attention |
| **VideoMAE** | 2022 | Masked auto‑encoding for self‑supervised pretraining |
| **Video Swin‑V2** | 2023 | Improved tokenization and hierarchical structure |
| **UniViT** | 2023 | Unified architecture for vision, language, and audio |

All these models share a common design: a **patch‑based tokenization** of video frames, followed by a transformer encoder that processes tokens across both spatial and temporal dimensions. The choice of architecture often depends on the target compute budget and the desired granularity of temporal modeling.

### 1.3. Pretraining Objectives

| Objective | Description | Reference |
|-----------|-------------|-----------|
| **Masked Frame Prediction** | Predict missing frames from context (VideoMAE) | He et al., 2022 |
| **Contrastive Video‑Text** | Align video and caption embeddings (VideoCLIP) | Li et al., 2021 |
| **Audio‑Visual Contrastive** | Align audio and visual streams (Audio‑Visual BERT) | Chen et al., 2019 |
| **Temporal Order Prediction** | Predict correct ordering of shuffled frames | Jain et al., 2020 |
| **Multi‑Modal Cross‑Entropy** | Jointly predict text, audio, and visual tokens | Liu et al., 2022 |

A combination of contrastive and generative objectives yields a robust representation that is sensitive to both *what* is happening and *when* it happens.

---

## 2. Promptable Video Queries with Natural Language and Temporal Descriptors

### 2.1. Prompt Design

Promptable video models