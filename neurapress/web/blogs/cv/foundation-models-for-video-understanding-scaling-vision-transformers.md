# Foundation Models for Video Understanding  
**Scaling Vision Transformers to Temporal Data**

---

## Introduction  

Video is the richest visual signal we can capture: it bundles spatial appearance, motion dynamics, audio, and often textual cues such as subtitles or on‑screen captions. Over the past few years, **Vision Transformers (ViTs)** have become the de‑facto backbone for image understanding, and their extension to the temporal domain is unlocking a new generation of *foundation models* that can be adapted to any downstream video task with minimal effort.

This tutorial walks through the complete lifecycle of building, pre‑training, prompting, fine‑tuning, and deploying a transformer‑based video foundation model. We will cover:

1. Core transformer architectures for video.  
2. Large‑scale multimodal pre‑training on datasets that combine RGB frames, optical flow, audio, and subtitles.  
3. Promptable video queries using natural language and explicit time descriptors.  
4. Parameter‑efficient fine‑tuning for action recognition, video segmentation, and video captioning.  
5. Real‑time edge inference strategies for AR/VR, robotics, and surveillance.  

By the end you should have a concrete blueprint for turning a generic video transformer into a production‑ready system.

---

## 1. Vision Transformers for Video  

### 1.1 From Spatial to Spatio‑Temporal Modeling  

Classic ViTs operate on a **2‑D patch sequence** extracted from a single image. To handle video, we must incorporate the **time axis**. Two design families dominate:

| Design family | Core idea | Representative papers |
|---|---|---|
| **Joint space‑time attention** | Treat a video as a 3‑D token grid (T × H × W) and apply full self‑attention across all dimensions. | *ViViT* (Arnab et al., 2021); *TimeSformer* (Bertasius et al., 2021) |
| **Factorized attention** | Separate spatial and temporal attention to reduce quadratic cost. | *MViT* (Fan et al., 2021); *Swin‑V2* for video (Liu et al., 2022) |
| **Masked video modeling** | Mask a subset of spatio‑temporal tokens and reconstruct them, akin to BERT for video. | *VideoMAE* (Tong et al., 2022) |

All three families share a **patch embedding** stage, positional encodings (often *learned* or *sinusoidal* for time), and a stack of transformer blocks. The choice of factorization determines the trade‑off between accuracy and compute.

### 1.2 Core Architectures  

* **ViViT** – Stacks of pure 3‑D attention; excels on short clips (≤ 8 s) but scales quadratically.  
* **TimeSformer** – Alternates spatial and temporal attention; offers a sweet spot for medium‑range videos (≈ 16 s).  
* **MViT** – Hierarchical multi‑scale attention that progressively reduces temporal resolution; suited for long‑form content.  
* **VideoMAE** – Pre‑trains a vanilla ViT on video by masking 90 % of tokens; produces highly transferable features with modest compute.

These backbones serve as the **foundation** upon which multimodal signals and downstream heads are attached.

---

## 2. Large‑Scale Multimodal Pre‑Training  

### 2.1 Datasets that Fuse Vision, Motion, Audio, and Text  

| Modality | Representative Datasets | Scale |
|---|---|---|
| **RGB + Subtitles** | *HowTo100M* (Miech et al., 2020) – 136 M video‑subtitle pairs | 100 M clips |
| **RGB + Optical Flow** | *Kinetics‑700* (Carreira et al., 2019) – manually annotated actions | 650 k clips |
| **Audio + Video** | *AudioSet* (Gemmeke et al., 2017) – 2 M 10‑s clips with audio tags | 2 M clips |
| **Full Multimodal** | *VIOLET* (Li et al., 2022) – 1.2 B video‑audio‑text triples | 1 B clips |

Combining these sources yields a **multimodal pre‑training corpus** that covers diverse domains (tutorials, sports, movies, surveillance) and provides **temporal metadata** such as start/end timestamps for subtitles or action boundaries.

### 2.2 Input Representation  

| Modality | Pre‑processing | Tokenization |
|---|---|---|
| **RGB frames** | Uniform sampling (e.g., 16 fps), 16 × 16 patches | Linear projection + positional embedding |
| **Optical flow** | TV‑L1 flow, stacked as a 2‑channel image | Same patch embedding as RGB |
| **Audio** | Log‑Mel spectrogram (e.g., 128 × 128) → 2‑D patches | Separate audio transformer or shared token space |
| **Subtitles** | Tokenize with a BPE tokenizer (e.g., SentencePiece) | Learned text embeddings, aligned via timestamps |

All token streams are **concatenated** and fed into a single transformer, optionally preceded by modality‑specific encoders that map each stream to a common dimension.

### 2.3 Training Objectives  

1. **Masked Token Modeling (MTM)** – Randomly mask spatio‑temporal patches and predict them (VideoMAE).  
2. **Contrastive Alignment** – Pull together representations of matching video‑audio‑text triples (CLIP‑style loss; *AudioCLIP* (Morgado et al., 2022)).  
3. **Temporal Order Prediction** – Shuffle clip segments and train the model to recover the correct order, encouraging temporal reasoning.  
4. **Multimodal Captioning** – Autoregressive decoder predicts subtitles given video tokens, reinforcing language grounding.

A typical pre‑training schedule mixes these losses with weighting tuned on a held‑out validation set.

---

## 3. Promptable Video Queries  

### 3.1 Natural‑Language Prompts  

Foundation models can be **conditioned** on free‑form text. A prompt such as  

> “*Find all moments where a person is juggling*”  

is tokenized and injected either as a **prefix** to the video token sequence or via a **cross‑attention** layer (as in *Flamingo* (Alayrac et al., 2022)). The model learns to attend to frames that satisfy the semantic constraint.

### 3.2 Time‑Based Descriptors  

Temporal grounding can be expressed explicitly:

| Example Prompt | Interpretation |
|---|---|
| “*Show the first 5 seconds of a dog running*” | Attend to frames 0 – 5 s, filter by “dog running”. |
| “*Describe the scene between 00:10 and 00:20*” | Restrict attention to tokens whose timestamps lie in that interval. |
| “*What happens after the alarm rings?*” | Use the word “after” to bias the model toward future frames relative to the detected alarm sound. |

Implementation-wise, we **mask** tokens outside the requested interval or add a **binary time mask** to the attention scores.

### 3.3 Prompt Engineering Tips  

* Keep prompts **concise** (< 30 tokens) to avoid diluting attention.  
* Align **temporal granularity** of the prompt with the clip sampling rate (e.g., 2 fps for long videos).  
* Use **canonical verbs** (“show”, “list”, “describe”) that the model has seen during pre‑training.

### 3.4 Example Applications  

* **Video search engines** – Users type “*clip of a basketball dunk at halftime*”.  
* **Robotic perception** – “*Locate the red object within the next 3 seconds*”.  
* **AR overlays** – “*Display subtitles for the dialogue that starts at 00:45*”.

---

## 4. Efficient Fine‑Tuning for Downstream Tasks  

### 4.1 Parameter‑Efficient Strategies  

| Technique | How it works | Typical overhead |
|---|---|---|
| **Adapter layers** (Houlsby et al., 2019) | Small bottleneck MLPs inserted after each transformer block; only adapters are trained. | < 5 % of parameters |
| **LoRA** (Hu et al., 2021) | Low‑rank updates to the query/key/value matrices; original weights stay frozen. | < 1 % of parameters |
| **Prompt tuning** (Liu et al., 2021) | Learnable prompt tokens prepended to the input; the backbone remains untouched. | < 0.1 % of parameters |

These methods preserve the **general knowledge** acquired during massive pre‑training while adapting to task‑specific nuances.

### 4.2 Task‑Specific Heads  

| Downstream Task | Head Architecture | Loss |
|---|---|---|
| **Action Recognition** | Global average pooling → linear classifier (|C| classes