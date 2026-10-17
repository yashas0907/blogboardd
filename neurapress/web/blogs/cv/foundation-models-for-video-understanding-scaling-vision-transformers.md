# Foundation Models for Video Understanding: Scaling Vision Transformers for Spatiotemporal Reasoning  

*Published by the Computer Vision Technical Review*  

---

## Introduction  

Video is the richest visual modality we have: every frame carries color, texture, depth, motion, and often sound. Extracting **spatiotemporal semantics** from such data has long required handcrafted pipelines—optical‑flow extraction, hand‑engineered motion descriptors, and separate audio‑processing stages. The rise of **Vision Transformers (ViTs)** has fundamentally altered this landscape. By treating a video as a sequence of *tubelets* (spatial patches extended across time) and applying self‑attention jointly over space and time, ViTs can learn unified representations that capture motion, context, and multimodal cues in a single forward pass.

However, scaling ViTs to the size and diversity of modern video corpora raises several engineering challenges:

1. **Massive pre‑training** on multimodal datasets (RGB, optical flow, audio, depth).  
2. **Promptable video queries** that accept natural‑language and temporal descriptors.  
3. **Efficient fine‑tuning** for downstream tasks such as action recognition, video segmentation, captioning, and retrieval.  
4. **Real‑time edge inference** for AR/VR, surveillance, and autonomous driving.  

This tutorial walks through each of these pillars, explains the most effective techniques, and provides a concrete end‑to‑end pipeline that can be reproduced on commodity hardware.  

---  

## 1. Vision Transformers for Spatiotemporal Reasoning  

### 1.1 From Images to Video  

Classic ViTs split an image into fixed‑size patches, linearly embed each patch, add positional encodings, and feed the resulting token sequence to a stack of transformer layers. Extending this to video requires two key modifications:

| Modification | Rationale |
|--------------|-----------|
| **Tubelet embedding** – 3‑D patches of shape *(T × H × W)* rather than 2‑D patches. | Captures short‑range motion directly in the tokenization stage (e.g., **ViViT** 2021). |
| **Space‑time attention** – either full joint attention (quadratic in *T·N*) or factored attention (e.g., **TimeSformer** 2021) that alternates spatial and temporal blocks. | Balances expressivity with computational cost. |

Recent works such as **MViT** (2021) and **VideoMAE** (2022) demonstrate that *progressive down‑sampling* of tubelets and *masked video modeling* can further improve scalability while preserving fine‑grained motion cues.

---  

## 2. Large‑Scale Multimodal Pre‑training  

### 2.1 Data Sources  

| Modality | Representative Datasets | Typical Size | Notes |
|----------|--------------------------|--------------|-------|
| **RGB video** | *Kinetics‑700* (650 k clips), *HowTo100M* (100 M narrated clips) | Hundreds of millions of frames | Provides visual semantics and coarse action labels. |
| **Optical flow** | *FlyingThings3D* (synthetic flow), *MPI‑Sintel* (realistic flow) | ~20 k sequences | Flow can be stored as JPEG‑compressed 2‑channel images and used as a second channel in multimodal token streams. |
| **Audio** | *AudioSet* (2 M 10‑s clips), *VGGSound* (200 k clips) | Tens of millions of seconds | Raw waveform or log‑mel spectrograms are tokenized with a small ConvNet before concatenation. |
| **Depth** | *Ego4D* (first‑person depth), *NYU‑Depth V2* (indoor scenes) | ~1 M frames | Depth maps are treated as an additional channel; can be encoded with a lightweight depth encoder. |

Combining these sources yields **multimodal video corpora** that exceed a petabyte when stored uncompressed, motivating the need for **efficient data pipelines** (e.g., streaming TFRecord/Parquet files, on‑the‑fly decoding).

### 2.2 Pre‑training Objectives  

| Objective | Description | Key Papers |
|-----------|-------------|------------|
| **Masked Video Modeling (MVM)** | Randomly mask a subset of tubelets and predict raw pixels or latent features. | *VideoMAE* 2022 |
| **Contrastive Multimodal Alignment** | Pull together embeddings from different modalities (RGB ↔ audio ↔ flow) while pushing apart unrelated pairs. | *VATT* 2021, *OmniVL* 2022 |
| **Temporal Order Prediction** | Shuffle tubelet order and ask the model to recover the correct chronology. | *TimeSformer* 2021 |
| **Cross‑modal Captioning** | Generate natural‑language descriptions from video tokens; loss is cross‑entropy on captions. | *CoCa* 2022 (image‑text) extended to video in *Flamingo* 2022 |

A **joint pre‑training schedule** typically interleaves MVM (70 % of steps) with contrastive alignment (20 %) and captioning (10 %). The resulting **foundation model** learns a unified spatiotemporal embedding space that can be queried with language.

---  

## 3. Promptable Video Queries  

### 3.1 Natural‑Language Prompts  

Foundation models such as **CLIP** (2021) have shown that a single linear projection can map text embeddings into the same space as visual embeddings. Extending this to video, we prepend a **text token** (e.g., “*a person is juggling*”) to the tubelet sequence and let the transformer attend across modalities. At inference, the model returns a similarity score between the prompt and each video segment.

### 3.2 Temporal Descriptors  

Video queries often require **when** something happens, not just **what**. Temporal descriptors can be encoded in two ways:

1. **Explicit time tokens** – special tokens like `[T0]`, `[T5]` inserted at the appropriate position in the token stream.  
2. **Relative positional encodings** – augment the sinusoidal positional embeddings with a *temporal scale* factor, enabling the model to understand phrases such as “*in the first 3 seconds*” or “*between frame 30 and 45*”.

Empirically, adding a small set of **temporal prompt tokens** improves zero‑shot retrieval on benchmarks like *YouCook2* (≈ +4 % mAP) without any fine‑tuning.

### 3.3 Prompt Engineering Tips  

| Tip | Effect |
|-----|--------|
| Use **action‑oriented verbs** (“*running*, *opening*”) rather than nouns. | Aligns with the action‑biased pre‑training data. |
| Include **adverbial modifiers** (“*quickly*, *slowly*”) for finer motion discrimination. | Improves temporal precision. |
| Keep prompts **≤ 12 tokens** to avoid exceeding the model’s maximum sequence length (typically 512 tokens). | Prevents truncation of video tokens. |

---  

## 4. Efficient Fine‑Tuning for Downstream Tasks  

### 4.1 Parameter‑Efficient Techniques  

| Technique | How It Works | Typical Savings |
|-----------|--------------|-----------------|
| **Adapter layers** (Houlsby et al., 2019) | Insert lightweight MLP bottlenecks after each transformer block; only adapters are trained. | 1–2 % of total parameters |
| **LoRA (Low‑Rank Adaptation)** (Hu et al., 2021) | Decompose weight updates into low‑rank matrices added to the frozen backbone. | 0.5–1 % of parameters |
| **Prefix‑tuning** (Li & Liang, 2021) | Learn a set of virtual tokens that are concatenated to the input sequence; backbone stays frozen. | 0.1–0.5 % of parameters |

These methods reduce GPU memory consumption and enable **few‑shot** adaptation on small labeled datasets (e.g., *DAVIS* for video segmentation).

### 4.2 Task‑Specific Heads  

| Task | Head Architecture | Typical Loss |
|------|-------------------|--------------|
| **Action recognition** | Linear classifier on the [CLS] token; optionally temporal pooling (max/average). | Cross‑entropy |
| **Video segmentation** | Mask decoder (e.g., **Mask2Former** 2022) that attends to the transformer output and predicts per‑pixel masks. | Dice + cross‑entropy |
| **Captioning** | Autoregressive decoder (Transformer‑XL) conditioned on video embedding; teacher‑forcing during training. | Cross‑entropy + CIDEr |
| **Retrieval** | Dual‑encoder: video encoder + text encoder; similarity computed via dot‑product. | Contrastive InfoNCE |

A **multi‑task fine‑tuning** schedule—alternating batches from each downstream dataset—often yields a single model that can perform all four tasks with only a modest drop in per‑task performance (≈ 2 % relative).

---  

## 5. Real‑Time Edge Inference  

### 5.1 Hardware Considerations  

| Platform | Compute | Memory | Typical Latency Budget |
|----------|---------|--------|------------------------|
| **Mobile GPU (Adreno 730)** | 1.2 TFLOPs | 8 GB | ≤ 30 ms per 16‑frame clip |
| **NPU (Apple Neural Engine)** | 2.5 TFLOPs | 6 GB | ≤ 20 ms |
| **Edge ASIC (Qualcomm Hexagon)** | 0.8 TFLOPs | 4 GB | ≤ 40 ms |

Because video transformers are memory‑intensive, **model scaling** is essential. The community now defines four canonical sizes:

| Size | Parameters | FLOPs (per 16‑frame clip, 224×224) |
|------|------------|-----------------------------------|
| **ViT‑T (tiny)** | 30 M | 2.5 B |
| **ViT‑S (small)** | 55 M | 4.8 B |
| **ViT‑B (base)** | 86 M | 8.1 B |
| **ViT‑L (large)** | 200 M | 19 B |

For edge deployment, **ViT‑T** or **ViT‑S** combined with aggressive compression (see Section 6) meet