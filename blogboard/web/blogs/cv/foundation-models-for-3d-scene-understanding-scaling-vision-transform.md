# Foundation Models for 3D Scene Understanding  
## Scaling Vision Transformers to Point Clouds and Meshes  

*Published by the Computer Vision Technical Review*  

---

## Introduction  

The past three years have witnessed a paradigm shift in 3‑D perception: **foundation models**—large, pre‑trained networks that can be adapted to a wide range of downstream tasks—are moving from 2‑D image domains into the world of point clouds, meshes, and multi‑modal depth data. By leveraging **Vision Transformers (ViTs)** as the backbone, researchers have built models that ingest raw geometric signals, learn universal 3‑D representations, and support **promptable queries** that combine textual and spatial cues.  

This tutorial walks through the full pipeline, from large‑scale multi‑modal pre‑training to efficient fine‑tuning and real‑time inference on edge/AR devices. We will:  

1. Explain how 3‑D data is tokenized for transformer consumption.  
2. Survey the most influential multi‑modal 3‑D datasets and pre‑training objectives.  
3. Describe prompt engineering for 3‑D queries.  
4. Outline fine‑tuning strategies for segmentation, detection, and reconstruction.  
5. Review attention variants that improve scalability.  
6. Present concrete benchmark results on a **Qualcomm Snapdragon 8 Gen 2** SoC and close with actionable take‑aways.  

---

## 1. Tokenizing Geometry for Transformers  

| Modality | Tokenization Strategy | Key References |
|----------|----------------------|----------------|
| **Point Clouds** | *Voxel‑based grouping* → learnable **point tokens** (e.g., Point‑BERT) or *patch‑wise sampling* (Point‑MAE). | Point‑BERT (Yu et al., 2022); Point‑MAE (Liu et al., 2022) |
| **Meshes** | *Face‑centric tokens* (triangular patches) + **positional encodings** derived from barycentric coordinates. | MeshTransformer (Wang et al., 2021) |
| **Depth / RGB‑D** | *Depth‑aware patch embedding* where depth values modulate the 2‑D token positions (DepthFormer, 2023). | DepthFormer (Wang et al., 2023) |
| **LiDAR** | *Range image* projection → 2‑D tokens with range‑aware sinusoidal encodings. | LaserFormer (Zhou et al., 2022) |

All approaches share a common pipeline:  

1. **Spatial Partitioning** – divide the raw signal into local groups (voxels, patches, faces).  
2. **Linear Embedding** – project each group into a fixed‑dimensional vector using a shared MLP or 1‑D/2‑D convolution.  
3. **Positional Encoding** – inject absolute or relative geometry (e.g., spherical coordinates, edge‑wise distances).  
4. **Class Token** – optional global token that aggregates scene‑level information.  

The resulting token sequence is fed to a standard ViT encoder, often with **hierarchical** or **sparse** attention (see § 5) to keep memory usage tractable for millions of points.

---

## 2. Large‑Scale Multi‑Modal Pre‑Training  

### 2.1 Datasets  

| Dataset | Modality(s) | Size (scenes) | Notable Characteristics |
|---------|-------------|---------------|--------------------------|
| **Waymo Open Dataset** | LiDAR + RGB | ~1 M frames | High‑density 3‑D sweeps, diverse urban scenarios. |
| **KITTI‑360** | LiDAR + RGB + GPS/IMU | ~2 M frames | Long‑range trajectories, fine‑grained pose. |
| **ScanNetV2** | RGB‑D + mesh reconstructions | 1 540 scenes | Indoor, richly annotated (semantic, instance). |
| **Matterport3D** | RGB‑D + textured meshes | 10 800 scans | Photorealistic indoor environments, multi‑view coverage. |
| **ARKitScenes** (Apple) | Depth + RGB + device pose | 300 k scenes | Mobile‑grade depth maps, real‑world AR capture. |
| **OmniScenes** (Meta) | 360° LiDAR + 360° RGB | 500 k frames | Full‑surround capture for autonomous driving. |

These datasets enable **cross‑modal pre‑training**: a single backbone can ingest LiDAR point clouds, depth maps, and RGB images, learning to align the modalities through shared token embeddings.

### 2.2 Pre‑Training Objectives  

| Objective | Description | Example |
|-----------|-------------|---------|
| **Masked Point Modeling (MPM)** | Randomly mask a subset of point tokens; predict their coordinates, features, or occupancy. | Point‑MAE (Liu et al., 2022) |
| **Contrastive Multi‑Modal Alignment** | Pull together embeddings from different modalities of the same scene while pushing apart mismatched pairs. | MVP (Gao et al., 2023) |
| **Cross‑Modal Reconstruction** | Reconstruct depth or RGB from LiDAR tokens (and vice‑versa). | DepthFormer (Wang et al., 2023) |
| **Scene‑Level Captioning** | Generate a textual description from a token set; back‑propagate through a language model. | CLIP‑3D (Chen et al., 2023) |
| **Geometric Consistency Loss** | Enforce that transformed (rotated/scaled) point sets produce consistent token embeddings. | 3D‑BERT (Zhang et al., 2022) |

A typical pre‑training schedule mixes **masking (≈70 %)** and **contrastive alignment (≈30 %)**, yielding a model that can both reconstruct missing geometry and understand cross‑modal semantics.

---

## 3. Promptable 3‑D Queries  

Prompt engineering, popularized in language models, now extends to 3‑D vision. A **prompt** can be:

* **Textual** – “Find all chairs in the living room.”  
* **Spatial** – “Select points within a 1 m radius of (x=2.3, y=1.1, z=0.5).”  
* **Hybrid** – “Locate the *red* car that is *behind* the bus.”

### 3.1 Prompt Encoder  

1. **Text Encoder** – a frozen language model (e.g., BERT‑base) produces a **text token**.  
2. **Spatial Encoder** – a lightweight MLP encodes geometric descriptors (e.g., bounding box, sphere).  
3. **Fusion Layer** – concatenates or cross‑attends the two embeddings, yielding a **prompt token** that is inserted into the transformer sequence (often at the beginning, similar to the CLS token).

### 3.2 Inference Workflow  

| Step | Operation |
|------|-----------|
| 1 | Encode the raw 3‑D scene into tokens (Section 1). |
| 2 | Encode the prompt (text + optional geometry). |
| 3 | Run a **single forward pass** through the ViT. |
| 4 | Read out the **prompt‑conditioned token** to retrieve results (e.g., per‑point classification scores, bounding‑box regressions). |

Because the prompt is part of the attention graph, the model can **reason jointly** over scene geometry and language, enabling zero‑shot queries such as “segment all windows” without any task‑specific fine‑tuning.

---

## 4. Efficient Fine‑Tuning for Downstream Tasks  

### 4.1 Parameter‑Efficient Strategies  

| Strategy | Mechanism | Typical Overhead |
|----------|-----------|------------------|
| **Adapter Layers** | Small bottleneck MLPs inserted after each transformer block; only adapters are trained. | 0.5 % of total parameters |
| **Prompt Tuning** | Learn a set of *soft prompt vectors* while freezing the backbone. | 0.1 % |
| **LoRA (Low‑Rank Adaptation)** | Decompose weight updates into low‑rank matrices; adds negligible FLOPs. | 0.2 % |
| **Selective Block Freezing** | Freeze early blocks (geometry extraction) and fine‑tune only higher‑level layers. | 30–40 % of parameters |

Empirically, **Adapter + LoRA** combinations achieve > 95 % of full‑fine‑tuning performance on 3‑D semantic segmentation while reducing GPU memory by 3× (see Table 1).

### 4.2 Task Heads  

| Task | Head Design | Loss |
|------|-------------|------|
| **Semantic Segmentation** | Per‑point linear classifier on top of token embeddings; optional CRF post‑processing. | Cross‑entropy |
| **Instance Segmentation** | Mask‑based decoder (similar to Mask‑R‑CNN) + centroid regression. | Dice + L1 |
| **Object Detection** | Anchor‑free center‑point prediction + 3‑D bounding‑box regression. | Focal + Smooth‑L1 |
| **Scene Reconstruction** | Implicit decoder (e.g., DeepSDF) conditioned on global token; predicts signed distance fields. | Chamfer + Eikonal |

All heads share the same **pre‑trained backbone**, which dramatically reduces data requirements. For example, fine‑tuning on **ScanNetV2** with only 5 % of the labeled frames yields 71.3 % mIoU—close to the 73.1 % achieved with full supervision.

---

## 5. Scalable Attention Variants  

Standard full‑self‑attention scales quadratically with token count, which is prohibitive for dense point clouds (> 100 k points). Researchers have introduced several **efficient attention** mechanisms:

| Variant | Core Idea | Complexity |
|---------|-----------|------------|
| **Sparse Sub‑Window Attention** | Partition tokens into spatial windows; attend only within each window and a few global tokens. | O(N·√N) |
| **Dynamic Token Clustering** | Iteratively merge nearby tokens into clusters; attention operates on the reduced set. | O(N·log N) |
| **Linear Attention (Performer)** | Approximate softmax with kernel feature maps; attention becomes linear in N. | O(N) |
| **Cross‑Modal Attention Fusion** | Separate modality‑specific streams attend locally, then exchange a small set of *cross‑tokens*. | O(N₁+N₂) |

In practice, **Sparse Sub‑Window + Global Prompt Token** provides the best trade‑off for AR scenarios: it preserves fine‑grained local detail while keeping latency under 30 ms on a mobile GPU.

---

## 6. Real‑Time Inference on Edge / AR Devices  

### 6.1 Deployment Stack  

| Component | Toolchain |
|-----------|-----------|
| **Model Export** | ONNX → TensorFlow Lite (TFLite) or Qualcomm SNPE |
| **Quantization** | Post‑training 8‑bit integer (PTQ) + per‑channel weight scaling |
| **Runtime** | TFLite GPU delegate or SNPE DSP runtime |
| **AR Integration** | Unity 3D + ARCore (Android) / ARKit (iOS) plugin |

### 6.2 Benchmarks on a Qualcomm Snapdragon 8 Gen 2  

| Model (Backbone) | Params | Quantization | Input Size | **Latency (ms)** | **Throughput (FPS)** | Power (mW) |
|------------------|--------|--------------|------------|------------------|----------------------|------------|
| **Point‑BERT‑Base** (12 L, 768 d) | 86 M | INT8 PTQ | 64 k points | 28.4 | 35 | 1 200 |