# Foundation Models for Earth Observation  
## Scaling Vision Transformers for Satellite and Aerial Imagery Analysis  

---

## Introduction  

The rapid proliferation of high‑resolution satellite constellations, unmanned aerial vehicles (UAVs), and ground‑based remote‑sensing platforms has produced petabytes of geospatial data every year. Traditional convolutional pipelines struggle to keep pace with the **spectral diversity**, **spatial scale**, and **real‑time latency** requirements of modern Earth‑observation (EO) applications such as land‑cover mapping, disaster response, and precision agriculture.  

**Vision Transformers (ViTs)**—originally introduced for natural‑image classification—have emerged as a unifying architecture for *foundation models* that can be pre‑trained on massive, multimodal EO datasets and subsequently adapted to a wide spectrum of downstream tasks. This tutorial walks through a complete workflow:

1. **Large‑scale multimodal pre‑training** (multispectral, hyperspectral, SAR, LiDAR).  
2. **Promptable geospatial queries** using natural language and coordinate descriptors.  
3. **Efficient fine‑tuning** for specific tasks (land‑cover, change detection, disaster assessment, object detection).  
4. **Model compression** (weight pruning, quantization) for edge inference on UAVs, low‑orbit satellites, and ground stations.  

By the end of the article you will understand how to assemble a production‑ready pipeline that balances accuracy, scalability, and latency.

---

## 1. Large‑Scale Multimodal Pre‑training  

### 1.1 Why Multimodality Matters  

Remote‑sensing sensors capture complementary physical phenomena:

| Modality | Typical Bands / Resolution | Physical Insight |
|----------|---------------------------|------------------|
| **Multispectral (MS)** | 4–13 bands, 10–30 m | Vegetation indices, water quality |
| **Hyperspectral (HS)** | 100–400 bands, 5–30 m | Material identification, mineralogy |
| **Synthetic Aperture Radar (SAR)** | L‑, C‑, X‑band, 1–30 m | Surface roughness, moisture, all‑weather imaging |
| **LiDAR** | 3‑30 m (point density) | Elevation, canopy structure |

Training a single transformer on all modalities forces the model to learn **cross‑modal embeddings** that generalize beyond any single sensor. Recent works such as **SatMAE** (Wang et al., 2022) and **GeoMAE** (Zhang et al., 2023) demonstrate that masked‑autoencoding pre‑training on heterogeneous EO collections yields representations that transfer to both optical and SAR tasks with minimal fine‑tuning.

### 1.2 Dataset Curation  

A robust pre‑training corpus should satisfy three criteria:

1. **Scale** – >10 M image patches (≈1 TB of raw data).  
2. **Diversity** – Global coverage, multiple seasons, and varied sensor configurations.  
3. **Alignment** – Co‑registered multimodal stacks (e.g., Sentinel‑2 MS + Sentinel‑1 SAR + GEDI LiDAR).  

Publicly available sources that meet these criteria include:

* **Sentinel‑2** (MS, 13 bands, 10 m) – ESA Copernicus Open Access Hub.  
* **Sentinel‑1** (C‑band SAR, 5 m ground‑range) – ESA.  
* **Landsat‑8/9** (MS + thermal) – USGS EarthExplorer.  
* **AVIRIS‑NG** (HS, 224 bands, 4 m) – NASA.  
* **GEDI** (LiDAR waveform) – NASA Lidar Archive.  

When building the pre‑training set, adopt a *tile‑based* pipeline:

```text
1. Tile the globe into 256 × 256 px patches at a common ground sample distance (e.g., 10 m).
2. For each tile, retrieve all available modalities; fill missing bands with a sentinel value.
3. Store patches in sharded TFRecord/Parquet files with metadata (timestamp, sensor, cloud‑mask).
```

### 1.3 Architecture Choices  

A **Hybrid Vision Transformer (HViT)**—convolutional stem + transformer encoder—offers a good trade‑off between local texture extraction and global attention. Typical configurations for EO pre‑training:

| Model | Params | Patch Size | Positional Encoding | Spectral Fusion |
|-------|--------|------------|---------------------|-----------------|
| ViT‑B/16 (Hybrid) | 86 M | 16 × 16 | Learnable 2‑D + spectral embeddings | Early (concatenate bands before patching) |
| Swin‑L (Shifted Window) | 197 M | 4 × 4 windows | Relative window bias | Mid‑level (after stage‑2) |
| Conv‑ViT‑HS (custom) | 120 M | 8 × 8 | Rotary embeddings | Late (cross‑attention between modalities) |

**Spectral embeddings** can be injected by adding a learnable vector per band (similar to token type embeddings in BERT). This enables the model to differentiate, for example, a SAR backscatter channel from a NIR band.

### 1.4 Pre‑training Objectives  

1. **Masked Autoencoding (MAE)** – Randomly mask 75 % of patches across all modalities; reconstruct the missing pixels with a lightweight decoder.  
2. **Cross‑Modal Contrastive Learning** – Treat each modality as a “view” of the same geographic location; maximize agreement between their CLS tokens (inspired by CLIP, Radford et al., 2021).  
3. **Geospatial Positional Regression** – Predict latitude/longitude from the CLS token to encourage spatial awareness (Zhu et al., 2020).  

A combined loss often yields the best downstream transfer:

\[
\mathcal{L}= \lambda_{MAE}\mathcal{L}_{MAE}+ \lambda_{CL}\mathcal{L}_{CL}+ \lambda_{GP}\mathcal{L}_{GP}
\]

Typical hyper‑parameters: \(\lambda_{MAE}=1.0\), \(\lambda_{CL}=0.5\), \(\lambda_{GP}=0.1\).

---

## 2. Promptable Geospatial Queries  

### 2.1 From Text to Pixels  

Large language‑vision models (e.g., **FLAVA**, Singh et al., 2022) have shown that a *prompt* can steer a frozen visual encoder toward a specific concept. In EO, prompts can be:

* **Natural‑language** – “urban areas with rooftop solar panels”.  
* **Coordinate‑based** – “polygon covering (34.05° N, ‑118.25° W) to (34.06° N, ‑118.24° W)”.  

By concatenating a **text token** to the CLS token and feeding the combined sequence through a lightweight cross‑modal transformer, the model produces a **query‑aware feature map**. The resulting map can be thresholded or passed to a downstream classifier.

### 2.2 Prompt Engineering for EO  

| Prompt Type | Example | Recommended Tokenization |
|-------------|---------|---------------------------|
| **Descriptive** | “dense forest with canopy height > 30 m” | Use domain‑specific vocab (e.g., “canopy”, “NDVI”). |
| **Temporal** | “areas that changed from water to land between 2020‑01‑01 and 2021‑01‑01” | Encode dates as special tokens (e.g., `<DATE_20200101>`). |
| **Spatial** | “all pixels within 5 km of the coordinates (45.0 N, ‑122.0 W)” | Convert radius to a mask and prepend a `<RADIUS_5KM>` token. |

Fine‑tuning the prompt encoder on a small **prompt‑response dataset** (≈5 k examples) dramatically improves retrieval precision (Zhou et al., 2023).

### 2.3 Retrieval Pipeline  

1. **Encode** the whole scene with the frozen ViT backbone → token grid.  
2. **Encode** the prompt → query vector.  
3. **Compute** cosine similarity between query and each spatial token.  
4. **Aggregate** top‑k tokens to produce a binary mask or heatmap.  

Because similarity is computed on the GPU, the pipeline can answer thousands of queries per second on a single A100, making it suitable for interactive GIS portals.

---

## 3. Efficient Fine‑tuning for Downstream Tasks  

### 3.1 Parameter‑Efficient Transfer  

Full fine‑tuning of a 200 M‑parameter ViT is often unnecessary and costly. The following strategies retain most of the pre‑trained knowledge while adapting to a new task:

| Strategy | Mechanism | Typical Overhead |
|----------|-----------|------------------|
| **Adapter Layers** (Houlsby et al., 2019) | Small bottleneck MLP inserted after each transformer block | < 5 % of total params |
| **Prompt Tuning** (Liu et al., 2021) | Learnable token(s) prepended to input sequence | < 0.1 % of params |
| **LoRA** (Hu et al., 2021) | Low‑rank decomposition of weight updates | < 1 % of params |
| **BitFit** (Zaken et al., 2022) | Only bias terms are updated | < 0.01 % of params |

Empirically, **Adapter + Prompt** combinations achieve > 95 % of full fine‑tuning performance on land‑cover classification while reducing GPU memory by 40 %.

### 3.2 Task‑Specific Heads  

| Task | Head Architecture | Loss |
|------|-------------------|------|
| **Land‑Cover Classification** | Global average pooling → linear classifier (C classes) | Cross‑entropy |
| **Change Detection** | Siamese ViT (shared weights) → subtraction → 1 × 1 conv → sigmoid | Binary cross‑entropy + Dice |
| **Disaster Assessment** | Multi‑scale feature pyramid → per‑pixel regression (damage score) | Smooth L1 |
| **Object Detection** | ViT‑Backbone + Deformable DETR decoder (Carion et al., 2020) | Hungarian loss (classification + bbox) |

When the downstream dataset is small (< 1 k annotated images), **self‑training** with pseudo‑labels generated by the foundation model can boost performance (Zhou et al., 2022).

### 3‑Step Fine‑tuning Workflow  

1. **Freeze** the backbone; train only the task head for 5–10 epochs.  
2. **Unfreeze** adapters or LoRA modules; continue training with a lower learning rate (1e‑4).  
3. **Optional**: fine‑tune the entire model for a few epochs if GPU budget permits.  

Early stopping based on a validation metric (e.g., mIoU for segmentation) prevents over‑fitting.

---

## 4. Model Compression & Weight Pruning for Edge Deployment  

### 4.1 Structured Weight Pruning  

Pruning removes redundant weights, reducing FLOPs and memory. A **gradual magnitude‑based pruning** schedule (e.g., 0 % → 30 % → 50 % sparsity over 30 epochs) works well for ViTs (Mou et al., 2023).  

**Key steps**:

1. **Initialize** a binary mask for each weight matrix.  
2. **Update** masks every N steps based on current magnitude.  
3. **Fine‑tune** the pruned model for 5–10 epochs to recover accuracy.  

Structured pruning (e.g., pruning entire heads or MLP columns) yields hardware‑friendly sparsity patterns that can be accelerated on GPUs and specialized AI chips.

### 4.2 Quantization  

Post‑training **8‑bit integer (INT8)** quantization, followed by **dynamic range calibration** on a representative subset (≈1 % of the training data), typically incurs < 1 % accuracy loss (Jacob et al., 2018). For ultra‑low power UAVs, **4‑bit quantization** with mixed‑precision fine‑tuning (Zhou et al., 2022) can be employed, albeit with a modest mIoU drop (~2–3 %).

### 4.3 Edge‑Ready Formats  

Export the compressed model to **ONNX** or **TensorRT** (for NVIDIA Jetson) and **TVM** (for ARM‑based processors). These runtimes support sparse kernels and INT8/4‑bit