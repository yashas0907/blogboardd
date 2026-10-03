# Foundation Models for Aerial and Satellite Imagery  
## Scaling Vision Transformers to Geospatial Data  

*Published by the Computer Vision Technical Review*  

---  

## Introduction  

The explosion of high‑resolution optical, multispectral, and synthetic‑aperture‑radar (SAR) satellite constellations—such as **Sentinel‑2**, **Sentinel‑1**, **Landsat 8**, and the commercial constellations of Planet and Maxar—has turned the Earth’s surface into a continuously refreshed data lake.  At the same time, **Vision Transformers (ViTs)** have become the de‑facto backbone for large‑scale visual representation learning, thanks to their ability to model long‑range dependencies and to scale efficiently with data volume.  

Bridging these two trends yields **foundation models for geospatial imagery**: massive, self‑supervised ViTs pre‑trained on terabytes of satellite data, enriched with auxiliary GIS metadata, and capable of being prompted with natural‑language or coordinate‑based queries.  This tutorial walks through the entire pipeline—from dataset curation and self‑supervised pre‑training, through prompt engineering and efficient fine‑tuning for downstream tasks (land‑use classification, change detection, disaster assessment), to real‑time inference on cloud and edge platforms.  

---  

## 1. Foundations of Vision Transformers for Geospatial Data  

### 1.1 Why Vision Transformers Scale Better Than CNNs for Satellite Imagery  

| Aspect | Convolutional Neural Networks (CNNs) | Vision Transformers (ViTs) |
|--------|--------------------------------------|----------------------------|
| **Receptive field** | Grows slowly with depth; large kernels increase compute | Global self‑attention gives *instant* full‑image context |
| **Spectral flexibility** | Fixed channel mixing in early layers | Tokens can be defined per spectral band or per patch‑wise spectral vector |
| **Pre‑training transfer** | Strongly architecture‑specific | Architecture‑agnostic self‑supervised objectives (MAE, DINO) transfer across modalities |
| **Scalability** | Memory bound by depth × width | Linear‑ith‑quadratic trade‑off can be mitigated with hierarchical designs (Swin, PVT) |

Key papers that demonstrated ViT scalability include **Dosovitskiy et al., 2020** (ViT), **Liu et al., 2021** (Swin Transformer), and **Wang et al., 2022** (SatMAE).  

### 1.2 Multi‑Spectral and SAR Tokenization  

* **Optical / Multi‑Spectral** – Each pixel can be represented as a *spectral vector* (e.g., 13‑band Sentinel‑2). Patch tokenization proceeds by flattening a \(P \times P\) spatial window and concatenating its spectral channels, optionally followed by a linear projection.  
* **SAR** – Complex‑valued backscatter is split into magnitude and phase, or transformed into log‑ratio and texture features (e.g., GLCM). Tokens can be enriched with polarimetric decompositions (Pauli, Freeman‑Durden).  
* **Hybrid Tokens** – For co‑registered optical‑SAR pairs, concatenate optical and SAR token streams and let the self‑attention layers learn cross‑modal interactions (see **Zhou et al., 2022**, SAR‑ViT).  

---  

## 2. Pre‑training Strategies at Planetary Scale  

### 2.1 Datasets  

| Dataset | Modality | Spatial Resolution | Coverage | Size (TB) |
|---------|----------|-------------------|----------|-----------|
| **Sentinel‑2 L2A** (Copernicus) | Multi‑spectral (13 bands) | 10–60 m | Global, 2015‑present | ~2.8 |
| **Sentinel‑1 IW GRD** | SAR (VV, VH) | 5–20 m | Global, 2014‑present | ~1.5 |
| **Landsat 8 Level‑2** | Optical (11 bands) | 30 m | Global, 2013‑present | ~0.9 |
| **PlanetScope Daily** (commercial) | RGB + NIR | 3–5 m | Selected continents | ~3.0 |
| **OpenStreetMap (OSM) vector tiles** | GIS metadata (road, building) | Vector | Global | < 0.1 |

All datasets are pre‑processed to a common geographic tiling scheme (e.g., 256 × 256 px tiles at 10 m resolution) and stored in **Zarr** containers for efficient streaming.  

### 2.2 Self‑Supervised Objectives  

| Objective | Core Idea | Suitability for Geospatial Data |
|-----------|-----------|---------------------------------|
| **Masked Autoencoder (MAE)** – He et al., 2022 | Randomly mask a high proportion of patches; reconstruct pixel values. | Handles high redundancy in satellite imagery; works with multi‑spectral channels. |
| **DINO (self‑distillation)** – Caron et al., 2021 | Teacher–student network with contrastive loss, no negative pairs. | Encourages *semantic* clustering across seasons and illumination. |
| **CLIP‑style contrastive alignment** – Radford et al., 2021 | Align image tokens with text captions (e.g., “urban area”, “cropland”). | Enables **promptable** queries; requires curated caption datasets (e.g., from ESA’s Land Cover Climate Change Initiative). |
| **Cross‑modal MAE** – Wang et al., 2022 (SatMAE) | Mask patches in one modality (optical) and reconstruct from the other (SAR). | Learns robust representations for cloudy or night‑time conditions. |

A typical pre‑training schedule uses **Swin‑B** (Swin‑Base) as the backbone, a patch size of 4 × 4 px, and a masking ratio of 75 % for MAE. Training on 256 GPUs for 400 k steps (~2 weeks) yields a 1.2 B‑parameter foundation model that can be fine‑tuned with < 1 % of the downstream data.  

### 2.3 Incorporating GIS Metadata  

GIS vectors (roads, parcels, elevation) are rasterized to the same tile resolution and concatenated as *auxiliary channels*. Alternatively, a **dual‑encoder** architecture processes vector attributes through a lightweight MLP and injects the resulting embeddings into the ViT via **cross‑attention** layers (see **Kong et al., 2022**, GeoViT). This approach improves downstream performance on tasks that depend on spatial context, such as *road network extraction* or *building footprint delineation*.  

---  

## 3. Promptable Geospatial Queries  

### 3.1 Natural‑Language Prompts  

By fine‑tuning the image encoder together with a frozen language encoder (e.g., **BERT‑base**, Devlin et al., 2019) under a contrastive loss, the model learns a joint embedding space where queries such as  

> “Show me all wheat fields within 10 km of the Mekong River”  

can be transformed into a **text token** that is compared against tile embeddings. Retrieval is performed via an approximate nearest‑neighbor index (FAISS).  

### 3.2 Coordinate‑Based Descriptors  

For precise spatial queries, the model can be conditioned on a **geographic token** consisting of latitude, longitude, and optional radius. This token is passed through a sinusoidal positional encoder (similar to the original Transformer) and fused with the visual token stream via **cross‑attention**. The resulting representation enables *region‑specific* retrieval without needing a separate GIS database.  

### 3.3 Retrieval Pipeline  

1. **Tile embedding generation** – Run the foundation model on the entire archive (offline).  
2. **Index construction** – Store embeddings in a disk‑resident FAISS IVF‑PQ index.  
3. **Query encoding** – Encode the natural‑language or coordinate prompt.  
4. **Similarity search** – Retrieve top‑k tiles; optionally re‑rank with a lightweight classifier (e.g., land‑cover probability).  

---  

## 4. Efficient Fine‑tuning for Downstream Tasks  

Fine‑tuning follows a **parameter‑efficient** paradigm: either **Adapter** modules (Houlsby et al., 2019) or **LoRA** (Hu et al., 2021) are inserted into the transformer blocks, leaving the bulk of the pre‑trained weights frozen. This reduces GPU memory and enables rapid adaptation with as few as 500 labeled samples.  

### 4.1 Land‑Use / Land‑Cover Classification  

| Dataset | Classes | Resolution | mIoU (baseline) | mIoU (ViT‑Adapter) |
|---------|---------|------------|----------------|--------------------|
| **BigEarthNet** (Sumbul et al., 2019) | 19 | 10 m | 68.2 % | **77.5 %** |
| **DeepGlobe Land‑Cover** (Demir et al., 2018) | 7 | 30 m | 61.4 % | **70.1 %** |

Training uses cross‑entropy loss, AdamW (β₁=0.9, β₂=0.999), learning rate 2e‑4, batch size 256, and early stopping after 20 epochs.  

### 4.2 Change Detection  

Change detection benefits from **dual‑stream** encoders that process pre‑ and post‑event images in parallel, sharing the same ViT weights. The output tokens are concatenated and fed to a lightweight decoder that predicts a binary change mask.  

#### 4.2.1 Training Recipe  

| Step | Description | Hyper‑parameters |
|------|-------------|-------------------|
| **1. Data preparation** | Sample aligned image pairs (pre/post) at 10 m, mask clouds using Sentinel‑2 QA band. | Pair size: 256 × 256 px |
| **2. Tokenization** | Apply shared Swin‑B tokenizer; include SAR auxiliary channel for cloudy pairs. | Patch size: 4 × 4 px |
| **3. Encoder** | Frozen foundation ViT (Swin‑B) + LoRA adapters (rank = 4). | LoRA α = 32 |
| **4. Decoder** | 3‑layer Conv‑Transpose head (kernel = 3, stride = 2) → upsample to original resolution. | Dropout = 0.1 |
| **5. Loss** | Weighted binary cross‑entropy (weight = 3 for change class) + Dice loss (λ = 0.5). |  |
| **6. Optimizer** | AdamW with cosine annealing. | LR = 1e‑4, weight decay = 0.05 |
| **7. Augmentation** | Random rotation (±15°