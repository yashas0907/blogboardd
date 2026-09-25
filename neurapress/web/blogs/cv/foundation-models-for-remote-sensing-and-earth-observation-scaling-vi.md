# Foundation Models for Remote Sensing and Earth Observation  
## Scaling Vision Transformers to Satellite Imagery  

*Remote sensing is undergoing a paradigm shift. The convergence of massive multi‑spectral and SAR archives, self‑supervised learning, and the versatility of Vision Transformers (ViTs) is giving rise to foundation models that can be prompted, fine‑tuned, and deployed at scale. This tutorial walks through the entire pipeline—from pre‑training on petabyte‑scale satellite data to real‑time inference on edge devices—while highlighting practical recipes and recent research breakthroughs.*  

---  

## 1. Introduction  

Satellite imagery is intrinsically **multimodal** (optical, multi‑spectral, SAR), **spatio‑temporally rich**, and **geographically diverse**. Traditional convolutional pipelines excel when the target domain is narrow, but they struggle to generalize across sensors, resolutions, and tasks.  

Vision Transformers (ViT) and their hierarchical variants (e.g., Swin Transformer) have shown that **global self‑attention** can capture long‑range dependencies that are crucial for interpreting large‑scale scenes such as urban sprawls, flood extents, or deforestation fronts. When these architectures are trained on massive, unlabeled satellite corpora, they become **foundation models**—general‑purpose encoders that can be **prompted** with text, coordinates, or timestamps, and **efficiently fine‑tuned** for downstream Earth‑observation (EO) applications.  

The remainder of this post is organized around four pillars that together enable a production‑ready workflow:

1. **Large‑scale pre‑training** on multi‑spectral and SAR data.  
2. **Promptable multimodal queries** that fuse language, geo‑coordinates, and temporal context.  
3. **Efficient fine‑tuning** for tasks such as land‑cover mapping, change detection, and disaster assessment.  
4. **Real‑time inference** on edge or edge‑cloud platforms for rapid response.  

Each section provides a concise technical overview, concrete implementation tips, and a curated set of references to the state‑of‑the‑art literature.  

---  

## 2. Large‑Scale Pre‑Training on Multi‑Spectral and SAR Satellite Data  

### 2.1 Why Pre‑Train on Satellite Data?  

* **Domain shift** – Models pre‑trained on ImageNet (RGB, 224 × 224) see a very different distribution from 10‑m Sentinel‑2 or 0.5‑m PlanetScope images.  
* **Label scarcity** – High‑quality pixel‑wise annotations are expensive; self‑supervised objectives exploit the abundance of raw imagery.  
* **Cross‑modal synergy** – Jointly learning from optical and SAR streams encourages representations that are robust to clouds, illumination, and sensor noise.  

### 2.2 Data Sources & Pre‑Processing  

| Modality | Typical Sensors | Spatial Resolution | Spectral Bands | Public Archives (as of 2024) |
|----------|----------------|--------------------|----------------|------------------------------|
| Optical (RGB + NIR) | Sentinel‑2 MSI, Landsat‑8 OLI, PlanetScope | 10 m – 3 m | 4–12 (incl. SWIR) | ESA Copernicus Open Access Hub, USGS EarthExplorer |
| Multi‑Spectral (including SWIR) | WorldView‑3, PlanetScope | 0.3 m – 5 m | 8–16 | Maxar Open Data Program |
| SAR (C‑band, L‑band) | Sentinel‑1 SAR, ALOS‑2 PALSAR‑2 | 5 m – 25 m | Single (HH/VV) + polarimetric combos | Alaska Satellite Facility, ESA Sentinel‑1 Archive |

**Pre‑processing pipeline (recommended):**  

1. **Radiometric calibration** → Top‑of‑atmosphere reflectance (optical) or sigma‑0 (SAR).  
2. **Geometric co‑registration** → Align all modalities to a common grid (e.g., 10 m EPSG:4326).  
3. **Patch extraction** → Random 256 × 256 or 384 × 384 tiles; overlap 50 % for data efficiency.  
4. **Band selection & normalization** → Z‑score per band; optionally concatenate a *sensor‑type token* for modality identification.  

### 2.3 Self‑Supervised Objectives  

| Objective | Core Idea | Satellite‑Specific Adaptations |
|-----------|-----------|--------------------------------|
| **Masked Autoencoding (MAE)** – *He et al., 2022* | Randomly mask a high‑percentage of patches; decoder reconstructs pixels. | Masking can be **spectral‑aware** (mask entire bands) to force cross‑band reasoning. |
| **Contrastive Learning (e.g., DINO, MoCo‑v2)** – *Caron et al., 2021*; *He et al., 2020* | Pull together representations of augmented views; push apart others. | Use **cross‑modal augmentations** (optical ↔ SAR) as positive pairs, encouraging sensor‑invariant embeddings. |
| **Temporal Consistency (TS‑ViT)** – *Zhu et al., 2022* | Enforce that representations of the same location at different timestamps are close. | Leverages the natural revisit cycle of Sentinel‑2 (5 days) for unsupervised change detection pre‑training. |
| **Geo‑Contrastive (GeoCLR)** – *Li et al., 2023* | Positive pairs are geographically proximal; negatives are far apart. | Encourages spatial smoothness while preserving discriminative power for fine‑grained objects. |

### 2.4 Architectural Choices  

| Architecture | Strengths for EO | Typical Configurations |
|--------------|------------------|------------------------|
| **ViT‑B/16** – *Dosovitskiy et al., 2020* | Simple, global attention; easy to scale. | Patch size 16 px, 12 layers, 768 hidden dim. |
| **Swin‑T** – *Liu et al., 2021* | Hierarchical, shifted windows → better locality & memory. | Window size 7, 4 stages, 96–384 hidden dims. |
| **SatViT** – *Gao et al., 2023* | Introduces *spectral tokens* and *sensor embeddings*. | Multi‑spectral + SAR token streams merged after stage‑2. |
| **Hybrid Conv‑ViT** – *Xie et al., 2022* (MAE) | Early convolutional stem reduces token count for high‑res tiles. | 3×3 conv stem → 14 × 14 tokens for 224 px input. |

**Practical tip:** For 10 m Sentinel‑2 tiles (256 × 256), a hybrid stem with a 4‑× downsampling conv reduces the token count to 64 × 64, keeping GPU memory under 24 GB even for 12‑layer ViTs.

### 2.5 Pre‑Training Benchmarks  

| Model | Pre‑Training Data | Objective | FLOPs (B) | Downstream mIoU (Land‑Cover) |
|-------|-------------------|-----------|-----------|------------------------------|
| ViT‑B/16 (MAE) | 2 M Sentinel‑2 patches (RGB+NIR) | 75 % mask MAE | 5.5 | 71.2 % |
| Swin‑T (DINO) | 1.5 M Sentinel‑1 SAR + Sentinel‑2 co‑registered | Contrastive + cross‑modal | 6.8 | 68.5 % |
| SatViT‑L (Temporal) | 3 M multi‑spectral + SAR time series (5‑year span) | TS‑ViT + MAE hybrid | 12.3 | **78.4 %** |
| Hybrid Conv‑ViT (GeoCLR) | 4 M PlanetScope + WorldView‑3 (high‑res) | Geo‑contrastive | 9.1 | 73.6 % |

> **Takeaway:** Combining **temporal** and **cross‑modal** objectives yields the highest land‑cover performance, confirming that satellite foundation models benefit from the intrinsic spatio‑temporal structure of EO data.

---  

## 3. Promptable Multimodal Queries  

### 3.1 From Fixed Classifiers to Open‑Ended Prompts  

Instead of a static softmax over a fixed label set, a **promptable foundation model** receives a **textual description** (or other modality) that conditions the attention flow. This mirrors CLIP’s image‑text alignment but extends it to **geo‑coordinates** and **time stamps**.  

**Prompt format (example):**  

```
[TEXT] "urban residential area" 
[COORD] (lat: 34.0522, lon: -118.2437) 
[TIME] "2023-08-15"
```

The model encodes each component:

* **Text encoder** – a lightweight BERT or RoBERTa (e.g., `roberta-base`).  
* **Coordinate encoder** – sinusoidal positional embedding of latitude/longitude, optionally enriched with elevation or administrative boundary IDs.  
* **Temporal encoder** – sinusoidal embedding of day‑of‑year + year offset.  

All embeddings are **projected** into the same latent dimension and **added** to the visual token stream as **prompt tokens** before the first transformer block.

### 3.2 Cross‑Modal Attention Mechanics  

1. **Prompt token injection** – Insert *N* prompt tokens (e.g., 4) at the beginning of the token sequence.  
2. **Self‑attention** – Prompt tokens attend to every visual patch, allowing the text/geo context to modulate feature extraction globally.  
3. **Fusion layer** – After the final transformer block, a **multimodal pooling** (CLS token + prompt tokens) is passed to a task‑specific head (e.g., segmentation decoder).  

> **Implementation note:** In PyTorch, this can be realized by concatenating the prompt embeddings to the patch embeddings before feeding them to `nn.TransformerEncoder`.  

### 3.3 Use Cases  

| Scenario | Prompt Example | Expected Output |
|----------|----------------|-----------------|
| **Land‑cover query** | `"forest canopy"` + coordinates of a region + `"summer 2022"` | Pixel‑wise probability map of forest canopy for the specified area and season. |
| **Disaster assessment** | `"flooded roads"` + coordinates of a city + `"2024-09-10"` | Binary mask highlighting roads inundated on the given date. |
| **Temporal trend** | `"average NDVI"` + coordinates of a watershed + `"last 5 years"` | Time‑series plot generated from model‑derived NDVI estimates. |
| **Cross‑sensor retrieval** | `"high‑resolution SAR"` + `"area of interest"` | Retrieval of the most relevant SAR tiles from an archive, ranked by similarity to the prompt. |

### 3.4 Prompt Engineering Tips  

| Tip | Reason |
|-----|--------|
| **Use domain‑specific vocabularies** (e.g., “cropland”, “bare soil”) rather than generic synonyms. | Improves alignment with the satellite visual semantics learned during pre‑training. |
| **Include elevation or land‑use codes** as auxiliary coordinates. | Adds discriminative signal for ambiguous regions (e.g., mountainous vs. flat). |
| **Leverage temporal windows** (`"last month"`, `"peak summer"`). | Allows the model to attend to season‑specific spectral signatures. |
| **Fine‑tune the text encoder** on a small corpus of EO captions (e.g., Sentinel‑2 product descriptions).