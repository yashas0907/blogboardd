# Foundation Models for Time‑Series and Sequential Data  
**Scaling Forecasting, Classification, and Anomaly Detection**

---

## Introduction  

Time‑series data permeate every industry—from electricity grids and financial markets to industrial IoT and healthcare. Historically, practitioners built **task‑specific models** (ARIMA, Prophet, LSTMs) that required hand‑crafted features and extensive hyper‑parameter tuning for each new dataset.  

The emergence of **foundation models**—large, self‑supervised neural networks pretrained on massive, heterogeneous corpora—has transformed natural language processing and computer vision. The same paradigm is now reshaping time‑series analytics. By pretraining on billions of multivariate observations, a single model can acquire generic temporal representations that are **fine‑tuned** for downstream forecasting, classification, or anomaly detection with far fewer labeled examples.

This tutorial walks through the entire pipeline:

1. **Self‑supervised pretraining** on massive multivariate time‑series collections.  
2. **Scalable temporal transformer architectures** and the distributed training pipelines that make them feasible.  
3. **Fine‑tuning strategies** for the three canonical downstream tasks.  
4. **Benchmarks, evaluation metrics, and real‑world deployment considerations** (latency, compression, monitoring).  

By the end you will understand how to build, evaluate, and ship a foundation model for sequential data in production.

---

## 1. Self‑Supervised Pretraining on Massive Multivariate Time‑Series Datasets  

### 1.1 Why Self‑Supervision?  

* Labeled time‑series are scarce and expensive (e.g., fault annotations in manufacturing).  
* Unlabeled streams are abundant: sensor logs, market tick data, server metrics, etc.  
* Self‑supervised objectives encourage the model to learn **temporal invariances**, **cross‑channel dependencies**, and **long‑range dynamics** without explicit supervision.

### 1.2 Representative Pretraining Objectives  

| Objective | Core Idea | Typical Implementation |
|-----------|-----------|------------------------|
| **Masked Modeling** (e.g., **TS‑Mask**, **TS2Vec**) | Randomly mask contiguous windows across channels and ask the model to reconstruct them. | Mask ratio 15‑30 %; loss = MSE on masked tokens. |
| **Contrastive Predictive Coding (CPC)** | Predict future latent representations from past context and maximize agreement with true future embeddings. | InfoNCE loss; negative samples drawn from other time steps or series. |
| **Temporal Order Prediction** | Shuffle subsequences and train the model to recover the correct order. | Cross‑entropy over permutation classes. |
| **Multi‑Resolution Forecasting** | Predict coarse‑grained aggregates (e.g., daily mean) from fine‑grained inputs. | Hierarchical loss combining MSE at multiple scales. |
| **Cross‑Channel Inpainting** | Hide an entire channel and ask the model to infer it from the remaining channels. | Channel‑wise MSE or MAE. |

Recent work demonstrates that **masked modeling** combined with **contrastive alignment** yields the most robust universal embeddings (Yue et al., 2022; Liu et al., 2023).

### 1.3 Data Sources for Pretraining  

| Source | Scale | Modalities | Example |
|--------|-------|------------|---------|
| **Industrial IoT** (e.g., Siemens, GE) | > 10 B samples | Vibration, temperature, pressure | Predictive maintenance logs |
| **Financial Tick Streams** (NASDAQ, CME) | > 5 B samples | Price, volume, order‑book depth | Market microstructure |
| **Smart‑City Sensors** (traffic, air quality) | > 2 B samples | Vehicle counts, pollutants | Urban planning |
| **Public Cloud Metrics** (AWS, Azure) | > 1 B samples | CPU, memory, network I/O | Cloud‑ops observability |
| **Healthcare Wearables** (Fitbit, Apple) | > 500 M samples | Heart rate, steps, SpO₂ | Remote patient monitoring |

A practical pretraining pipeline ingests these streams in **Parquet** or **TFRecord** format, shuffles at the **record‑level**, and applies **on‑the‑fly augmentations** (jitter, scaling, time warping) to improve robustness.

---

## 2. Scalable Temporal Transformer Architectures & Distributed Training  

### 2.1 Temporal Transformers: From Vanilla to Efficient Variants  

| Architecture | Key Innovation | Complexity | Typical Use‑Case |
|--------------|----------------|------------|------------------|
| **Vanilla Time‑Series Transformer (TST)** (Zhou et al., 2020) | Full self‑attention over all timestamps | O(L²) | Short‑to‑medium horizons (L ≤ 512) |
| **Informer** (Zhou et al., 2021) | ProbSparse self‑attention + distilling | O(L log L) | Long sequences (L ≈ 10k) |
| **PatchTST** (Zhou et al., 2022) | Patch embedding + channel‑independent attention | O(L) per patch | Multivariate series with many channels |
| **Time‑Series Transformer with Linear Attention (Linformer)** (Wang et al., 2021) | Low‑rank projection of K/V | O(L) | Real‑time inference |
| **Hierarchical Temporal Encoder (HTE)** (Li et al., 2023) | Multi‑scale pooling + cross‑scale attention | O(L log L) | Multi‑resolution forecasting |
| **Anomaly Transformer** (Xu et al., 2021) | Dual‑stage attention for reconstruction & prediction | O(L²) (but pruned) | Anomaly detection |

**PatchTST** has become the de‑facto backbone for many foundation models because it treats each channel independently during attention, dramatically reducing memory while preserving cross‑channel interactions via **channel‑wise feed‑forward layers**.

### 2.2 Distributed Training Pipelines  

1. **Data Parallelism** – Replicate the model across GPUs/TPUs; each worker processes a distinct mini‑batch.  
2. **Model Parallelism** – Split the attention matrix across devices for extremely long sequences (e.g., > 50 k steps).  
3. **Pipeline Parallelism** – Partition the transformer layers into stages; useful when model size exceeds a single device’s memory.  

A typical production configuration on a **8‑node GPU cluster (NVIDIA A100, 40 GB)**:

| Component | Configuration |
|-----------|----------------|
| **Framework** | PyTorch 2.0 + **torch.distributed** (NCCL backend) |
| **Mixed‑Precision** | FP16 with **torch.cuda.amp** (≈ 2× speedup) |
| **Optimizer** | AdamW with cosine‑annealing LR schedule |
| **Gradient Accumulation** | 4 steps to achieve effective batch size 4096 |
| **Checkpointing** | ZeRO‑3 (DeepSpeed) for memory‑efficient sharding |
| **Logging** | TensorBoard + Weights & Biases for loss curves, attention maps |

**Best practice:** keep the **sequence length** constant within a mini‑batch to avoid padding overhead, and use **dynamic bucketing** to group series of similar lengths.

---

## 3. Fine‑Tuning Strategies for Downstream Tasks  

### 3.1 Forecasting  

1. **Head Design** – Append a **Temporal Convolutional Decoder** (TCN) or a simple **linear projection** per horizon.  
2. **Loss Functions** – Combine **MAE** (robust to outliers) with **SMAPE** for relative error.  
3. **Curriculum Learning** – Start with short horizons (e.g., 1‑step) and gradually increase to the target horizon.  
4. **Ensemble of Checkpoints** – Average predictions from the last *k* checkpoints to reduce variance.

**Tip:** For **hierarchical forecasting** (e.g., product‑level → category‑level), fine‑tune a shared encoder and attach **hierarchical heads** that enforce coherence via a **bottom‑up loss** (see Oreshkin et al., 2020, N‑BEATS).

### 3.2 Classification  

1. **Pooling** – Use **attention‑weighted pooling** over the encoder outputs to obtain a fixed‑size representation.  
2. **Classifier Head** – A shallow MLP (2‑3 layers) with **softmax** for multi‑class or **sigmoid** for multi‑label tasks.  
3. **Few‑Shot Adaptation** – Freeze the encoder and train only the head on < 100 labeled series; optionally apply **parameter‑efficient adapters** (Houlsby et al., 2019).  
4. **Contrastive Fine‑Tuning** – Add a **supervised contrastive loss** (Khosla et al., 2020) to pull together samples of the same class.

### 3.3 Anomaly Detection  

| Approach | How It Uses the Foundation Model |
|----------|-----------------------------------|
| **Reconstruction‑Based** | Pass the series through the encoder‑decoder; high reconstruction error → anomaly. |
| **Prediction‑Based** | Forecast the next window; large deviation from observation flags anomaly. |
| **Hybrid Attention Scores** | Leverage the **attention‑based anomaly score** from Anomaly Transformer (Xu et al., 2021). |
| **Embedding‑Distance** | Compute distance between current embedding and a **reference distribution** (e.g., Gaussian Mixture) learned from normal data. |

Fine‑tuning typically involves **unsupervised adaptation** on a clean subset, followed by a **threshold calibration** using a small labeled validation set (e.g., Precision‑Recall curve).

---

## 4. Benchmarks, Evaluation Metrics, and Real‑World Deployment Considerations  

### 4.1 Benchmark Suite  

| Dataset | Domain | #Series | Horizon (Steps) | Task | Primary Metric(s) | Baseline (PatchTST) | Baseline (Informer) |
|---------|--------|---------|-----------------|------|-------------------|---------------------|----------------------|
| **M4** | Business & Economics | 100 000 | 1‑18 (monthly) | Forecasting | sMAPE, OWA | 9.8 % (sMAPE) | 10.4 % |
| **M5** | Retail Sales | 42 000 | 28 (daily) | Forecasting | RMSSE, MAE | 0.71 (RMSSE) | 0.77 |
| **ETTh1** | Electricity | 7 | 96 (15 min) | Forecasting | MSE, MAE | 0.42 (MSE) | 0.48 |
| **ETTm2** | Electricity | 7 | 96 (15 min) | Forecasting | MSE, MAE | 0.36 | 0.41 |
| **Traffic** | Transportation | 862 | 96 (15 min) | Forecasting | MSE, MAE | 0.45 | 0.51 |
| **Solar‑10min** | Energy | 137 | 96 (10 min) | Forecasting | MSE, MAE | 0.28 | 0.33 |
| **UCR Archive (ECG200)** | Healthcare | 200 | — | Classification | Accuracy, F1 | 0.96 | 0.94 |
| **UCI HAR** | Wearables | 10 299 | — | Classification | Accuracy | 0.95 | 0.92 |
| **SMD (Server Machine Dataset)** |