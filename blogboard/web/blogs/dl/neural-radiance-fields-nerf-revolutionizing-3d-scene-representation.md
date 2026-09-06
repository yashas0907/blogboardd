# Neural Radiance Fields (NeRF): Revolutionizing 3D Scene Representation  

*Published by the Deep Learning Review*  

---  

## Introduction  

Three‑dimensional (3‑D) content lies at the heart of emerging technologies such as augmented reality (AR), virtual reality (VR), autonomous robotics, and photorealistic rendering. For decades, the graphics pipeline has relied on explicit geometry (meshes, point clouds) and handcrafted shading models. **Neural Radiance Fields (NeRF)** upended this paradigm by showing that a fully‑connected multilayer perceptron (MLP) can implicitly encode a scene’s geometry **and** appearance directly from a sparse set of calibrated images. Since the seminal work of Mildenhall *et al.* (2020), a vibrant ecosystem of extensions—Instant‑NGP, mip‑NeRF, dynamic NeRFs, and more—has turned NeRF from a research curiosity into a practical tool for real‑world applications.

This tutorial walks through the core concepts, training pipelines, and data requirements of NeRF, explores its most impactful applications, surveys the latest extensions, and offers concrete recommendations for practitioners looking to adopt the technology today.  

---  

## 1. Fundamentals of NeRF and Volumetric Rendering  

### 1.1 What a NeRF Represents  

A **Neural Radiance Field** is a continuous function  

\[
F_\theta : (\mathbf{x}, \mathbf{d}) \mapsto (\mathbf{c}, \sigma)
\]

that maps a 3‑D location **x** ∈ ℝ³ and a 2‑D viewing direction **d** ∈ ℝ² (often encoded as a unit vector) to:

* **c** – RGB color (radiance) emitted from that point toward the direction **d**.  
* **σ** – Volume density (opacity) at **x**, governing how much light is absorbed or scattered.  

The parameters **θ** are learned by an MLP, typically 8–10 layers wide, that is trained to reproduce the observed pixel colors of a set of input photographs.

### 1.2 Volumetric Rendering Equation  

NeRF adopts classic **volume rendering** (Blinn & Newell, 1995) to synthesize a pixel color **Ĉ(r)** along a camera ray **r(t) = o + t·d**, where **o** is the camera origin and **d** the ray direction. The rendering integral is discretized into **N** sampled points {t₁,…,t_N}:

\[
\hat{C}(r) = \sum_{i=1}^{N} T_i \, (1 - \exp(-\sigma_i \delta_i)) \, \mathbf{c}_i
\]

where  

* **σ_i**, **c_i** = Fθ(r(t_i), d)  
* **δ_i = t_{i+1} - t_i** (distance between consecutive samples)  
* **T_i = \exp\!\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)** is the accumulated transmittance up to sample *i*.  

This formulation naturally handles view‑dependent effects (specularities, translucency) because the color output depends on the viewing direction **d**.

### 1.3 Hierarchical Sampling  

Training a naïve uniform sampling of points along each ray is inefficient. The original NeRF introduced a **two‑stage hierarchical sampling**:

1. **Coarse network** predicts a rough density distribution, yielding a probability density function (PDF) over depth.  
2. **Fine network** draws additional samples from this PDF, concentrating computation where the scene actually emits light.  

The coarse‑fine strategy reduces the number of required samples (typically 64 coarse + 128 fine) while preserving high‑frequency detail.

---  

## 2. Training Pipelines and Data Requirements  

### 2.1 Input Data  

| Requirement | Typical Specification | Why It Matters |
|-------------|----------------------|----------------|
| **Calibrated RGB images** | ≥ 20–100 views, 800 × 800 px or higher | Provides diverse viewpoints for the network to infer geometry. |
| **Camera poses** | Extrinsic (R, t) and intrinsic (focal length, principal point) matrices | Accurate pose information is essential for correct ray construction. |
| **Exposure & color balance** | Linearized (e.g., RAW) or gamma‑corrected with known tone‑mapping | Non‑linearities corrupt the radiance relationship and hinder convergence. |
| **Scene coverage** | Uniform angular distribution, minimal occlusion gaps | Sparse regions lead to “holes” in the reconstructed volume. |

Pose estimation can be obtained via structure‑from‑motion pipelines such as COLMAP (Schönberger & Frahm, 2016) or ARKit/ARCore for handheld capture.

### 2.2 Pre‑processing  

1. **Linearize** images (undo gamma, apply camera response function).  
2. **Resize** to a manageable resolution (e.g., 800 × 800) to balance memory vs. detail.  
3. **Normalize** ray coordinates to the unit cube ([-1, 1]³) for stable MLP training.  

### 2.3 Training Loop Overview  

```text
for epoch in 1..E:
    sample a batch of rays (origin, direction) from random training images
    for each ray:
        sample N_coarse points uniformly in depth
        evaluate coarse MLP → (c_coarse, σ_coarse)
        compute coarse weights → PDF over depth
        sample N_fine points from PDF
        evaluate fine MLP → (c_fine, σ_fine)
        render coarse and fine colors via volume rendering
    compute L2 loss between rendered colors and ground‑truth pixels
    back‑propagate and update θ with Adam (β1=0.9, β2=0.999)
```

Key hyper‑parameters (learning rate, number of samples, batch size) are often set as in the original paper: **lr = 5e‑4**, **batch = 1024** rays, **E ≈ 250k** iterations for high‑quality results.

### 2.4 Common Pitfalls  

| Symptom | Likely Cause | Remedy |
|---------|--------------|--------|
| Blurry reconstructions | Insufficient fine samples or low learning rate | Increase N_fine (e.g., 256) and/or decay learning rate slower. |
| “Floating” artifacts near edges | Inaccurate camera intrinsics | Re‑calibrate or refine poses using bundle adjustment. |
| Slow convergence (>48 h) | Large image resolution + vanilla MLP | Switch to a more efficient encoding (hash grid, spherical‑harmonics) (see §4). |

---  

## 3. Applications in AR/VR and View Synthesis  

### 3.1 Real‑Time View Synthesis for VR  

NeRF can generate photorealistic novel views at interactive rates when combined with **GPU‑accelerated inference** and **compact scene encodings**. In VR headsets, a pre‑computed NeRF of a static environment enables **six‑degree‑of‑freedom (6‑DoF) navigation** without the need for explicit meshes. Recent demos (e.g., Google Research’s “NeRF‑VR”) have shown sub‑30 ms latency on modern mobile GPUs.

### 3.2 AR Object Insertion  

Because NeRF models view‑dependent reflectance, it can be used to **relight virtual objects** consistently with the captured environment. By extracting the **environment radiance field** from a NeRF, AR pipelines can compute realistic illumination maps for inserted assets, reducing the “flat” look typical of image‑based lighting.

### 3.3 Telepresence & Remote Collaboration  

Dynamic NeRFs (Section 4.3) enable **real‑time capture of moving participants**, allowing remote collaborators to view a volumetric avatar from any angle. When combined with low‑latency streaming (e.g., NVIDIA’s Omniverse), NeRF‑based telepresence offers a compelling alternative to point‑cloud avatars.

### 3.4 Content Creation & Gaming  

Game studios are experimenting with **NeRF‑based level design**: artists capture a physical set, train a NeRF, and then import the field into a game engine as a background skybox or interactive backdrop. The continuous representation eliminates texture seams and LOD popping.

---  

## 4. Recent Extensions  

Since 2020, a wave of research has tackled NeRF’s main limitations: training speed, memory footprint, anti‑aliasing, and handling of dynamics.

### 4.1 Instant‑NGP (Neural Graphics Primitives)  

**Instant‑NGP** (Müller *et al.*, 2022) replaces the positional encoding of NeRF with a **multiresolution hash table** that maps 3‑D coordinates to a high‑dimensional feature vector. This yields:

* **Training in seconds** (≈ 5 s for a 100‑image scene on an RTX 3080).  
* **Memory usage < 1 GB** for a full‑resolution field.  

The hash‑grid encoding preserves the expressive power of the original sinusoidal embedding while enabling fast lookup and gradient propagation.

### 4.2 mip‑NeRF  

**mip‑NeRF** (Barron *et al.*, 2021) addresses **aliasing** caused by undersampling high‑frequency textures when rendering at low resolution. It treats each sampled point as a **cone** rather than a ray, integrating over a Gaussian‑like footprint. Benefits include:

* **Scale‑aware rendering**—smooth results when zooming out.  
* **Improved PSNR** (≈ 0.5 dB gain) on the Blender dataset.  

The method also introduces **integrated positional encoding**, which analytically averages the sinusoidal basis over the cone’s volume.

### 4.3 Dynamic NeRFs  

Modeling **time‑varying scenes** requires extending the static formulation to a spatio‑temporal field **Fθ(x, d, t)**. Several approaches have emerged:

| Method | Core Idea | Notable Contributions |
|--------|-----------|------------------------|
| **D-NeRF** (Park *et al.*, 2021) | Decompose scene into a static NeRF + a low‑dimensional latent motion vector per frame; use a small MLP to map time → latent code. | Enables smooth interpolation of motion between captured frames. |
| **NeRF‑Time** (Niklaus *et al.*, 2022) | Treat time as an additional input dimension with sinusoidal encoding; train on densely sampled video. | Handles fast, non‑rigid motion (e.g., facial expressions). |
| **HyperNeRF** (Peng *et al.*, 2022) | Learn a **hypernetwork** that generates NeRF weights conditioned on a latent pose vector; supports articulated objects. | Provides compact representation for many pose variations. |
| **ST-NeRF** (Li *et al.*, 2023) | Combine spatial hash encoding (Instant‑NGP) with a temporal hash to achieve **real‑time dynamic capture** on consumer GPUs. | Achieves 30 fps rendering of a moving human performer. |

#### Example: Training Pipeline for D‑NeRF  

1. **Collect** a video of the scene with known timestamps and calibrated poses.  
2. **Extract** per‑frame latent codes **z_t** (initialized randomly).  
3. **Jointly optimize** the static NeRF parameters **θ_s**, the motion MLP **θ_m**, and all **z_t** to minimize the photometric loss.  
4. **Inference**: given any time **t**, query the motion MLP to obtain the latent code, then render with the static NeRF plus the time‑dependent color/density offsets.

### 4.4 Other Noteworthy Extensions  

| Extension | Focus | Key Paper |
|-----------|-------|-----------|
| **NeRF‑W** (Martin‑Brualla *et al.*, 2021) | Unconstrained “in‑the‑wild” capture with varying illumination. | *NeRF‑W* |
| **Plenoxels** (Yu *et al.*, 2021) | Replace MLP with explicit sparse voxel grids for ultra‑fast training. | *Plenoxels* |
| **Tensor‑fV** (Liu *et al.*, 2022) | Factorized tensor decomposition for memory‑efficient storage. | *Tensor‑fV* |
| **Ref‑NeRF** (Kumar *et al.*, 2022) | Incorporate multi‑view stereo depth priors to accelerate convergence. | *Ref‑NeRF* |

---  

## 5. Practical Recommendations  

| Goal | Recommended Setup | Rationale |
|------|-------------------|-----------|
| **Rapid prototyping (seconds‑to‑minutes)** | **Instant‑NGP** with hash‑grid