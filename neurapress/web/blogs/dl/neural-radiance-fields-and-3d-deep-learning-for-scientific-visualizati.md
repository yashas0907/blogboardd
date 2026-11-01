# Neural Radiance Fields & 3‑D Deep Learning for Scientific Visualization and Virtual Reality  

*An end‑to‑end tutorial that walks you from raw volumetric data to an interactive, VR‑ready NeRF renderer.*

---

## 1. Introduction  

Scientific domains such as medical imaging, climate modeling, and molecular dynamics generate massive **volumetric datasets** (CT, MRI, CFD fields, electron density maps). Traditional visualization pipelines—isosurface extraction, volume rendering, or slice browsing—often require hand‑tuned transfer functions and struggle to deliver **photorealistic, view‑consistent** results at interactive rates.

**Neural Radiance Fields (NeRF)**, introduced by Mildenhall *et al.* (2020), model a scene as a continuous function that maps a 3‑D coordinate **x** and view direction **d** to color **c** and volume density **σ**. By learning this function from a set of posed images, NeRF can synthesize novel views with high fidelity. Recent extensions—*physics‑informed* NeRFs, hierarchical sampling, and fast inference back‑ends—make the technique suitable for scientific visualization and immersive VR.

This tutorial covers:

1. **NeRF‑based reconstruction of volumetric scientific data** – preprocessing, positional encoding, and training.  
2. **Physics‑informed neural rendering** – embedding governing equations (e.g., Navier‑Stokes, Maxwell) into the loss.  
3. **Real‑time interactive pipelines** – hierarchical sampling, CUDA kernels, and TensorRT acceleration.  
4. **VR integration** – exporting to Unity/Unreal, handling head‑tracked view matrices, and latency budgeting.  
5. **Evaluation & benchmarks** – quantitative metrics, performance numbers, and deployment tips.  

All code snippets are runnable with **PyTorch 2.2** (or later) and **CUDA 12**.  

---

## 2. From Raw Volume to a NeRF Training Set  

### 2.1 Data Preparation  

Scientific volumes are usually stored as 3‑D arrays `V ∈ ℝ^{X×Y×Z}` with scalar fields (e.g., density) or multi‑channel data (e.g., RGB absorption). To train a NeRF we need **RGB images** and **camera poses**. Two common strategies are:

| Strategy | Description | When to Use |
|----------|-------------|--------------|
| **Synthetic rendering** | Render a dense set of perspective images from the volume using a high‑quality volume renderer (e.g., VTK, OSPRay). | When ground‑truth camera poses are known (simulation data). |
| **Empirical acquisition** | Use a calibrated imaging system (CT scanner, light‑field microscope) to capture real images and extract poses via Structure‑from‑Motion (SfM). | When the dataset originates from physical experiments. |

Below is a minimal script that creates a synthetic training set from a scalar field `density`. The script uses **PyTorch3D** for differentiable volume rendering, which ensures that the generated images are fully compatible with the NeRF loss.

```python
import torch
import numpy as np
from pytorch3d.renderer import (
    VolumeRenderer,
    NDCMultiviewPerspectiveCameras,
    VolumeSamples,
    EmissionAbsorptionRaymarcher,
)

def generate_synthetic_nerf_dataset(density, device="cuda", n_views=100, img_res=256):
    """
    Parameters
    ----------
    density : torch.Tensor
        3‑D scalar field of shape (X, Y, Z). Values should be in [0, 1].
    n_views : int
        Number of training images to render.
    img_res : int
        Width/height of each rendered image (square).

    Returns
    -------
    images : torch.Tensor
        Rendered RGB images, shape (n_views, 3, img_res, img_res).
    poses  : torch.Tensor
        Camera extrinsics as 4×4 matrices, shape (n_views, 4, 4).
    """
    # Normalise volume to unit cube
    X, Y, Z = density.shape
    grid = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, X, device=device),
        torch.linspace(-1, 1, Y, device=device),
        torch.linspace(-1, 1, Z, device=device),
        indexing="ij"
    ), dim=-1)  # (X, Y, Z, 3)

    # Convert scalar density to RGB by a simple colormap (e.g., viridis)
    cmap = torch.tensor(plt.cm.viridis(np.linspace(0, 1, 256))[:, :3], device=device)  # (256, 3)
    density_norm = (density - density.min()) / (density.max() - density.min())
    color = cmap[(density_norm * 255).long()]  # (X, Y, Z, 3)

    # Assemble volume as (C, X, Y, Z)
    volume = torch.cat([color.permute(3, 0, 1, 2), density.unsqueeze(0)], dim=0)  # (4, X, Y, Z)

    # Raymarcher that outputs RGB and opacity
    raymarcher = EmissionAbsorptionRaymarcher()

    # Randomly sample camera poses on a sphere
    theta = torch.rand(n_views, device=device) * 2 * np.pi
    phi   = torch.arccos(2 * torch.rand(n_views, device=device) - 1)
    radius = 2.0  # distance from volume centre

    cam_positions = torch.stack([
        radius * torch.sin(phi) * torch.cos(theta),
        radius * torch.sin(phi) * torch.sin(theta),
        radius * torch.cos(phi)
    ], dim=-1)  # (n_views, 3)

    # Look at the origin with up = (0, 0, 1)
    cameras = NDCMultiviewPerspectiveCameras(
        R=torch.eye(3, device=device).unsqueeze(0).repeat(n_views, 1, 1),
        T=-cam_positions,
        device=device,
    )

    # Render
    renderer = VolumeRenderer(
        raymarcher=raymarcher,
        sampler=VolumeSamples(num_samples=256, stratified=True),
    )
    images = renderer(cameras=cameras, volume=volume)  # (n_views, 3, H, W)

    # Extract extrinsics as 4×4 matrices
    poses = torch.eye(4, device=device).unsqueeze(0).repeat(n_views, 1, 1)
    poses[:, :3, :3] = cameras.R
    poses[:, :3, 3] = -cameras.T.squeeze(-1)

    return images, poses
```

> **Tip:** For large scientific volumes, down‑sample the grid to a manageable resolution (e.g., 128³) before rendering. The NeRF will later learn a *continuous* representation, so the final rendering quality is not limited by the initial discretisation.

---

## 3. NeRF Architecture for Scientific Data  

### 3.1 Positional Encoding  

NeRF relies on a high‑frequency **positional encoding** to allow MLPs to represent fine details. The encoding maps a scalar coordinate `x ∈ ℝ` to a 2L‑dimensional vector:

\[
\gamma(x) = \big[ \sin(2^{0}\pi x), \cos(2^{0}\pi x), \dots, \sin(2^{L-1}\pi x), \cos(2^{L-1}\pi x) \big].
\]

For scientific volumes we often increase `L` (e.g., `L=10`) to capture steep gradients (e.g., shock fronts).

```python
class PositionalEncoding(torch.nn.Module):
    def __init__(self, num_frequencies=10, include_input=True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.arange(num_frequencies)

    def forward(self, x):
        """
        x: (..., 3) tensor of 3‑D coordinates in [-1, 1]
        Returns: (..., C) where C = 3 * (2 * num_frequencies) (+3 if include_input)
        """
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * np.pi * x))
            out.append(torch.cos(freq * np.pi * x))
        return torch.cat(out, dim=-1)
```

### 3.2 Coarse‑Fine MLP  

The classic NeRF uses a **coarse** network to propose sample depths, followed by a **fine** network that refines the radiance field. For scientific data we augment the MLP with an extra output channel `σ_phys` that predicts a *physics‑consistent* density (e.g., mass density) which can be regularised against known conservation laws.

```python
class NeRFMLP(torch.nn.Module):
    def __init__(self, D=8, W=256, input_ch=63, input_ch_dir=27, skips=[4]):
        """
        D: depth (number of linear layers)
        W: width (neurons per layer)
        input_ch: dimensionality after positional encoding of (x, y, z)
        input_ch_dir: dimensionality after encoding of view direction
        skips: layers where the input is concatenated again
        """
        super().__init__()
        self.D, self.W = D, W
        self.skips = skips

        # ----- Spatial branch -----
        self.pts_linears = torch.nn.ModuleList(
            [torch.nn.Linear(input_ch, W)] +
            [torch.nn.Linear(W + input_ch, W) if i in skips else torch.nn.Linear(W, W) for i in range(D-1)]
        )

        # Output density σ and a feature vector for color
        self.sigma_linear = torch.nn.Linear(W, 1)          # density
        self.feature_linear = torch.nn.Linear(W, W)       # latent feature for color

        # ----- Direction branch -----
        self.dir_linear = torch.nn.Linear(input_ch_dir + W, W // 2)
        self.rgb_linear = torch.nn.Linear(W // 2, 3)      # RGB

    def forward(self, x, d):
        """
        x: (..., input_ch) encoded 3‑D points
        d: (..., input_ch_dir) encoded view directions
        Returns: rgb (..., 3), sigma (..., 1)
        """
        h = x
        for i, l in enumerate(self.pts_linears):
            h = torch.relu(l(h))
            if i in self.skips:
                h = torch.cat([x, h], -1)

        sigma = torch.relu(self.sigma_linear(h))          # positive density
        feature = self.feature_linear(h)

        h_dir = torch.cat([feature, d], -1)
        h_dir = torch.relu(self.dir_linear(h_dir))
        rgb = torch.sigmoid(self.rgb_linear(h_dir))       # colour in [0,1]

        return rgb, sigma
```

The **coarse** and **fine** networks share the same class; we instantiate two copies with independent parameters.

---

## 4. Rendering a Single Ray – Complete Implementation  

The core of NeRF is the **volume rendering equation** discretised by hierarchical sampling. Below is a fully functional `render_ray` routine that works for both the coarse and fine passes. The function accepts a batch of rays (`origins`, `directions`) and returns the rendered RGB, accumulated opacity, and depth.

```python
def render_ray(
    origins,
    directions,
    network_coarse,
    network_fine,
    embed_fn,
    embed_dir_fn,
    N_samples=64,
    N_importance=128,
    near=0.0,
    far=2.0,
    white_bkgd=False,
):
    """
    Parameters
    ----------
    origins, directions : torch.Tensor
        Shape (B, 3). Ray origins and normalized view directions.
    network_coarse, network_fine : NeRFMLP
        Two independent MLPs.
    embed_fn, embed_dir_fn : callable
        Positional encoders for points and view directions.
    N_samples : int
        Number of stratified samples for the coarse pass.
    N_importance : int
        Number of additional samples for the fine pass.
    near, far : float
        Depth interval.
    white_bkgd : bool
        If True, composite over a white background (useful for medical CT).

    Returns
    -------
    rgb_map : torch.Tensor (B, 3)
        Rendered colour.
    depth_map : torch.Tensor (B,)
        Expected depth.
    acc_map : torch.Tensor (B,)
        Accumulated opacity (∑α_i).
    """
    device = origins.device
    B = origins.shape[0]

    # -------------------------------------------------
    # 1. Coarse strat