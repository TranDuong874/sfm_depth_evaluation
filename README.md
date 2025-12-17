# Evaluating Sparse-View Object Reconstruction: A Comparative Study

This repository implements a modular pipeline to evaluate the quality of 3D object reconstruction from sparse image collections ($N \in \{5, 10, 20\}$ views). It compares deep Structure-from-Motion (SfM) baselines against reconstructions augmented by state-of-the-art Monocular Depth Estimation (MDE) priors.

## 1. Overview

We address the problem of reconstructing high-fidelity 3D objects from sparse, uncalibrated views. Standard SfM often fails to produce dense geometry in textureless regions. Our approach leverages dense geometric priors from foundation models (MASt3R, Murre, Depth Anything V2, Metric3D, UniDepth) to fuse accurate, scale-consistent point clouds.

### Key Features
*   **Orthodox Scale Calibration:** Rigorous sequence-level scale alignment to unify heterogeneous depth priors with SfM scale.
*   **Dense Fusion:** Masked multi-view backprojection strategy that eliminates background noise.
*   **Modular Design:** Plug-and-play support for new SfM or Depth methods.
*   **Benchmarking:** Automated evaluation against CO3D ground truth using Chamfer Distance and F-Score.

---

## 2. Methodology

The pipeline consists of five distinct phases:

### Phase 1: Sparse View Sampling
*   **Input:** CO3D raw video sequence.
*   **Process:** Uniformly samples $N$ frames from the video trajectory to simulate sparse capture. Resizes images to 512px max edge (16-pixel aligned) while preserving aspect ratio.
*   **Output:** RGB images and Ground Truth foreground masks.

### Phase 2: Structure-from-Motion (Pose & Scale Anchor)
*   **Algorithm:** **MASt3R (Matching All Strides 3D Reconstruction)** using Global Point Cloud Optimization.
*   **Goal:** Recover camera poses ($T_{cw}$) and a sparse geometric anchor.
*   **Output:** Camera poses, intrinsics, and sparse depth maps ($D_{sfm}$) in a consistent **Scale A**.

### Phase 3: Monocular Depth Estimation & Alignment
*   **Input:** RGB images and SfM sparse depth anchors.
*   **Depth Priors:**
    *   **Murre:** Sparse-guided diffusion (natively aligned to Scale A).
    *   **Metric3D / UniDepth / DepthAnything:** Monocular inference (aligned via global sequence calibration).
*   **Calibration:** We compute a single global scale factor $s$ per sequence per method:
    $$ s = \text{median}\left(\frac{D_{sfm}}{D_{pred}}\right) $$
    This ensures all depth maps are unified to **Scale A**.

### Phase 4: Dense Reconstruction (Fusion)
*   **Algorithm:** Masked Multi-View Backprojection.
*   **Process:**
    1.  **Masking:** Depth maps are strictly masked by object segmentation *before* reprojection to prevent background artifacts.
    2.  **Backprojection:** $ \mathbf{X}_w = R^T (K^{-1} [u, v, 1]^T \cdot d) + \mathbf{t} $. 
    3.  **Cleaning:** Statistical Outlier Removal (SOR) to prune noise.
*   **Output:** Dense, clean object point clouds.

### Phase 5: Evaluation
*   **Metrics:** Chamfer Distance (CD) and F-Score (@2%, 5%, 10%).
*   **Alignment:** Sim3 Procrustes alignment (Rotation, Translation, Scale) + ICP refinement to register prediction to Ground Truth.

---

## 3. Installation

### Prerequisites
*   Linux
*   Python 3.10+
*   CUDA 11.8+ (for PyTorch)

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/sfm_depth_evaluation.git
    cd sfm_depth_evaluation
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Submodules (Manual):**
    You need to clone the specific model repositories into `dependency/`:
    ```bash
    mkdir -p dependency
    # Example for Depth Anything V2
    git clone https://github.com/DepthAnything/Depth-Anything-V2 dependency/Depth-Anything-V2
    # (Repeat for MASt3R, Murre, Metric3D, UniDepth as needed or ensure they are in python path)
    ```
    *Note: The code assumes `dependency/` contains the repositories. Paths are added dynamically.*

5.  **Prepare Data:**
    Place your CO3D data in `data/co3d`. Structure:
    ```
    data/co3d/
    ├── <category>/
    │   └── <object>/
    │       └── <sequence_id>/
    │           ├── images/
    │           └── ...
    ```

---

## 4. Usage

### Quick Start
To run the full pipeline on a specific sequence with 5 views:

```bash
# 1. Edit run.sh or call directly
./run.sh --n-views "5" --sequence "category/object/sequence_id"
```

### Manual Execution (Step-by-Step)

**Phase 1: Sampling**
```bash
python scripts/phase1_sampling.py --co3d-root data/co3d --output output --n-views 5 --max-sequences 1
```

**Phase 2: SfM**
```bash
python scripts/phase2_sfm.py --input output/phase1_sampled --output output/phase2_sfm --methods mast3r
```

**Phase 3: Depth Estimation**
```bash
python scripts/phase3_depth.py --sampled output/phase1_sampled --sfm output/phase2_sfm --output output/phase3_depth --depth-methods murre depth_anything
```

**Phase 4: Reconstruction**
```bash
python scripts/phase4_reconstruction.py --sampled output/phase1_sampled --sfm output/phase2_sfm --depth output/phase3_depth --output output/phase4_reconstruction --methods mast3r_murre mast3r_depth_anything
```

**Phase 5: Evaluation**
```bash
python scripts/phase5_evaluation.py --reconstructions output/phase4_reconstruction --sampled output/phase1_sampled --co3d data/co3d --output output/phase5_evaluation
```

---

## 5. Directory Structure

```
sfm_depth_evaluation/
├── core/               # Core algorithms (SfM, Depth, Reconstruction)
├── scripts/            # Pipeline phase scripts
├── data/               # Input data
├── output/             # Pipeline intermediates and results
│   ├── phase1_sampled/
│   ├── phase2_sfm/
│   ├── phase3_depth/
│   ├── phase4_reconstruction/
│   └── phase5_evaluation/
└── dependency/         # Third-party model weights/code
```