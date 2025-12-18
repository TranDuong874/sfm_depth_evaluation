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

Follow these steps to set up the environment and dependencies.

### 3.1. System Requirements
*   **OS:** Linux (Ubuntu 20.04+ recommended)
*   **Python:** 3.10+
*   **CUDA:** 11.8+ (Required for PyTorch and most depth models)
*   **COLMAP:** Required if running the COLMAP baseline.

### 3.2. Install System Dependencies (COLMAP)
If you intend to run the COLMAP baseline, you must install COLMAP on your system.

**Ubuntu:**
```bash
sudo apt-get update
sudo apt-get install colmap
```
*Verify installation:*
```bash
colmap -h
```

### 3.3. Clone Repository
Clone the main repository and prepare the workspace.

```bash
git clone https://github.com/yourusername/sfm_depth_evaluation.git
cd sfm_depth_evaluation
```

### 3.4. Python Environment
Create and activate a virtual environment to manage dependencies cleanly.

```bash
# Create virtual environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3.5. Install Python Dependencies
Install the required packages listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 3.6. Install Model Submodules (Crucial Step)
This pipeline relies on external codebases for state-of-the-art models. You must clone them into the `dependency/` folder.

**1. Create dependency folder:**
```bash
mkdir -p dependency
```

**2. Clone MASt3R & DUSt3R (Required for SfM):**
```bash
# MASt3R
git clone https://github.com/naver/mast3r dependency/mast3r
# DUSt3R (dependency of MASt3R)
git clone https://github.com/naver/dust3r dependency/mast3r/dust3r
```

**3. Clone Depth Estimators (As needed):**

*   **Murre:**
    ```bash
    # Clone Murre repository (Ensure you have access or correct URL)
    git clone https://github.com/your-murre-repo/Murre dependency/Murre
    ```

*   **Depth Anything V2:**
    ```bash
    git clone https://github.com/DepthAnything/Depth-Anything-V2 dependency/Depth-Anything-V2
    ```

*   **Metric3D:**
    ```bash
    git clone https://github.com/YvanYin/Metric3D dependency/Metric3D
    ```

*   **UniDepth:**
    ```bash
    git clone https://github.com/lpiccinelli-eth/UniDepth dependency/UniDepth
    ```

*Note: The pipeline code dynamically adds these paths to `sys.path`, so they must be located exactly as shown above.*

---

## 4. Data Preparation

This pipeline is designed for the **CO3D (Common Objects in 3D)** dataset.

1.  Download the CO3D dataset (or a subset).
2.  Organize it in `data/co3d` following this structure:

```
data/co3d/
├── <category>/              # e.g., apple
│   └── <object_name>/       # e.g., 110_13051_23361
│       └── <sequence_id>/   # e.g., 110-13051-23361
│           ├── images/      # RGB frames
│           ├── masks/       # Foreground masks
│           └── point_cloud.ply (Ground Truth)
```

---

## 5. Usage

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

## 6. Directory Structure

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