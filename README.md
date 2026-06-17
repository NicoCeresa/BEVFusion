# BEVFusion

From-scratch implementation of camera–LiDAR 3D object detection in PyTorch, with a C++ TensorRT inference engine. Based on [BEVFusion (MIT CSAIL, 2022)](https://arxiv.org/abs/2205.13542), evaluated on the [nuScenes](https://www.nuscenes.org/) benchmark.

## What it does

Takes synchronized camera images (6×) and a LiDAR point cloud as input and outputs 3D bounding boxes with class labels, velocities, and headings.

```
Camera frames (6x)  ──► Swin-T + BEV Pooling ──┐
                                                  ├──► Fusion Head ──► [class, bbox, velocity]
LiDAR point cloud   ──► PointPillars        ──┘
```

Both modalities are projected into a shared Bird's Eye View (BEV) space before fusion, avoiding the information loss of late-fusion approaches.

## Architecture

| Component | Method |
|-----------|--------|
| Camera encoder | Swin Transformer + FPN + LSS BEV projection |
| LiDAR encoder | PointPillars (voxelization + PointNet MLP) |
| Fusion | Channel-wise BEV concatenation + conv neck |
| Detection head | CenterPoint-style: center offset, box dims, heading, velocity |
| Tracker | Kalman filter for multi-object temporal consistency |

## Target performance (nuScenes val)

| Configuration | mAP | NDS |
|---------------|-----|-----|
| Camera-only | ~35% | ~40% |
| LiDAR-only | ~50% | ~58% |
| Fused (BEVFusion) | ~67% | ~71% |

## Setup

```bash
pip install torch torchvision nuscenes-devkit open3d einops timm
```

Download the nuScenes dataset (registration required at [nuscenes.org](https://www.nuscenes.org/)).

For the C++ TensorRT inference engine (requires CUDA 11.8+, TensorRT 8.6+, CMake 3.20+):

```bash
cd inference/
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Lift, Splat, Shoot (LSS)

LSS is the camera-to-BEV transformation at the core of BEVFusion's camera branch. It solves the problem of converting 2D perspective camera images into the 3D bird's-eye view space where LiDAR features live.

### How it works

**Lift** — For each pixel in the camera feature map, a depth distribution is predicted over D discrete depth bins. Each pixel is then "lifted" into 3D space by placing a feature vector at every depth along its camera ray, weighted by the predicted depth probability. This produces a 3D feature volume of size `(N×H×W×D, C)`.

**Splat** — The 3D feature volume is projected into the ego vehicle's coordinate frame using the camera intrinsics and extrinsics, then pooled into a 2D BEV grid by aggregating all features that fall within each grid cell and collapsing the Z axis.

**Shoot** — In the original LSS paper, a planning head consumes the BEV features to predict future ego trajectories. In BEVFusion this is replaced by the fusion encoder and task-specific heads.

### Where LSS fits in BEVFusion

```
Camera Images (6x)
        │
        ▼
  EfficientNet-b0          ← feature extraction (semantic encoder)
        │
        ▼
   Depth Network           ← predicts depth distribution per pixel
        │
        ▼
   BEV Pooling             ← lifts features to 3D, splats into BEV grid
        │
        ▼
Camera BEV Features  ──────────────────────────────┐
                                                    ▼
LiDAR Point Cloud ──► PointPillars ──► LiDAR BEV ──► Fusion Encoder ──► Task Heads
```

LSS produces the camera BEV feature map. BEVFusion fuses it with the LiDAR BEV map and runs it through shared task heads for detection and segmentation.

### LSS output on nuScenes mini

Camera inputs (6 views) and the resulting BEV feature map from the pretrained LSS model:

![LSS BEV Output](images/lss_output.png)

## Project structure

```
BEVFusion/
├── config.yaml
├── images/             # visualizations
├── scripts/            # runnable scripts
│   ├── read_nuscenes.py
│   └── run_lss.py
└── src/
    ├── backbones/      # LSS model and tools
    └── view_transform/ # camera-to-BEV transformation
```

## Key papers

- [BEVFusion](https://arxiv.org/abs/2205.13542) — Liu et al., 2022 (primary reference)
- [Lift, Splat, Shoot](https://arxiv.org/abs/2008.05711) — Philion & Fidler, 2020
- [PointPillars](https://arxiv.org/abs/1812.05784) — Lang et al., 2019
- [Swin Transformer](https://arxiv.org/abs/2103.14030) — Liu et al., 2021
- [CenterPoint](https://arxiv.org/abs/2006.11275) — Yin et al., 2021
