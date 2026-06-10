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

## Project structure

```
bevfusion/
├── data/           # nuScenes dataset, voxelization, augmentation
├── models/         # Camera encoder, LiDAR encoder, fusion head, detection head
├── tracking/       # Kalman filter tracker
├── inference/      # C++ TensorRT engine and inference wrapper
├── eval/           # mAP/NDS metrics and BEV visualization
├── configs/        # Training hyperparameters
├── train.py
└── export_onnx.py
```

## Key papers

- [BEVFusion](https://arxiv.org/abs/2205.13542) — Liang et al., 2022 (primary reference)
- [PointPillars](https://arxiv.org/abs/1812.05784) — Lang et al., 2019
- [Lift, Splat, Shoot](https://arxiv.org/abs/2008.05711) — Philion & Fidler, 2020
- [Swin Transformer](https://arxiv.org/abs/2103.14030) — Liu et al., 2021
- [CenterPoint](https://arxiv.org/abs/2006.11275) — Yin et al., 2021
