# Source: Lang et al., "PointPillars: Fast Encoders for Object Detection from Point Clouds," CVPR 2019.
# Orchestrates pillarization → PointNet encoding → BEV scatter.

import torch
import torch.nn as nn

from .pillarize import discretize_point_cloud, get_pillar_centers, augment_pillars, optimize_pillars
from .simple_pointnet import SimplifiedPointNet


class PointPillars(nn.Module):
    def __init__(self):
        super().__init__()
