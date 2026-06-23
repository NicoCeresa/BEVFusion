"""
Export each BEVFusion sub-module to ONNX for TensorRT compilation.

Pillarization and scatter remain in C++ preprocessing — only the neural
network segments are exported here.
"""
import sys
import torch
import yaml
from pathlib import Path

ROOT = (Path(__file__).parent / "..").resolve()
sys.path.insert(0, str(ROOT / "src"))

from camera.lss import CamEncode, BevEncode
from lidar.pointnet import SimplifiedPointNet
from lidar.backbone import PillarBackbone
from fusion.bev_encoder import BEVEncoder
from fusion.detection_head import SSD

with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

dbound = cfg["camera"]["dbound"]
D            = int((dbound[1] - dbound[0]) / dbound[2]) 
C            = 64   
N_CAMS       = 6
IMG_H, IMG_W = 128, 352
BEV_H, BEV_W = 200, 200
MAX_PILLARS  = 10000 
MAX_PTS      = 32     
NUM_ANCHORS  = 5


def export_to_onnx(name, model, example_inputs):
    engines_dir = ROOT / "engines"
    engines_dir.mkdir(exist_ok=True)

    model.eval()
    out_path = str(engines_dir / f"{name}.onnx")
    torch.onnx.export(model, example_inputs, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    models = {
        "cam_encode":      (CamEncode(D=D, C=C, downsample=16),
                            (torch.randn(N_CAMS, 3, IMG_H, IMG_W),)),

        "bev_encode":      (BevEncode(inC=C, outC=1),
                            (torch.randn(1, C, BEV_H, BEV_W),)),

        "pointnet":        (SimplifiedPointNet(input_dim=9, output_dim=C),
                            (torch.randn(MAX_PILLARS, MAX_PTS, 9),)),

        "pillar_backbone": (PillarBackbone(C=C),
                            (torch.randn(1, C, BEV_H * 2, BEV_W * 2),)),  # 400x400 pre-stride

        "bev_encoder":     (BEVEncoder(camera_channels=1, lidar_channels=C * 6, out_channels=256),
                            (torch.randn(1, 1, BEV_H, BEV_W),
                             torch.randn(1, C * 6, BEV_H, BEV_W))),

        "ssd":             (SSD(in_channels=256, num_classes=3, num_anchors=NUM_ANCHORS),
                            (torch.randn(1, 256, BEV_H, BEV_W),)),
    }

    for name, (model, example_inputs) in models.items():
        export_to_onnx(name, model, example_inputs)
