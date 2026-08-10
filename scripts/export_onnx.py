"""
Export each BEVFusion sub-module to ONNX for TensorRT compilation.

Pillarization and scatter remain in C++ preprocessing — only the neural
network segments are exported here.

Weights are loaded from a trained checkpoint and split by submodule prefix.
Exporting freshly-constructed modules instead would silently produce engines
full of random weights that run fine and detect nothing.

Usage: python scripts/export_onnx.py [checkpoint.pt]
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

sys.path.insert(0, str(ROOT / "scripts"))
from common import default_checkpoint

with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

dbound = cfg["camera"]["dbound"]
D            = int((dbound[1] - dbound[0]) / dbound[2]) 
C            = 64   
N_CAMS       = 6
IMG_H, IMG_W = cfg["camera"]["image_size"]
BEV_H, BEV_W = 200, 200
MAX_PILLARS  = 10000
MAX_PTS      = 32
NUM_ANCHORS  = 5
NUM_ATTRS    = 8


# Which checkpoint state_dict prefix feeds each exported sub-model.
CHECKPOINT_PREFIXES = {
    "cam_encode":      "camera_encoder.camencode",
    "bev_encode":      "camera_encoder.bevencode",
    "pointnet":        "lidar_encoder.pointnet",
    "pillar_backbone": "lidar_encoder.backbone",
    "bev_encoder":     "bev_encoder",
    "ssd":             "head",
}


def load_submodule_weights(model, state_dict, prefix):
    """Load the `prefix.*` slice of a full-model checkpoint into a standalone
    submodule. strict=True on purpose — a silent mismatch here means exporting
    an engine with random weights, which is exactly the failure being avoided."""
    sub = {k[len(prefix) + 1:]: v for k, v in state_dict.items() if k.startswith(prefix + ".")}
    if not sub:
        raise KeyError(f"No checkpoint keys under prefix '{prefix}'")
    model.load_state_dict(sub, strict=True)
    return model


def export_to_onnx(name, model, example_inputs):
    engines_dir = ROOT / "engines"
    engines_dir.mkdir(exist_ok=True)

    model.eval()
    out_path = str(engines_dir / f"{name}.onnx")
    torch.onnx.export(model, example_inputs, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_checkpoint()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Checkpoint: {ckpt_path.name}")
    state_dict = torch.load(ckpt_path, map_location="cpu")

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

        "ssd":             (SSD(in_channels=256, num_classes=3, num_anchors=NUM_ANCHORS, num_attrs=NUM_ATTRS),
                            (torch.randn(1, 256, BEV_H, BEV_W),)),
    }

    for name, (model, example_inputs) in models.items():
        load_submodule_weights(model, state_dict, CHECKPOINT_PREFIXES[name])
        export_to_onnx(name, model, example_inputs)
