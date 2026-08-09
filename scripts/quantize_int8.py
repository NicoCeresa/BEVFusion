"""
INT8 post-training quantization for cam_encode via nvidia-modelopt.

TensorRT 11 removed the classic IInt8Calibrator API entirely (BuilderFlag::
kINT8/kFP16 no longer exist — see src/cpp/CMakeLists.txt's TensorRT path and
the C++ engine parity section of the README). Quantization now means baking
QuantizeLinear/DequantizeLinear (Q/DQ) nodes into the ONNX graph before it
reaches TensorRT, which parses them natively — no calibrator class needed on
the C++ side at all. modelopt handles the calibration (collecting real
activation ranges) and inserts those nodes; calibration here runs on real
training samples rather than random tensors, since INT8 range selection is
only as good as the activation statistics it's calibrated on.

cam_encode is the largest submodel by compute (EfficientNet-b0 backbone) and
the standard first target for CNN INT8 — this script proves the toolchain
(calibrate -> Q/DQ ONNX export -> TensorRT build) on it before considering
whether to extend to the other five submodels.

Usage: python scripts/quantize_int8.py [checkpoint.pt]
"""
import sys
import torch
import torch.nn.functional as F
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
from pathlib import Path
from nuscenes.nuscenes import NuScenes

ROOT = (Path(__file__).parent / "..").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from camera.lss import CamEncode
from dataloader import NuScenesDataset
from common import cfg, default_checkpoint, split_scene_names
from export_onnx import load_submodule_weights, CHECKPOINT_PREFIXES


# modelopt only auto-wraps exact nn.Conv2d/nn.Linear instances, not
# subclasses with their own forward() — and EfficientNet-b0's MBConv blocks
# (nearly all of cam_encode's compute) are built entirely from this custom
# class. Without registering it, mtq.quantize silently quantizes only the
# small hand-written LSS head (up1/depthnet) and leaves the backbone at FP32.
class QuantConv2dStaticSamePadding(Conv2dStaticSamePadding):
    def _setup(self):
        self.input_quantizer = TensorQuantizer()
        self.weight_quantizer = TensorQuantizer()

    def forward(self, x):
        x = self.static_padding(x)
        x = self.input_quantizer(x)
        weight = self.weight_quantizer(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


mtq.register(original_cls=Conv2dStaticSamePadding, quantized_cls=QuantConv2dStaticSamePadding)

NUM_CALIB_SAMPLES = 32

dbound = cfg["camera"]["dbound"]
D = int((dbound[1] - dbound[0]) / dbound[2])
C = 64


def main(ckpt_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = ckpt_path or default_checkpoint()
    print(f"Checkpoint: {ckpt_path.name}")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    model = CamEncode(D=D, C=C, downsample=16)
    load_submodule_weights(model, state_dict, CHECKPOINT_PREFIXES["cam_encode"])
    model.eval().to(device)

    nusc = NuScenes(version=cfg['data']['version'], dataroot=cfg['data']['root'], verbose=False)
    dataset = NuScenesDataset(nusc, scene_names=split_scene_names(nusc, 'train'))
    print(f"Calibrating on {NUM_CALIB_SAMPLES} real training samples")

    def forward_loop(m):
        with torch.no_grad():
            for i in range(NUM_CALIB_SAMPLES):
                images = dataset[i]['images'].to(device)  # (N_CAMS, 3, H, W) -- matches cam_encode's own input
                m(images)

    mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)
    mtq.print_quant_summary(model)

    engines_dir = ROOT / "engines"
    engines_dir.mkdir(exist_ok=True)
    example = dataset[0]['images'].to(device)
    out_path = str(engines_dir / "cam_encode_int8.onnx")
    # dynamo=False: torch's new dynamo-based exporter can't trace modelopt's
    # fake-quant modules (their calibrated amax/scale are lifted as dynamic
    # constants, which trips torch.export's fake-tensor checks). The legacy
    # TorchScript-based exporter is what modelopt's Q/DQ insertion targets.
    torch.onnx.export(model, (example,), out_path, dynamo=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
