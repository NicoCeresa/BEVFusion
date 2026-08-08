"""
Diff the C++ engine's per-stage outputs against the PyTorch reference tensors.

Run scripts/dump_sample.py, then src/cpp/build/infer, then this. Comparing
stage by stage localizes a divergence to the stage that introduced it, rather
than only showing that the final detections disagree.

Judged on *relative* error (max|diff| / max|reference|), not absolute: later
stages carry much larger activations, so a fixed absolute threshold flags
harmless drift in big tensors while missing real errors in small ones.
TensorRT picks different conv kernels and accumulation orders than PyTorch,
which costs a few tenths of a percent even in FP32 — a genuine wiring bug
looks nothing like that (the ImageNet-normalization mismatch this script
originally caught sat ~250% relative).

Run infer with --ref-images to feed PyTorch's exact preprocessed images,
isolating pipeline correctness from stb-vs-PIL JPEG decode differences.
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

REL_TOL = 1e-2  # 1% — well above TensorRT FP32 drift, well below any real bug

# (label, pytorch reference, c++ output)
STAGES = [
    ("cam_input",  "ref_images.bin",         "cpp_cam_input.bin"),
    ("cam_encode", "ref_cam_encode_raw.bin", "cpp_cam_encode_raw.bin"),
    ("voxel_pool", "ref_voxel_pooled.bin",   "cpp_voxel_pooled.bin"),
    ("camera_bev", "ref_camera_bev.bin",     "cpp_camera_bev.bin"),
    ("lidar_bev",  "ref_lidar_bev.bin",      "cpp_lidar_bev.bin"),
    ("fused_bev",  "ref_fused_bev.bin",      "cpp_fused_bev.bin"),
    ("pred_cls",   "ref_pred_cls.bin",       "cpp_pred_cls.bin"),
    ("pred_reg",   "ref_pred_reg.bin",       "cpp_pred_reg.bin"),
]


def main():
    failures = 0
    for label, ref_name, cpp_name in STAGES:
        ref_path, cpp_path = DATA_DIR / ref_name, DATA_DIR / cpp_name
        if not cpp_path.exists():
            print(f"SKIP {label:11s} {cpp_name} not found — run src/cpp/build/infer first")
            failures += 1
            continue

        ref = np.fromfile(ref_path, dtype=np.float32)
        cpp = np.fromfile(cpp_path, dtype=np.float32)
        if ref.shape != cpp.shape:
            print(f"FAIL {label:11s} size mismatch: pytorch {ref.shape}, c++ {cpp.shape}")
            failures += 1
            continue

        diff = np.abs(ref - cpp)
        scale = max(np.abs(ref).max(), 1e-9)
        rel = diff.max() / scale
        ok = rel <= REL_TOL
        if not ok:
            failures += 1

        print(f"{'PASS' if ok else 'FAIL'} {label:11s} n={ref.size:<9d} "
              f"rel={rel:.2e}  max|diff|={diff.max():.3e}  mean|diff|={diff.mean():.3e}")

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} STAGE(S) FAILED'} "
          f"(relative tolerance {REL_TOL:.0e})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
