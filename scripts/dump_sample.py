"""
Dump one nuScenes sample plus per-stage reference tensors for the C++ engine.

The C++ pipeline needs the raw sensor data and calibration; the reference
tensors let `infer` be validated numerically stage by stage against PyTorch
rather than just checked for "it ran without crashing".

Everything is written as raw little-endian float32 (or int32) so the C++ side
can read it with a plain ifstream, no parser needed.

Usage: python scripts/dump_sample.py [sample_index]
"""
import sys
import shutil
import torch
import numpy as np
from pathlib import Path
from nuscenes.nuscenes import NuScenes

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataloader import NuScenesDataset, CAMERAS, NUM_SWEEPS
from common import cfg, build_model, default_checkpoint


DATA_DIR = ROOT / "data"


def write_f32(path, arr):
    np.asarray(arr, dtype=np.float32).ravel().tofile(path)
    print(f"  {path.name:32s} {tuple(np.shape(arr))}")


def write_i32(path, arr):
    np.asarray(arr, dtype=np.int32).ravel().tofile(path)
    print(f"  {path.name:32s} {tuple(np.shape(arr))}")


def main(sample_idx=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_DIR.mkdir(exist_ok=True)

    nusc = NuScenes(version=cfg['data']['version'], dataroot=cfg['data']['root'], verbose=False)
    dataset = NuScenesDataset(nusc)
    nusc_idx = dataset.sample_indices[sample_idx]
    sample_rec = nusc.sample[nusc_idx]
    sample = dataset[sample_idx]
    print(f"Sample {sample_idx} (nusc index {nusc_idx}, token {sample_rec['token']})")

    print("\nSensor data:")
    # Original-resolution JPEGs — the C++ side does its own resize/normalize.
    for cam in CAMERAS:
        cam_data = nusc.get('sample_data', sample_rec['data'][cam])
        src = Path(nusc.dataroot) / cam_data['filename']
        shutil.copy(src, DATA_DIR / f"{cam}.jpg")
        print(f"  {cam}.jpg")

    # The trained model expects the multi-sweep aggregated cloud, not a single
    # keyframe scan. The actual aggregation now happens in C++ (see
    # lidar_pipeline.cpp's aggregate_multisweep) from the raw sweeps dumped
    # below; this Python-computed copy is kept only as the numeric reference
    # for compare_cpp.py, in nuScenes' 5-float layout (ring=0) so the
    # existing C++ load_bin reads it unchanged.
    pts = sample['lidar_points'].numpy()
    padded = np.zeros((pts.shape[0], 5), dtype=np.float32)
    padded[:, :4] = pts
    padded.tofile(DATA_DIR / "LIDAR_TOP.bin")
    print(f"  LIDAR_TOP.bin                    {pts.shape} (multi-sweep aggregated, Python reference)")

    # Raw per-sweep scans + calibration, so the C++ pipeline can do its own
    # motion-compensated aggregation instead of depending on the dump above.
    sweep_dir = DATA_DIR / "lidar_sweeps"
    shutil.rmtree(sweep_dir, ignore_errors=True)
    sweep_dir.mkdir()

    poses = []
    current_sd = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    for i in range(NUM_SWEEPS):
        shutil.copy(Path(nusc.dataroot) / current_sd['filename'], sweep_dir / f"sweep_{i}.bin")

        ego_pose = nusc.get('ego_pose', current_sd['ego_pose_token'])
        cs       = nusc.get('calibrated_sensor', current_sd['calibrated_sensor_token'])
        poses.append(ego_pose['translation'] + ego_pose['rotation'] +
                     cs['translation'] + cs['rotation'])

        if current_sd['prev'] == '':
            break
        current_sd = nusc.get('sample_data', current_sd['prev'])

    np.asarray(poses, dtype=np.float32).tofile(DATA_DIR / "lidar_sweep_poses.bin")
    print(f"  lidar_sweeps/                     {len(poses)} raw sweeps + poses")

    print("\nCalibration:")
    write_f32(DATA_DIR / "rots.bin", sample['rots'].numpy())
    write_f32(DATA_DIR / "trans.bin", sample['trans'].numpy())
    write_f32(DATA_DIR / "intrins.bin", sample['intrins'].numpy())
    write_f32(DATA_DIR / "post_rots.bin", sample['post_rots'].numpy())
    write_f32(DATA_DIR / "post_trans.bin", sample['post_trans'].numpy())

    # ---- reference tensors, captured stage by stage from the PyTorch model ----
    model = build_model(device, default_checkpoint())

    images     = sample['images'].unsqueeze(0).to(device)
    rots       = sample['rots'].unsqueeze(0).to(device)
    trans      = sample['trans'].unsqueeze(0).to(device)
    intrins    = sample['intrins'].unsqueeze(0).to(device)
    post_rots  = sample['post_rots'].unsqueeze(0).to(device)
    post_trans = sample['post_trans'].unsqueeze(0).to(device)
    points     = sample['lidar_points'].to(device)

    cam = model.camera_encoder
    print("\nReference tensors:")
    with torch.no_grad():
        write_f32(DATA_DIR / "ref_images.bin", sample['images'].numpy())

        geom = cam.get_geometry(rots, trans, intrins, post_rots, post_trans)
        write_f32(DATA_DIR / "ref_geometry.bin", geom.cpu().numpy())

        # Raw CamEncode output, i.e. exactly what the cam_encode engine emits —
        # (N, C, D, fH, fW), before get_cam_feats' reshape/permute.
        cam_raw = cam.camencode(images.view(-1, *images.shape[2:]))
        write_f32(DATA_DIR / "ref_cam_encode_raw.bin", cam_raw.cpu().numpy())

        cam_feats = cam.get_cam_feats(images)
        write_f32(DATA_DIR / "ref_cam_feats.bin", cam_feats.cpu().numpy())

        pooled = cam.voxel_pooling(geom, cam_feats)
        write_f32(DATA_DIR / "ref_voxel_pooled.bin", pooled.cpu().numpy())

        camera_bev = cam.bevencode(pooled)
        write_f32(DATA_DIR / "ref_camera_bev.bin", camera_bev.cpu().numpy())

        lidar_bev = model.lidar_encoder(points)
        write_f32(DATA_DIR / "ref_lidar_bev.bin", lidar_bev.cpu().numpy())

        fused = model.bev_encoder(camera_bev, lidar_bev)
        write_f32(DATA_DIR / "ref_fused_bev.bin", fused.cpu().numpy())

        pred_cls, pred_reg = model.head(fused)
        write_f32(DATA_DIR / "ref_pred_cls.bin", pred_cls.cpu().numpy())
        write_f32(DATA_DIR / "ref_pred_reg.bin", pred_reg.cpu().numpy())

    print(f"\nWrote sample + reference tensors to {DATA_DIR}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
