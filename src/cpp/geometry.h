#pragma once
#include <vector>
#include <string>

// LSS camera→BEV geometry, ported from src/camera/lss.py (get_geometry /
// voxel_pooling). These stages are pure math with no learned weights, so they
// live here rather than in an ONNX engine.

static const int N_CAMS_G   = 6;
static const int CAM_C      = 64;   // feature channels per depth bin
static const int DEPTH_BINS = 41;   // dbound [4, 45) step 1
static const int FEAT_H     = 8;    // 128 / downsample(16)
static const int FEAT_W     = 22;   // 352 / downsample(16)

static const int BEV_X = 200;       // xbound [-50, 50) step 0.5
static const int BEV_Y = 200;
static const int BEV_Z = 1;         // zbound [-10, 10) step 20

// gen_dx_bx(xbound, ybound, zbound) from src/camera/tools.py
static const float DX[3] = {0.5f, 0.5f, 20.0f};
static const float BX[3] = {-49.75f, -49.75f, 0.0f};
static const int   NX[3] = {BEV_X, BEV_Y, BEV_Z};

// Frustum grid in (u, v, depth) pixel space, shape (D, fH, fW, 3).
std::vector<float> create_frustum();

// Maps every frustum point into the ego frame. Shape (N, D, fH, fW, 3).
// Calibration args are row-major: rots/intrins/post_rots (N,3,3), trans/post_trans (N,3).
std::vector<float> get_geometry(const std::vector<float>& frustum,
                                const std::vector<float>& rots,
                                const std::vector<float>& trans,
                                const std::vector<float>& intrins,
                                const std::vector<float>& post_rots,
                                const std::vector<float>& post_trans);

// Scatter-adds camera features into the BEV grid. cam_feats is the cam_encode
// engine output, shape (N, C, D, fH, fW). Returns (C, BEV_X, BEV_Y) — the Z
// axis is collapsed into channels, which is a no-op while BEV_Z == 1.
std::vector<float> voxel_pooling(const std::vector<float>& geom,
                                 const std::vector<float>& cam_feats);

// Reads a raw little-endian float32 dump (see scripts/dump_sample.py).
std::vector<float> read_f32(const std::string& path);
