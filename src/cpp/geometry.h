#pragma once
#include <vector>
#include <string>

// LSS camera→BEV geometry, ported from src/camera/lss.py (get_geometry /
// voxel_pooling). Pure math with no learned weights, so it lives here rather
// than in an ONNX engine.
//
// Nothing here is fixed at compile time. Tensor shapes are read from the
// cam_encode engine's I/O at runtime and grid bounds from a small file the
// Python side writes, so changing camera resolution or BEV extents needs no
// C++ edit — previously IMG_H/IMG_W/FEAT_H/FEAT_W were duplicated constants
// that silently disagreed with the engine when the resolution changed.
struct GeometryConfig {
    int n_cams     = 0;
    int cam_c      = 0;   // feature channels per depth bin
    int depth_bins = 0;
    int img_h      = 0;
    int img_w      = 0;
    int feat_h     = 0;   // img_h / LSS downsample
    int feat_w     = 0;

    float dbound_start = 0.0f;   // first depth bin, in metres
    float dbound_step  = 0.0f;

    float dx[3] = {0, 0, 0};     // BEV voxel size
    float bx[3] = {0, 0, 0};     // BEV grid origin (centre of first cell)
    int   nx[3] = {0, 0, 0};     // BEV grid dimensions

    size_t frustum_points() const {
        return static_cast<size_t>(depth_bins) * feat_h * feat_w;
    }
    size_t bev_cells() const {
        return static_cast<size_t>(cam_c) * nx[0] * nx[1];
    }
};

// Reads the grid bounds written by scripts/dump_grid_config.py.
// Shapes are filled in separately from the engine.
GeometryConfig load_grid_config(const std::string& path);

// Frustum grid in (u, v, depth) pixel space, shape (D, fH, fW, 3).
std::vector<float> create_frustum(const GeometryConfig& g);

// Maps every frustum point into the ego frame. Shape (N, D, fH, fW, 3).
// Calibration args are row-major: rots/intrins/post_rots (N,3,3), trans/post_trans (N,3).
std::vector<float> get_geometry(const GeometryConfig& g,
                                const std::vector<float>& frustum,
                                const std::vector<float>& rots,
                                const std::vector<float>& trans,
                                const std::vector<float>& intrins,
                                const std::vector<float>& post_rots,
                                const std::vector<float>& post_trans);

// Scatter-adds camera features into the BEV grid. cam_feats is the cam_encode
// engine output, shape (N, C, D, fH, fW). Returns (C, nx[0], nx[1]) — the Z
// axis is collapsed into channels, a no-op while nx[2] == 1.
std::vector<float> voxel_pooling(const GeometryConfig& g,
                                 const std::vector<float>& geom,
                                 const std::vector<float>& cam_feats);

// Reads a raw little-endian float32 dump (see scripts/dump_sample.py).
std::vector<float> read_f32(const std::string& path);
