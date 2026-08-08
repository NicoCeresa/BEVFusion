#pragma once
#include "pillarize.h"
#include <string>

// Pillar tensor plus the grid cell each pillar came from — the scatter back
// into the dense BEV grid needs the indices, so they're returned alongside.
struct lidar_pillars {
    std::vector<float>        features;     // max_pillars * max_points_per_pillar * 9, zero-padded
    std::vector<pillar_index> indices;      // num_pillars entries
    int                       num_pillars;  // may exceed max_pillars before clamping
};

std::vector<point>       load_bin(const std::string& path);
std::vector<cluster_center> get_cluster_centers(const std::vector<point>& points, const std::vector<pillar_index>& pillar_indices);
// Zero-pads (or truncates) to max_pillars so the tensor matches the fixed
// input shape the pointnet engine was exported with.
std::vector<float>       flatten_pillars(const pillar_batch& batch, int max_points_per_pillar, int max_pillars);
lidar_pillars            run_lidar_pipeline(const std::string& bin_path, voxel_size vs, point_cloud_range range, int max_points_per_pillar, int max_pillars);