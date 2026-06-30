#pragma once
#include "pillarize.h"
#include <string>

std::vector<point>       load_bin(const std::string& path);
std::vector<cluster_center> get_cluster_centers(const std::vector<point>& points, const std::vector<pillar_index>& pillar_indices);
std::vector<float>       flatten_pillars(const pillar_batch& batch, int max_points_per_pillar);
std::vector<float>       run_lidar_pipeline(const std::string& bin_path, voxel_size vs, point_cloud_range range, int max_points_per_pillar);