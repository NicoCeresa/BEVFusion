#pragma once
#include <vector>

struct point_cloud_range {
    float x_min, x_max;
    float y_min, y_max;
    float z_min, z_max;
};

struct voxel_size {
    float x, y, z;
};

struct point {
    float x, y, z, intensity;
};

struct pillar_index {
    int ix, iy;
};

struct pillar_center {
    float x, y;
};

struct cluster_center {
    float x, y, z;
};

struct augmented_point {
    float x, y, z, intensity;
    float x_cluster_offset, y_cluster_offset, z_cluster_offset;
    float x_pillar_offset, y_pillar_offset;
};

struct grid_params {
    float x_min, y_min;
    float x_size, y_size;
};

struct pillar_batch {
    std::vector<std::vector<augmented_point>> features;
    std::vector<pillar_index> unique_indices;
};

grid_params   get_grid_params(voxel_size vs, point_cloud_range range);
std::vector<pillar_index>    discretize_point_clouds(const std::vector<point>& points, voxel_size vs, point_cloud_range range);
std::vector<pillar_center>   get_pillar_centers(const std::vector<pillar_index>& pillar_indices, voxel_size vs, point_cloud_range range);
std::vector<augmented_point> augment_pillars(const std::vector<point>& points, const std::vector<pillar_center>& pillar_centers, const std::vector<cluster_center>& cluster_centers);
pillar_batch                 optimize_pillars(const std::vector<augmented_point>& points, const std::vector<pillar_index>& pillar_indices, int max_points_per_pillar);