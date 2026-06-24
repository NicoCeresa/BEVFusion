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
    float x, y, z, intenstity;
};

struct pillar_index {
    int ix, iy;
};

struct pillar_center {
    float cx, cy;
};

struct grid_params {
    float x_min, y_min;
    float x_size, y_size;
};