#include <stdio.h>
#include <array>
#include <vector>
#include <cmath>

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


std::vector<pillar_index> discretize_point_clouds(std::vector<point> points, voxel_size voxel_size, point_cloud_range range){
    float x_min = range.x_min;
    float y_min = range.y_min;

    float x_size = voxel_size.x;
    float y_size = voxel_size.y;

    std::vector<pillar_index> indices;
    indices.reserve(points.size());
    for (const point& p : points) {
        int ix = (int)std::floor((p.x - x_min) / x_size);
        int iy = (int)std::floor((p.y - y_min) / y_size);
        indices.push_back({ix, iy});
    }
    return indices;
}