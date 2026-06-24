#include <cmath>
#include <pillarize.h>

grid_params get_grid_params(voxel_size voxel_size, point_cloud_range range) {
    return {range.x_min, range.y_min, voxel_size.x, voxel_size.y};
}

std::vector<pillar_index> discretize_point_clouds(std::vector<point> points, voxel_size voxel_size, point_cloud_range range){
    auto [x_min, y_min, x_size, y_size] = get_grid_params(voxel_size, range);

    std::vector<pillar_index> indices;
    indices.reserve(points.size());
    for (const point& p : points) {
        int ix = (int)std::floor((p.x - x_min) / x_size);
        int iy = (int)std::floor((p.y - y_min) / y_size);
        indices.push_back({ix, iy});
    }
    return indices;
}

std::vector<pillar_center> get_pillar_centers(std::vector<pillar_index> pillar_indices, voxel_size voxel_size, point_cloud_range range){
    auto [x_min, y_min, x_size, y_size] = get_grid_params(voxel_size, range);

    std::vector<pillar_center> centers;
    centers.reserve(pillar_indices.size());
    for (const pillar_index& p : pillar_indices){
        float cx = x_min + (p.ix + 0.5f) * x_size;
        float cy = y_min + (p.iy + 0.5f) * y_size;
        centers.push_back({cx, cy});
    }
    return centers;
}