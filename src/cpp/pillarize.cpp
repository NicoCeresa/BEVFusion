#include <map>
#include <cmath>
#include <pillarize.h>

grid_params get_grid_params(voxel_size vs, point_cloud_range range) {
    return {range.x_min, range.y_min, vs.x, vs.y};
}

std::vector<pillar_index> discretize_point_clouds(const std::vector<point>& points,
                                                  voxel_size vs,
                                                  point_cloud_range range) {
    const auto gp = get_grid_params(vs, range);

    std::vector<pillar_index> indices;
    indices.reserve(points.size());
    for (const point& p : points) {
        int ix = static_cast<int>(std::floor((p.x - gp.x_min) / gp.x_size));
        int iy = static_cast<int>(std::floor((p.y - gp.y_min) / gp.y_size));
        indices.push_back({ix, iy});
    }
    return indices;
}

std::vector<pillar_center> get_pillar_centers(const std::vector<pillar_index>& pillar_indices,
                                              voxel_size vs,
                                              point_cloud_range range) {
    const auto gp = get_grid_params(vs, range);

    std::vector<pillar_center> centers;
    centers.reserve(pillar_indices.size());
    for (const pillar_index& p : pillar_indices) {
        float cx = gp.x_min + (p.ix + 0.5f) * gp.x_size;
        float cy = gp.y_min + (p.iy + 0.5f) * gp.y_size;
        centers.push_back({cx, cy});
    }
    return centers;
}

std::vector<augmented_point> augment_pillars(const std::vector<point>& points,
                                             const std::vector<pillar_center>& pillar_centers,
                                             const std::vector<cluster_center>& cluster_centers) {
    std::vector<augmented_point> augmented_points;
    augmented_points.reserve(points.size());
    for (size_t i = 0; i < points.size(); i++) {
        augmented_point ap;
        ap.x         = points[i].x;
        ap.y         = points[i].y;
        ap.z         = points[i].z;
        ap.intensity = points[i].intensity;

        ap.x_cluster_offset = points[i].x - cluster_centers[i].x;
        ap.y_cluster_offset = points[i].y - cluster_centers[i].y;
        ap.z_cluster_offset = points[i].z - cluster_centers[i].z;

        ap.x_pillar_offset = points[i].x - pillar_centers[i].x;
        ap.y_pillar_offset = points[i].y - pillar_centers[i].y;
        augmented_points.push_back(ap);
    }
    return augmented_points;
}

pillar_batch optimize_pillars(const std::vector<augmented_point>& points,
                              const std::vector<pillar_index>& pillar_indices,
                              int max_points_per_pillar) {
    std::map<std::pair<int, int>, int> pillar_map;
    std::vector<int> inverse(pillar_indices.size());

    for (size_t i = 0; i < pillar_indices.size(); i++) {
        auto key = std::make_pair(pillar_indices[i].ix, pillar_indices[i].iy);
        if (pillar_map.find(key) == pillar_map.end()) {
            int slot = static_cast<int>(pillar_map.size());
            pillar_map[key] = slot;
        }
        inverse[i] = pillar_map[key];
    }

    int P = static_cast<int>(pillar_map.size());
    std::vector<std::vector<augmented_point>> pillar_features(P);

    for (size_t i = 0; i < points.size(); i++) {
        int slot = inverse[i];
        if (pillar_features[slot].size() < static_cast<size_t>(max_points_per_pillar)) {
            pillar_features[slot].push_back(points[i]);
        }
    }

    std::vector<pillar_index> unique_indices(P);
    for (const auto& [key, slot] : pillar_map) {
        unique_indices[slot] = {key.first, key.second};
    }
    return {pillar_features, unique_indices};
}