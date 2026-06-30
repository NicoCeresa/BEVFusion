#include "lidar_pipeline.h"
#include <fstream>
#include <stdexcept>
#include <vector>
#include <map>

std::vector<point> load_bin(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Could not open: " + path);

    size_t bytes = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> raw(bytes / sizeof(float));
    file.read(reinterpret_cast<char*>(raw.data()), bytes);

    // nuScenes LiDAR: 5 floats per point (x, y, z, intensity, ring)
    // point struct has 4 fields — drop ring_index
    size_t num_points = bytes / (5 * sizeof(float));
    std::vector<point> points;
    points.reserve(num_points);

    for (size_t i = 0; i < num_points; i++) {
        points.push_back({raw[i*5+0], raw[i*5+1], raw[i*5+2], raw[i*5+3]});
    }
    return points;
}

std::vector<cluster_center> get_cluster_centers(const std::vector<point>& points,
                                                 const std::vector<pillar_index>& pillar_indices) {
    std::map<std::pair<int,int>, std::vector<size_t>> pillar_point_map;
    for (size_t i = 0; i < pillar_indices.size(); i++) {
        auto key = std::make_pair(pillar_indices[i].ix, pillar_indices[i].iy);
        pillar_point_map[key].push_back(i);
    }

    std::vector<cluster_center> centers(points.size());
    for (size_t i = 0; i < pillar_indices.size(); i++) {
        auto key = std::make_pair(pillar_indices[i].ix, pillar_indices[i].iy);
        const auto& members = pillar_point_map[key];

        float sx = 0, sy = 0, sz = 0;
        for (size_t j : members) {
            sx += points[j].x;
            sy += points[j].y;
            sz += points[j].z;
        }
        float n = static_cast<float>(members.size());
        centers[i] = {sx / n, sy / n, sz / n};
    }
    return centers;
}

std::vector<float> flatten_pillars(const pillar_batch& batch, int max_points_per_pillar) {
    const int POINT_DIM = 9;
    int P = static_cast<int>(batch.features.size());
    std::vector<float> out(P * max_points_per_pillar * POINT_DIM, 0.0f);

    for (int p = 0; p < P; p++) {
        const auto& pillar = batch.features[p];
        for (size_t n = 0; n < pillar.size(); n++) {
            const augmented_point& ap = pillar[n];
            int base = (p * max_points_per_pillar + n) * POINT_DIM;
            out[base + 0] = ap.x;
            out[base + 1] = ap.y;
            out[base + 2] = ap.z;
            out[base + 3] = ap.intensity;
            out[base + 4] = ap.x_cluster_offset;
            out[base + 5] = ap.y_cluster_offset;
            out[base + 6] = ap.z_cluster_offset;
            out[base + 7] = ap.x_pillar_offset;
            out[base + 8] = ap.y_pillar_offset;
        }
    }
    return out;
}

std::vector<float> run_lidar_pipeline(const std::string& bin_path,
                                       voxel_size vs,
                                       point_cloud_range range,
                                       int max_points_per_pillar) {
    std::vector<point> points = load_bin(bin_path);
    std::vector<pillar_index> indices = discretize_point_clouds(points, vs, range);
    std::vector<pillar_center> centers = get_pillar_centers(indices, vs, range);
    std::vector<cluster_center> clusters = get_cluster_centers(points, indices);
    std::vector<augmented_point> augmented = augment_pillars(points, centers, clusters);
    pillar_batch batch = optimize_pillars(augmented, indices, max_points_per_pillar);
    return flatten_pillars(batch, max_points_per_pillar);
}