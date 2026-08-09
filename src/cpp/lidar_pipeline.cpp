#include "lidar_pipeline.h"
#include <cmath>
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

std::vector<sweep_pose> load_sweep_poses(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Could not open: " + path);

    size_t bytes = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> raw(bytes / sizeof(float));
    file.read(reinterpret_cast<char*>(raw.data()), bytes);

    // Layout per sweep (14 floats): ego_t(3), ego_q(4), cs_t(3), cs_q(4).
    size_t n = bytes / (14 * sizeof(float));
    std::vector<sweep_pose> poses(n);
    for (size_t i = 0; i < n; i++) {
        const float* p = &raw[i * 14];
        for (int k = 0; k < 3; k++) poses[i].ego_t[k] = p[k];
        for (int k = 0; k < 4; k++) poses[i].ego_q[k] = p[3 + k];
        for (int k = 0; k < 3; k++) poses[i].cs_t[k]  = p[7 + k];
        for (int k = 0; k < 4; k++) poses[i].cs_q[k]  = p[10 + k];
    }
    return poses;
}

namespace {

void quat_to_rotmat(const float q[4], float R[9]) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    R[0] = 1 - 2*(y*y + z*z); R[1] = 2*(x*y - w*z);     R[2] = 2*(x*z + w*y);
    R[3] = 2*(x*y + w*z);     R[4] = 1 - 2*(x*x + z*z); R[5] = 2*(y*z - w*x);
    R[6] = 2*(x*z - w*y);     R[7] = 2*(y*z + w*x);     R[8] = 1 - 2*(x*x + y*y);
}

// 4x4 row-major rigid transform from translation + quaternion, matching
// nuscenes-devkit's geometry_utils.transform_matrix.
void build_transform(const float t[3], const float q[4], bool inverse, float M[16]) {
    float R[9];
    quat_to_rotmat(q, R);

    if (!inverse) {
        M[0]=R[0]; M[1]=R[1]; M[2]=R[2];  M[3]=t[0];
        M[4]=R[3]; M[5]=R[4]; M[6]=R[5];  M[7]=t[1];
        M[8]=R[6]; M[9]=R[7]; M[10]=R[8]; M[11]=t[2];
    } else {
        float Rt[9] = {R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]};
        float neg_t[3] = {-t[0], -t[1], -t[2]};
        float t_inv[3];
        for (int r = 0; r < 3; r++)
            t_inv[r] = Rt[r*3+0]*neg_t[0] + Rt[r*3+1]*neg_t[1] + Rt[r*3+2]*neg_t[2];

        M[0]=Rt[0]; M[1]=Rt[1]; M[2]=Rt[2];  M[3]=t_inv[0];
        M[4]=Rt[3]; M[5]=Rt[4]; M[6]=Rt[5];  M[7]=t_inv[1];
        M[8]=Rt[6]; M[9]=Rt[7]; M[10]=Rt[8]; M[11]=t_inv[2];
    }
    M[12]=0; M[13]=0; M[14]=0; M[15]=1;
}

void mat4_mul(const float A[16], const float B[16], float C[16]) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            float sum = 0;
            for (int k = 0; k < 4; k++) sum += A[r*4+k] * B[k*4+c];
            C[r*4+c] = sum;
        }
}

}  // namespace

std::vector<point> aggregate_multisweep(const std::vector<std::string>& sweep_paths,
                                        const std::vector<sweep_pose>& poses,
                                        float min_distance) {
    if (sweep_paths.size() != poses.size())
        throw std::runtime_error("aggregate_multisweep: sweep_paths/poses size mismatch");
    if (sweep_paths.empty())
        return {};

    // Reference frame is sweep 0 — the keyframe itself (chan == ref_chan in
    // the Python original, since both are LIDAR_TOP).
    float ref_from_car[16], car_from_global[16];
    build_transform(poses[0].cs_t,  poses[0].cs_q,  /*inverse=*/true, ref_from_car);
    build_transform(poses[0].ego_t, poses[0].ego_q, /*inverse=*/true, car_from_global);

    std::vector<point> all_points;
    for (size_t i = 0; i < sweep_paths.size(); i++) {
        std::vector<point> pts = load_bin(sweep_paths[i]);

        float global_from_car[16], car_from_current[16];
        build_transform(poses[i].ego_t, poses[i].ego_q, /*inverse=*/false, global_from_car);
        build_transform(poses[i].cs_t,  poses[i].cs_q,  /*inverse=*/false, car_from_current);

        // trans_matrix = ref_from_car @ car_from_global @ global_from_car @ car_from_current
        float step1[16], step2[16], trans_matrix[16];
        mat4_mul(car_from_global, global_from_car, step1);
        mat4_mul(step1, car_from_current, step2);
        mat4_mul(ref_from_car, step2, trans_matrix);

        all_points.reserve(all_points.size() + pts.size());
        for (const point& p : pts) {
            // remove_close: drop points inside the |x|<r AND |y|<r box, before transforming.
            if (std::abs(p.x) < min_distance && std::abs(p.y) < min_distance)
                continue;

            float x = p.x, y = p.y, z = p.z;
            all_points.push_back({
                trans_matrix[0]*x + trans_matrix[1]*y + trans_matrix[2]*z + trans_matrix[3],
                trans_matrix[4]*x + trans_matrix[5]*y + trans_matrix[6]*z + trans_matrix[7],
                trans_matrix[8]*x + trans_matrix[9]*y + trans_matrix[10]*z + trans_matrix[11],
                p.intensity,
            });
        }
    }
    return all_points;
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

std::vector<float> flatten_pillars(const pillar_batch& batch, int max_points_per_pillar, int max_pillars) {
    const int POINT_DIM = 9;
    int P = static_cast<int>(batch.features.size());
    if (P > max_pillars) P = max_pillars;   // engine input shape is fixed
    std::vector<float> out(static_cast<size_t>(max_pillars) * max_points_per_pillar * POINT_DIM, 0.0f);

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

namespace {

lidar_pillars pillarize_points(const std::vector<point>& points,
                               voxel_size vs, point_cloud_range range,
                               int max_points_per_pillar, int max_pillars) {
    // The Python side filters to the point cloud range before pillarizing;
    // without this, out-of-range points land in out-of-bounds grid cells.
    std::vector<point> in_range;
    in_range.reserve(points.size());
    for (const point& p : points) {
        if (p.x >= range.x_min && p.x < range.x_max &&
            p.y >= range.y_min && p.y < range.y_max &&
            p.z >= range.z_min && p.z < range.z_max)
            in_range.push_back(p);
    }

    std::vector<pillar_index> indices = discretize_point_clouds(in_range, vs, range);
    std::vector<pillar_center> centers = get_pillar_centers(indices, vs, range);
    std::vector<cluster_center> clusters = get_cluster_centers(in_range, indices);
    std::vector<augmented_point> augmented = augment_pillars(in_range, centers, clusters);
    pillar_batch batch = optimize_pillars(augmented, indices, max_points_per_pillar);

    lidar_pillars out;
    out.features    = flatten_pillars(batch, max_points_per_pillar, max_pillars);
    out.indices     = batch.unique_indices;
    out.num_pillars = static_cast<int>(batch.unique_indices.size());
    return out;
}

}  // namespace

lidar_pillars run_lidar_pipeline_multisweep(const std::vector<std::string>& sweep_paths,
                                            const std::vector<sweep_pose>& poses,
                                            voxel_size vs, point_cloud_range range,
                                            int max_points_per_pillar, int max_pillars,
                                            std::vector<point>* aggregated_out) {
    std::vector<point> points = aggregate_multisweep(sweep_paths, poses);
    if (aggregated_out) *aggregated_out = points;
    return pillarize_points(points, vs, range, max_points_per_pillar, max_pillars);
}