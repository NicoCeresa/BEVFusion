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

// Ego pose + calibrated_sensor for one LIDAR_TOP sweep, as dumped by
// scripts/dump_sample.py. Quaternions are (w, x, y, z), matching nuScenes'
// own JSON convention.
struct sweep_pose {
    float ego_t[3];
    float ego_q[4];
    float cs_t[3];
    float cs_q[4];
};

std::vector<point>       load_bin(const std::string& path);
std::vector<sweep_pose>  load_sweep_poses(const std::string& path);
std::vector<cluster_center> get_cluster_centers(const std::vector<point>& points, const std::vector<pillar_index>& pillar_indices);
// Zero-pads (or truncates) to max_pillars so the tensor matches the fixed
// input shape the pointnet engine was exported with.
std::vector<float>       flatten_pillars(const pillar_batch& batch, int max_points_per_pillar, int max_pillars);

// Motion-compensates and concatenates raw sweeps into the reference (sweep 0)
// LIDAR_TOP frame — a C++ port of nuscenes-devkit's
// LidarPointCloud.from_file_multisweep, minus the per-point time channel
// (unused by this model; see dataloader.py's _load_lidar).
std::vector<point>       aggregate_multisweep(const std::vector<std::string>& sweep_paths,
                                              const std::vector<sweep_pose>& poses,
                                              float min_distance = 1.0f);

// aggregated_out, if non-null, receives the motion-compensated cloud fed into
// pillarization — used to validate aggregate_multisweep against the Python
// reference independently of the pillarization step.
lidar_pillars             run_lidar_pipeline_multisweep(const std::vector<std::string>& sweep_paths,
                                                        const std::vector<sweep_pose>& poses,
                                                        voxel_size vs, point_cloud_range range,
                                                        int max_points_per_pillar, int max_pillars,
                                                        std::vector<point>* aggregated_out = nullptr);