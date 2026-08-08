#include "geometry.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {

void mat3_inverse(const float* m, float* out) {
    const float a = m[0], b = m[1], c = m[2];
    const float d = m[3], e = m[4], f = m[5];
    const float g = m[6], h = m[7], i = m[8];

    const float c00 =  (e * i - f * h);
    const float c01 = -(d * i - f * g);
    const float c02 =  (d * h - e * g);

    const float det = a * c00 + b * c01 + c * c02;
    if (det == 0.0f) throw std::runtime_error("singular 3x3 matrix");
    const float inv_det = 1.0f / det;

    out[0] = c00 * inv_det;
    out[1] = -(b * i - c * h) * inv_det;
    out[2] =  (b * f - c * e) * inv_det;
    out[3] = c01 * inv_det;
    out[4] =  (a * i - c * g) * inv_det;
    out[5] = -(a * f - c * d) * inv_det;
    out[6] = c02 * inv_det;
    out[7] = -(a * h - b * g) * inv_det;
    out[8] =  (a * e - b * d) * inv_det;
}

void mat3_mul(const float* a, const float* b, float* out) {
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            out[r * 3 + c] = a[r * 3 + 0] * b[0 * 3 + c]
                           + a[r * 3 + 1] * b[1 * 3 + c]
                           + a[r * 3 + 2] * b[2 * 3 + c];
}

void mat3_vec3(const float* m, const float* v, float* out) {
    out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
    out[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
    out[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
}

}  // namespace

std::vector<float> read_f32(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Could not open: " + path);

    const size_t bytes = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> out(bytes / sizeof(float));
    file.read(reinterpret_cast<char*>(out.data()), bytes);
    return out;
}

GeometryConfig load_grid_config(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error(
        "Could not open " + path + " — run scripts/dump_grid_config.py first");

    std::unordered_map<std::string, float> kv;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string key;
        float value;
        if (ss >> key >> value) kv[key] = value;
    }

    auto need = [&](const std::string& key) {
        auto it = kv.find(key);
        if (it == kv.end()) throw std::runtime_error("missing key '" + key + "' in " + path);
        return it->second;
    };

    GeometryConfig g;
    g.n_cams       = static_cast<int>(need("n_cams"));
    g.cam_c        = static_cast<int>(need("cam_c"));
    g.img_h        = static_cast<int>(need("img_h"));
    g.img_w        = static_cast<int>(need("img_w"));
    g.feat_h       = static_cast<int>(need("feat_h"));
    g.feat_w       = static_cast<int>(need("feat_w"));
    g.depth_bins   = static_cast<int>(need("depth_bins"));
    g.dbound_start = need("dbound_start");
    g.dbound_step  = need("dbound_step");
    const char* axes[3] = {"x", "y", "z"};
    for (int i = 0; i < 3; i++) {
        g.dx[i] = need(std::string(axes[i]) + "_step");
        // Cell centre, matching gen_dx_bx: bx = min + step/2.
        g.bx[i] = need(std::string(axes[i]) + "_min") + g.dx[i] / 2.0f;
        g.nx[i] = static_cast<int>(
            (need(std::string(axes[i]) + "_max") - need(std::string(axes[i]) + "_min")) / g.dx[i]);
    }
    return g;
}

std::vector<float> create_frustum(const GeometryConfig& g) {
    // ds = arange(dbound_start, ..., dbound_step), broadcast over (fH, fW)
    // xs = linspace(0, img_w - 1, fW), ys = linspace(0, img_h - 1, fH)
    std::vector<float> frustum(g.frustum_points() * 3);

    for (int d = 0; d < g.depth_bins; d++) {
        const float depth = g.dbound_start + g.dbound_step * static_cast<float>(d);
        for (int h = 0; h < g.feat_h; h++) {
            const float v = static_cast<float>(g.img_h - 1) * static_cast<float>(h)
                          / static_cast<float>(g.feat_h - 1);
            for (int w = 0; w < g.feat_w; w++) {
                const float u = static_cast<float>(g.img_w - 1) * static_cast<float>(w)
                              / static_cast<float>(g.feat_w - 1);
                const size_t base = ((static_cast<size_t>(d) * g.feat_h + h) * g.feat_w + w) * 3;
                frustum[base + 0] = u;
                frustum[base + 1] = v;
                frustum[base + 2] = depth;
            }
        }
    }
    return frustum;
}

std::vector<float> get_geometry(const GeometryConfig& g,
                                const std::vector<float>& frustum,
                                const std::vector<float>& rots,
                                const std::vector<float>& trans,
                                const std::vector<float>& intrins,
                                const std::vector<float>& post_rots,
                                const std::vector<float>& post_trans) {
    std::vector<float> geom(static_cast<size_t>(g.n_cams) * g.frustum_points() * 3);

    for (int n = 0; n < g.n_cams; n++) {
        float inv_post_rot[9], inv_intrin[9], combine[9];
        mat3_inverse(&post_rots[n * 9], inv_post_rot);
        mat3_inverse(&intrins[n * 9], inv_intrin);
        mat3_mul(&rots[n * 9], inv_intrin, combine);   // rots @ inverse(intrins)

        for (int d = 0; d < g.depth_bins; d++) {
            for (int h = 0; h < g.feat_h; h++) {
                for (int w = 0; w < g.feat_w; w++) {
                    const size_t fbase = ((static_cast<size_t>(d) * g.feat_h + h) * g.feat_w + w) * 3;

                    // undo post-augmentation
                    float p[3] = {frustum[fbase + 0] - post_trans[n * 3 + 0],
                                  frustum[fbase + 1] - post_trans[n * 3 + 1],
                                  frustum[fbase + 2] - post_trans[n * 3 + 2]};
                    float q[3];
                    mat3_vec3(inv_post_rot, p, q);

                    // pixel → camera ray: (u*d, v*d, d)
                    const float ray[3] = {q[0] * q[2], q[1] * q[2], q[2]};

                    // camera → ego
                    float ego[3];
                    mat3_vec3(combine, ray, ego);

                    const size_t gbase = (static_cast<size_t>(n) * g.frustum_points()
                                          + (static_cast<size_t>(d) * g.feat_h + h) * g.feat_w + w) * 3;
                    geom[gbase + 0] = ego[0] + trans[n * 3 + 0];
                    geom[gbase + 1] = ego[1] + trans[n * 3 + 1];
                    geom[gbase + 2] = ego[2] + trans[n * 3 + 2];
                }
            }
        }
    }
    return geom;
}

std::vector<float> voxel_pooling(const GeometryConfig& g,
                                 const std::vector<float>& geom,
                                 const std::vector<float>& cam_feats) {
    std::vector<float> bev(g.bev_cells(), 0.0f);

    for (int n = 0; n < g.n_cams; n++) {
        for (int d = 0; d < g.depth_bins; d++) {
            for (int h = 0; h < g.feat_h; h++) {
                for (int w = 0; w < g.feat_w; w++) {
                    const size_t voxel = (static_cast<size_t>(d) * g.feat_h + h) * g.feat_w + w;
                    const size_t gbase = (static_cast<size_t>(n) * g.frustum_points() + voxel) * 3;

                    // Truncation toward zero, matching PyTorch .long() — not
                    // floor. Differs for negatives, and the original LSS code
                    // relies on this, so keep the plain cast.
                    const int ix = static_cast<int>((geom[gbase + 0] - (g.bx[0] - g.dx[0] / 2.0f)) / g.dx[0]);
                    const int iy = static_cast<int>((geom[gbase + 1] - (g.bx[1] - g.dx[1] / 2.0f)) / g.dx[1]);
                    const int iz = static_cast<int>((geom[gbase + 2] - (g.bx[2] - g.dx[2] / 2.0f)) / g.dx[2]);

                    if (ix < 0 || ix >= g.nx[0] || iy < 0 || iy >= g.nx[1] ||
                        iz < 0 || iz >= g.nx[2])
                        continue;

                    // cam_feats is the engine's (N, C, D, fH, fW) layout.
                    for (int c = 0; c < g.cam_c; c++) {
                        const size_t fidx = (static_cast<size_t>(n) * g.cam_c + c) * g.frustum_points() + voxel;
                        bev[(static_cast<size_t>(c) * g.nx[0] + ix) * g.nx[1] + iy] += cam_feats[fidx];
                    }
                }
            }
        }
    }
    return bev;
}
