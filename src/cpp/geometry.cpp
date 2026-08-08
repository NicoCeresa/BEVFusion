#include "geometry.h"
#include <fstream>
#include <stdexcept>

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

std::vector<float> create_frustum() {
    // ds  = arange(4.0, 45.0, 1.0), broadcast over (fH, fW)
    // xs  = linspace(0, IMG_W - 1, fW), ys = linspace(0, IMG_H - 1, fH)
    std::vector<float> frustum(DEPTH_BINS * FEAT_H * FEAT_W * 3);

    for (int d = 0; d < DEPTH_BINS; d++) {
        const float depth = 4.0f + static_cast<float>(d);
        for (int h = 0; h < FEAT_H; h++) {
            const float v = 127.0f * static_cast<float>(h) / static_cast<float>(FEAT_H - 1);
            for (int w = 0; w < FEAT_W; w++) {
                const float u = 351.0f * static_cast<float>(w) / static_cast<float>(FEAT_W - 1);
                const size_t base = ((static_cast<size_t>(d) * FEAT_H + h) * FEAT_W + w) * 3;
                frustum[base + 0] = u;
                frustum[base + 1] = v;
                frustum[base + 2] = depth;
            }
        }
    }
    return frustum;
}

std::vector<float> get_geometry(const std::vector<float>& frustum,
                                const std::vector<float>& rots,
                                const std::vector<float>& trans,
                                const std::vector<float>& intrins,
                                const std::vector<float>& post_rots,
                                const std::vector<float>& post_trans) {
    std::vector<float> geom(static_cast<size_t>(N_CAMS_G) * DEPTH_BINS * FEAT_H * FEAT_W * 3);

    for (int n = 0; n < N_CAMS_G; n++) {
        float inv_post_rot[9], inv_intrin[9], combine[9];
        mat3_inverse(&post_rots[n * 9], inv_post_rot);
        mat3_inverse(&intrins[n * 9], inv_intrin);
        mat3_mul(&rots[n * 9], inv_intrin, combine);   // rots @ inverse(intrins)

        for (int d = 0; d < DEPTH_BINS; d++) {
            for (int h = 0; h < FEAT_H; h++) {
                for (int w = 0; w < FEAT_W; w++) {
                    const size_t fbase = ((static_cast<size_t>(d) * FEAT_H + h) * FEAT_W + w) * 3;

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

                    const size_t gbase =
                        ((((static_cast<size_t>(n) * DEPTH_BINS + d) * FEAT_H + h) * FEAT_W) + w) * 3;
                    geom[gbase + 0] = ego[0] + trans[n * 3 + 0];
                    geom[gbase + 1] = ego[1] + trans[n * 3 + 1];
                    geom[gbase + 2] = ego[2] + trans[n * 3 + 2];
                }
            }
        }
    }
    return geom;
}

std::vector<float> voxel_pooling(const std::vector<float>& geom,
                                 const std::vector<float>& cam_feats) {
    std::vector<float> bev(static_cast<size_t>(CAM_C) * BEV_X * BEV_Y, 0.0f);

    for (int n = 0; n < N_CAMS_G; n++) {
        for (int d = 0; d < DEPTH_BINS; d++) {
            for (int h = 0; h < FEAT_H; h++) {
                for (int w = 0; w < FEAT_W; w++) {
                    const size_t gbase =
                        ((((static_cast<size_t>(n) * DEPTH_BINS + d) * FEAT_H + h) * FEAT_W) + w) * 3;

                    // Truncation toward zero, matching PyTorch .long() — not
                    // floor. Differs for negatives, and the original LSS code
                    // relies on this, so keep the plain cast.
                    const int ix = static_cast<int>((geom[gbase + 0] - (BX[0] - DX[0] / 2.0f)) / DX[0]);
                    const int iy = static_cast<int>((geom[gbase + 1] - (BX[1] - DX[1] / 2.0f)) / DX[1]);
                    const int iz = static_cast<int>((geom[gbase + 2] - (BX[2] - DX[2] / 2.0f)) / DX[2]);

                    if (ix < 0 || ix >= NX[0] || iy < 0 || iy >= NX[1] || iz < 0 || iz >= NX[2])
                        continue;

                    // cam_feats is the engine's (N, C, D, fH, fW) layout.
                    for (int c = 0; c < CAM_C; c++) {
                        const size_t fidx =
                            ((((static_cast<size_t>(n) * CAM_C + c) * DEPTH_BINS + d) * FEAT_H + h) * FEAT_W) + w;
                        bev[(static_cast<size_t>(c) * BEV_X + ix) * BEV_Y + iy] += cam_feats[fidx];
                    }
                }
            }
        }
    }
    return bev;
}
