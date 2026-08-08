// Validates the C++ geometry/pooling port against PyTorch reference tensors
// dumped by scripts/dump_sample.py. Pure math, no TensorRT — run this before
// trusting the wired pipeline in infer.cpp.

#include "geometry.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int compare(const std::string& name,
            const std::vector<float>& got,
            const std::vector<float>& want,
            float tol) {
    if (got.size() != want.size()) {
        std::printf("FAIL %-16s size mismatch: got %zu, want %zu\n",
                    name.c_str(), got.size(), want.size());
        return 1;
    }

    double max_abs = 0.0, sum_abs = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < got.size(); i++) {
        const double diff = std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
        if (diff > max_abs) { max_abs = diff; worst = i; }
        sum_abs += diff;
    }
    const double mean_abs = sum_abs / static_cast<double>(got.size());
    const bool ok = max_abs <= tol;

    std::printf("%s %-16s n=%-9zu max|diff|=%.3e mean|diff|=%.3e",
                ok ? "PASS" : "FAIL", name.c_str(), got.size(), max_abs, mean_abs);
    if (!ok) std::printf("  worst@%zu got=%.6f want=%.6f", worst, got[worst], want[worst]);
    std::printf("\n");
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string data = (argc > 1) ? argv[1] : "data";
    int failures = 0;

    const std::vector<float> rots       = read_f32(data + "/rots.bin");
    const std::vector<float> trans      = read_f32(data + "/trans.bin");
    const std::vector<float> intrins    = read_f32(data + "/intrins.bin");
    const std::vector<float> post_rots  = read_f32(data + "/post_rots.bin");
    const std::vector<float> post_trans = read_f32(data + "/post_trans.bin");

    const std::vector<float> frustum = create_frustum();
    const std::vector<float> geom =
        get_geometry(frustum, rots, trans, intrins, post_rots, post_trans);
    failures += compare("get_geometry", geom, read_f32(data + "/ref_geometry.bin"), 1e-3f);

    // Feed the engine-layout camera features so pooling is tested on exactly
    // what cam_encode.engine will hand it.
    const std::vector<float> cam_raw = read_f32(data + "/ref_cam_encode_raw.bin");
    const std::vector<float> pooled = voxel_pooling(geom, cam_raw);
    // Accumulation order differs from PyTorch's sorted cumsum, so float
    // rounding diverges slightly on cells with many contributions.
    failures += compare("voxel_pooling", pooled, read_f32(data + "/ref_voxel_pooled.bin"), 1e-2f);

    std::printf("\n%s\n", failures == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return failures == 0 ? 0 : 1;
}
