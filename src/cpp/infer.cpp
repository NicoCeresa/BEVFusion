#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "lidar_pipeline.h"
#include "camera_pipeline.h"
#include "geometry.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>

using namespace nvinfer1;
using namespace nvonnxparser;

// Must match src/lidar/point_pillars.py's PointPillars config.
static const float VOXEL_X = 0.25f, VOXEL_Y = 0.25f, VOXEL_Z = 4.0f;
static const int   NUM_CLASSES = 3, NUM_ANCHORS = 5, REG_DIM = 9;

static const float SCORE_THRESH = 0.3f;
static const float NMS_IOU_THRESH = 0.3f;
static const char* CLASS_NAMES[NUM_CLASSES] = {"car", "pedestrian", "bicycle"};

// (class_idx, w, l, h, rotation) — mirrors ANCHORS in scripts/train.py.
struct anchor_template { int cls; float w, l, h, rot; };
static const anchor_template ANCHOR_TEMPLATES[NUM_ANCHORS] = {
    {0, 4.73f, 2.08f, 1.77f, 0.0f},
    {0, 4.73f, 2.08f, 1.77f, 1.5708f},
    {1, 0.76f, 0.76f, 1.73f, 0.0f},
    {2, 1.76f, 0.60f, 1.73f, 0.0f},
    {2, 1.76f, 0.60f, 1.73f, 1.5708f},
};
static const float ANCHOR_Z = -1.0f;

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

void save_engine(IHostMemory* serializedModel, const char* outputFile) {
    std::ofstream file(outputFile, std::ios::binary);
    file.write(static_cast<const char*>(serializedModel->data()),
               serializedModel->size());
}

// TensorRT enables TF32 by default on Ampere+, which keeps only 10 mantissa
// bits and costs ~0.2-0.5% relative accuracy vs PyTorch FP32. Fine for
// deployment, but clearing it is how you tell a real wiring bug apart from
// ordinary precision drift when validating.
static bool g_strict_fp32 = false;
static bool g_bench_int8 = false;

void build_engine(const char* modelFile, const char* outputFile) {
    IBuilder* builder = createInferBuilder(logger);
    INetworkDefinition* network = builder->createNetworkV2(0);
    IParser* parser = createParser(*network, logger);

    parser -> parseFromFile(modelFile,
    static_cast<int32_t>(ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, free_bytes * 0.8); //global vram
    config->setMemoryPoolLimit(MemoryPoolType::kTACTIC_SHARED_MEMORY, 48 << 10); //on-chip memory
    if (g_strict_fp32) config->clearFlag(BuilderFlag::kTF32);

    IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
    if (serializedModel == nullptr)
        throw std::runtime_error(std::string("engine build failed for ") + modelFile);

    delete parser;
    delete network;
    delete config;
    delete builder;

    save_engine(serializedModel, outputFile);
    delete serializedModel;
}

void ensure_engine(const char* modelFile, const char* engineFile) {
    if (!std::filesystem::exists(engineFile)) {
        std::cout << "Building " << engineFile << " (first run)" << std::endl;
        build_engine(modelFile, engineFile);
    }
}

ICudaEngine* load_engine(const char* engineFile, IRuntime* runtime) {
    std::ifstream file(engineFile, std::ios::binary);
    std::vector<char> buffer((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    return runtime->deserializeCudaEngine(buffer.data(), buffer.size());
}

// Builds (on first use) and loads one sub-model. Precisions get separate
// engine files so switching modes doesn't silently reuse the other's cache.
ICudaEngine* prepare_engine(const std::string& name, IRuntime* runtime) {
    const std::string onnx   = "engines/" + name + ".onnx";
    const std::string engine = "engines/" + name + (g_strict_fp32 ? ".fp32.engine" : ".engine");
    ensure_engine(onnx.c_str(), engine.c_str());
    return load_engine(engine.c_str(), runtime);
}

// Runs an engine with any number of inputs/outputs — bev_encoder takes two
// inputs and ssd produces two outputs, so a single-in/single-out helper
// doesn't cover the pipeline. Inputs are bound in declaration order.
std::vector<std::vector<float>> infer(ICudaEngine* engine, cudaStream_t stream,
                                      const std::vector<const std::vector<float>*>& inputs) {
    IExecutionContext* context = engine->createExecutionContext();

    std::vector<std::string> input_names, output_names;
    for (int32_t i = 0; i < engine->getNbIOTensors(); i++) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) input_names.push_back(name);
        else                                                        output_names.push_back(name);
    }
    if (input_names.size() != inputs.size())
        throw std::runtime_error("engine expects a different number of inputs");

    std::vector<void*> device_buffers;
    auto cleanup = [&]() {
        for (void* p : device_buffers) cudaFree(p);
        delete context;
    };

    try {
        for (size_t i = 0; i < input_names.size(); i++) {
            const std::vector<float>& host = *inputs[i];
            void* dev = nullptr;
            cudaMalloc(&dev, host.size() * sizeof(float));
            device_buffers.push_back(dev);
            cudaMemcpyAsync(dev, host.data(), host.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
            context->setTensorAddress(input_names[i].c_str(), dev);
        }

        std::vector<std::vector<float>> outputs;
        std::vector<void*> output_devs;
        for (const std::string& name : output_names) {
            const Dims dims = context->getTensorShape(name.c_str());
            size_t count = 1;
            for (int32_t d = 0; d < dims.nbDims; d++) count *= dims.d[d];

            void* dev = nullptr;
            cudaMalloc(&dev, count * sizeof(float));
            device_buffers.push_back(dev);
            output_devs.push_back(dev);
            context->setTensorAddress(name.c_str(), dev);
            outputs.emplace_back(count);
        }

        if (!context->enqueueV3(stream)) throw std::runtime_error("enqueueV3 failed");

        for (size_t i = 0; i < outputs.size(); i++)
            cudaMemcpyAsync(outputs[i].data(), output_devs[i],
                            outputs[i].size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        cleanup();
        return outputs;
    } catch (...) {
        cleanup();
        throw;
    }
}

// Returns an engine tensor's shape, so dimensions come from the engine rather
// than being duplicated as constants that go stale when the model changes.
std::vector<int> tensor_shape(ICudaEngine* engine, int io_index) {
    const Dims d = engine->getTensorShape(engine->getIOTensorName(io_index));
    return std::vector<int>(d.d, d.d + d.nbDims);
}

// Scatters per-pillar features into the dense BEV grid — the C++ counterpart
// of PointPillars._scatter. Output is (C, grid_h, grid_w).
std::vector<float> scatter_pillars(const std::vector<float>& pillar_features,
                                   const std::vector<pillar_index>& indices,
                                   int num_pillars, int channels,
                                   int grid_h, int grid_w, int max_pillars) {
    std::vector<float> bev(static_cast<size_t>(channels) * grid_h * grid_w, 0.0f);
    const int P = std::min(num_pillars, max_pillars);

    for (int p = 0; p < P; p++) {
        const int ix = indices[p].ix, iy = indices[p].iy;
        if (ix < 0 || ix >= grid_w || iy < 0 || iy >= grid_h) continue;
        for (int c = 0; c < channels; c++)
            bev[(static_cast<size_t>(c) * grid_h + iy) * grid_w + ix] =
                pillar_features[static_cast<size_t>(p) * channels + c];
    }
    return bev;
}

struct detection {
    float x, y, z, w, l, h, theta, vx, vy;
    float score;
    int   cls;
};

// Inverse of encode_reg in scripts/train.py; mirrors decode_predictions in
// scripts/test.py.
std::vector<detection> decode(const GeometryConfig& g,
                              const std::vector<float>& pred_cls,
                              const std::vector<float>& pred_reg) {
    std::vector<detection> dets;
    const int bev_x = g.nx[0], bev_y = g.nx[1];
    // Anchor grid spans the same bounds as the BEV grid, so derive the cell
    // centres from it rather than repeating the -50..50 / 0.5 literals.
    const float x_min = g.bx[0] - g.dx[0] / 2.0f;
    const float y_min = g.bx[1] - g.dx[1] / 2.0f;

    for (int h = 0; h < bev_y; h++) {
        for (int w = 0; w < bev_x; w++) {
            const float ax_center = x_min + (static_cast<float>(w) + 0.5f) * g.dx[0];
            const float ay_center = y_min + (static_cast<float>(h) + 0.5f) * g.dx[1];

            for (int a = 0; a < NUM_ANCHORS; a++) {
                const anchor_template& t = ANCHOR_TEMPLATES[a];

                int best_cls = 0;
                float best_score = -1.0f;
                for (int c = 0; c < NUM_CLASSES; c++) {
                    const size_t idx =
                        (static_cast<size_t>(a * NUM_CLASSES + c) * bev_y + h) * bev_x + w;
                    const float score = 1.0f / (1.0f + std::exp(-pred_cls[idx]));
                    if (score > best_score) { best_score = score; best_cls = c; }
                }
                if (best_score <= SCORE_THRESH) continue;

                float r[REG_DIM];
                for (int k = 0; k < REG_DIM; k++)
                    r[k] = pred_reg[(static_cast<size_t>(a * REG_DIM + k) * bev_y + h) * bev_x + w];

                const float diag = std::sqrt(t.w * t.w + t.l * t.l);
                detection d;
                d.x     = ax_center + r[0] * diag;
                d.y     = ay_center + r[1] * diag;
                d.z     = ANCHOR_Z + r[2] * t.h;
                d.w     = t.w * std::exp(r[3]);
                d.l     = t.l * std::exp(r[4]);
                d.h     = t.h * std::exp(r[5]);
                d.theta = t.rot + std::asin(std::max(-1.0f, std::min(1.0f, r[6])));
                d.vx    = r[7];
                d.vy    = r[8];
                d.score = best_score;
                d.cls   = best_cls;
                dets.push_back(d);
            }
        }
    }
    return dets;
}

// Greedy NMS on axis-aligned BEV boxes, matching scripts/test.py's nms().
std::vector<detection> nms(std::vector<detection> dets) {
    std::sort(dets.begin(), dets.end(),
              [](const detection& a, const detection& b) { return a.score > b.score; });

    std::vector<detection> kept;
    std::vector<bool> suppressed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        kept.push_back(dets[i]);

        const float ax1 = dets[i].x - dets[i].w / 2, ax2 = dets[i].x + dets[i].w / 2;
        const float ay1 = dets[i].y - dets[i].l / 2, ay2 = dets[i].y + dets[i].l / 2;

        for (size_t j = i + 1; j < dets.size(); j++) {
            if (suppressed[j]) continue;
            const float bx1 = dets[j].x - dets[j].w / 2, bx2 = dets[j].x + dets[j].w / 2;
            const float by1 = dets[j].y - dets[j].l / 2, by2 = dets[j].y + dets[j].l / 2;

            const float iw = std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1));
            const float ih = std::max(0.0f, std::min(ay2, by2) - std::max(ay1, by1));
            const float inter = iw * ih;
            const float uni = dets[i].w * dets[i].l + dets[j].w * dets[j].l - inter;
            if (inter / (uni + 1e-6f) >= NMS_IOU_THRESH) suppressed[j] = true;
        }
    }
    return kept;
}

void write_f32(const std::string& path, const std::vector<float>& data) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

int main(int argc, char** argv) {
    std::string data_dir = "data";
    bool use_ref_images = false;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        // Skips JPEG decode/resize and feeds PyTorch's exact preprocessed
        // images, so a parity run measures the wiring alone rather than
        // stb-vs-PIL decode differences.
        if (arg == "--ref-images")       use_ref_images = true;
        else if (arg == "--strict-fp32") g_strict_fp32 = true;
        else if (arg == "--bench-int8")  g_bench_int8 = true;
        else                             data_dir = arg;
    }
    IRuntime* runtime = createInferRuntime(logger);
    if (g_strict_fp32) std::cout << "(TF32 disabled)" << std::endl;

    ICudaEngine* cam_encode      = prepare_engine("cam_encode", runtime);
    ICudaEngine* bev_encode      = prepare_engine("bev_encode", runtime);
    ICudaEngine* pointnet        = prepare_engine("pointnet", runtime);
    ICudaEngine* pillar_backbone = prepare_engine("pillar_backbone", runtime);
    ICudaEngine* bev_encoder     = prepare_engine("bev_encoder", runtime);
    ICudaEngine* ssd             = prepare_engine("ssd", runtime);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Grid bounds come from config.yaml via dump_grid_config.py; every tensor
    // dimension is read off the engines themselves, so changing camera
    // resolution or BEV size needs no edit here.
    GeometryConfig geo = load_grid_config(data_dir + "/grid_config.txt");
    {
        // Cross-check against the engine's real I/O. An engine exported at a
        // different resolution than config.yaml would otherwise produce
        // silently misaligned BEV features rather than an error.
        const std::vector<int> in  = tensor_shape(cam_encode, 0);   // (N, 3, H, W)
        const std::vector<int> out = tensor_shape(cam_encode, 1);   // (N, C, D, fH, fW)
        if (in.size() != 4 || out.size() != 5)
            throw std::runtime_error("unexpected cam_encode tensor ranks");

        const std::pair<const char*, std::pair<int, int>> checks[] = {
            {"n_cams",     {geo.n_cams,     in[0]}},
            {"img_h",      {geo.img_h,      in[2]}},
            {"img_w",      {geo.img_w,      in[3]}},
            {"cam_c",      {geo.cam_c,      out[1]}},
            {"depth_bins", {geo.depth_bins, out[2]}},
            {"feat_h",     {geo.feat_h,     out[3]}},
            {"feat_w",     {geo.feat_w,     out[4]}},
        };
        for (const auto& c : checks) {
            if (c.second.first != c.second.second)
                throw std::runtime_error(
                    std::string("grid_config ") + c.first + "=" + std::to_string(c.second.first) +
                    " but cam_encode engine says " + std::to_string(c.second.second) +
                    " — re-run scripts/export_onnx.py and delete engines/*.engine");
        }
    }
    std::printf("geometry: %d cams, %dx%d -> %dx%d feat, %d depth bins, %d ch;"
                " BEV %dx%dx%d\n",
                geo.n_cams, geo.img_h, geo.img_w, geo.feat_h, geo.feat_w,
                geo.depth_bins, geo.cam_c, geo.nx[0], geo.nx[1], geo.nx[2]);

    // Pillar shapes likewise come from the pointnet/backbone engines.
    const std::vector<int> pn_in   = tensor_shape(pointnet, 0);          // (P, pts, 9)
    const std::vector<int> pn_out  = tensor_shape(pointnet, 1);          // (P, C)
    const std::vector<int> pb_in   = tensor_shape(pillar_backbone, 0);   // (1, C, gh, gw)
    const int max_pillars   = pn_in[0];
    const int pts_per_pillar = pn_in[1];
    const int pillar_c      = pn_out[1];
    const int pillar_grid_h = pb_in[2], pillar_grid_w = pb_in[3];

    // ---- camera branch: images -> lifted features -> BEV -> bev_encode ----
    std::vector<std::string> image_paths = {
        data_dir + "/CAM_FRONT.jpg",      data_dir + "/CAM_FRONT_RIGHT.jpg",
        data_dir + "/CAM_FRONT_LEFT.jpg", data_dir + "/CAM_BACK.jpg",
        data_dir + "/CAM_BACK_LEFT.jpg",  data_dir + "/CAM_BACK_RIGHT.jpg"
    };
    std::vector<float> cam_input = use_ref_images
        ? read_f32(data_dir + "/ref_images.bin")
        : load_and_preprocess_images(image_paths, geo.img_h, geo.img_w);
    if (use_ref_images) std::cout << "(using PyTorch reference images)" << std::endl;
    std::vector<float> cam_feats = infer(cam_encode, stream, {&cam_input})[0];
    std::cout << "cam_encode      -> " << cam_feats.size() << " floats" << std::endl;

    const std::vector<float> frustum = create_frustum(geo);
    const std::vector<float> geom = get_geometry(geo, frustum,
                                                 read_f32(data_dir + "/rots.bin"),
                                                 read_f32(data_dir + "/trans.bin"),
                                                 read_f32(data_dir + "/intrins.bin"),
                                                 read_f32(data_dir + "/post_rots.bin"),
                                                 read_f32(data_dir + "/post_trans.bin"));
    std::vector<float> pooled = voxel_pooling(geo, geom, cam_feats);
    std::cout << "voxel_pooling   -> " << pooled.size() << " floats" << std::endl;

    std::vector<float> camera_bev = infer(bev_encode, stream, {&pooled})[0];
    std::cout << "bev_encode      -> " << camera_bev.size() << " floats" << std::endl;

    // ---- lidar branch: sweeps -> motion-compensated cloud -> pillars -> pointnet -> scatter -> backbone ----
    voxel_size vs{VOXEL_X, VOXEL_Y, VOXEL_Z};
    point_cloud_range range{-50.f, 50.f, -50.f, 50.f, -5.f, 3.f};

    std::vector<sweep_pose> sweep_poses = load_sweep_poses(data_dir + "/lidar_sweep_poses.bin");
    std::vector<std::string> sweep_paths;
    for (size_t i = 0; i < sweep_poses.size(); i++)
        sweep_paths.push_back(data_dir + "/lidar_sweeps/sweep_" + std::to_string(i) + ".bin");

    std::vector<point> lidar_aggregated;
    lidar_pillars pillars = run_lidar_pipeline_multisweep(sweep_paths, sweep_poses, vs, range,
                                                          pts_per_pillar, max_pillars, &lidar_aggregated);
    std::cout << "pillarize       -> " << pillars.num_pillars << " pillars";
    if (pillars.num_pillars > max_pillars)
        std::cout << " (truncated to " << max_pillars << ")";
    std::cout << std::endl;

    std::vector<float> pillar_feats = infer(pointnet, stream, {&pillars.features})[0];
    std::vector<float> lidar_scattered = scatter_pillars(pillar_feats, pillars.indices,
                                                         pillars.num_pillars, pillar_c,
                                                         pillar_grid_h, pillar_grid_w,
                                                         max_pillars);
    std::vector<float> lidar_bev = infer(pillar_backbone, stream, {&lidar_scattered})[0];
    std::cout << "pillar_backbone -> " << lidar_bev.size() << " floats" << std::endl;

    // ---- fusion + detection head ----
    std::vector<float> fused = infer(bev_encoder, stream, {&camera_bev, &lidar_bev})[0];
    std::cout << "bev_encoder     -> " << fused.size() << " floats" << std::endl;

    std::vector<std::vector<float>> head_out = infer(ssd, stream, {&fused});
    const std::vector<float>& pred_cls = head_out[0];
    const std::vector<float>& pred_reg = head_out[1];
    std::cout << "ssd             -> cls " << pred_cls.size()
              << ", reg " << pred_reg.size() << " floats" << std::endl;

    // Dumped so scripts/compare_cpp.py can diff every stage against PyTorch.
    const std::string p = data_dir + "/cpp_";
    std::vector<float> lidar_aggregated_flat;
    lidar_aggregated_flat.reserve(lidar_aggregated.size() * 4);
    for (const point& pt : lidar_aggregated) {
        lidar_aggregated_flat.push_back(pt.x);
        lidar_aggregated_flat.push_back(pt.y);
        lidar_aggregated_flat.push_back(pt.z);
        lidar_aggregated_flat.push_back(pt.intensity);
    }
    write_f32(p + "lidar_aggregated.bin", lidar_aggregated_flat);
    write_f32(p + "cam_input.bin", cam_input);
    write_f32(p + "cam_encode_raw.bin", cam_feats);
    write_f32(p + "voxel_pooled.bin", pooled);
    write_f32(p + "camera_bev.bin", camera_bev);
    write_f32(p + "lidar_bev.bin", lidar_bev);
    write_f32(p + "fused_bev.bin", fused);
    write_f32(p + "pred_cls.bin", pred_cls);
    write_f32(p + "pred_reg.bin", pred_reg);

    // Engine-only timing (excludes preprocessing and the CPU-side pooling and
    // scatter), since that's what quantization actually changes.
    {
        const int warmup = 3, iters = 20;
        for (int i = 0; i < warmup; i++) {
            infer(cam_encode, stream, {&cam_input});
            infer(bev_encode, stream, {&pooled});
            infer(pointnet, stream, {&pillars.features});
            infer(pillar_backbone, stream, {&lidar_scattered});
            infer(bev_encoder, stream, {&camera_bev, &lidar_bev});
            infer(ssd, stream, {&fused});
        }
        cudaStreamSynchronize(stream);

        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; i++) {
            infer(cam_encode, stream, {&cam_input});
            infer(bev_encode, stream, {&pooled});
            infer(pointnet, stream, {&pillars.features});
            infer(pillar_backbone, stream, {&lidar_scattered});
            infer(bev_encoder, stream, {&camera_bev, &lidar_bev});
            infer(ssd, stream, {&fused});
        }
        cudaStreamSynchronize(stream);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count() / iters;
        std::printf("\nengine chain: %.1f ms/frame over %d iters (%s)\n",
                    ms, iters, g_strict_fp32 ? "FP32 strict" : "FP32/TF32");
    }

    // Optional: cam_encode INT8 (Q/DQ ONNX, see scripts/quantize_int8.py) vs.
    // this FP32 engine — same input, same timing methodology, plus relative
    // error against the FP32 engine's own output.
    if (g_bench_int8) {
        ICudaEngine* cam_encode_int8 = prepare_engine("cam_encode_int8", runtime);
        std::vector<float> cam_feats_int8 = infer(cam_encode_int8, stream, {&cam_input})[0];

        float max_abs = 0.0f, max_rel = 0.0f, sum_abs = 0.0f;
        for (size_t i = 0; i < cam_feats.size(); i++) {
            float diff = std::abs(cam_feats[i] - cam_feats_int8[i]);
            float rel  = diff / (std::abs(cam_feats[i]) + 1e-6f);
            max_abs = std::max(max_abs, diff);
            max_rel = std::max(max_rel, rel);
            sum_abs += diff;
        }
        std::printf("\ncam_encode INT8 vs FP32: max|diff|=%.4e  max_rel=%.4e  mean|diff|=%.4e\n",
                    max_abs, max_rel, sum_abs / cam_feats.size());

        const int warmup = 3, iters = 20;
        for (int i = 0; i < warmup; i++) infer(cam_encode_int8, stream, {&cam_input});
        cudaStreamSynchronize(stream);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; i++) infer(cam_encode_int8, stream, {&cam_input});
        cudaStreamSynchronize(stream);
        const double int8_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - t0).count() / iters;

        for (int i = 0; i < warmup; i++) infer(cam_encode, stream, {&cam_input});
        cudaStreamSynchronize(stream);
        const auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; i++) infer(cam_encode, stream, {&cam_input});
        cudaStreamSynchronize(stream);
        const double fp32_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - t1).count() / iters;

        std::printf("cam_encode timing: FP32 %.2f ms/frame, INT8 %.2f ms/frame (%.2fx)\n",
                    fp32_ms, int8_ms, fp32_ms / int8_ms);
    }

    std::vector<detection> dets = nms(decode(geo, pred_cls, pred_reg));
    std::cout << "\n" << dets.size() << " detections above score " << SCORE_THRESH << std::endl;
    for (size_t i = 0; i < dets.size() && i < 20; i++) {
        const detection& d = dets[i];
        std::printf("  %-11s score=%.3f  xyz=(%7.2f,%7.2f,%6.2f)  wlh=(%.2f,%.2f,%.2f)  yaw=%6.3f  v=(%.2f,%.2f)\n",
                    CLASS_NAMES[d.cls], d.score, d.x, d.y, d.z, d.w, d.l, d.h, d.theta, d.vx, d.vy);
    }

    cudaStreamDestroy(stream);
    return 0;
}
