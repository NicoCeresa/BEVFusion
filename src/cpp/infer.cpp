#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "lidar_pipeline.h"
#include "camera_pipeline.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace nvinfer1;
using namespace nvonnxparser;

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

void build_engine(const char* modelFile) {
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
    
    IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);

    delete parser;
    delete network;
    delete config;
    delete builder;
    
    save_engine(serializedModel, "cam_encode.engine");
    delete serializedModel;
}

ICudaEngine* load_engine(const char* engineFile, IRuntime* runtime) {
    std::ifstream file(engineFile, std::ios::binary);
    std::vector<char> buffer((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    return runtime->deserializeCudaEngine(buffer.data(), buffer.size());
}

std::vector<float> infer(ICudaEngine* engine, cudaStream_t stream, std::vector<float> const& host_input, Dims const& input_dims){

    IExecutionContext *context = engine->createExecutionContext();

    char const* const input_name = engine->getIOTensorName(0);
    char const* const output_name = engine->getIOTensorName(1);

    context->setInputShape(input_name, input_dims);
    Dims const outputDims = context->getTensorShape(output_name);
    size_t outputCount = 1;
    for (int32_t i = 0; i < outputDims.nbDims; ++i)
    {
        outputCount *= outputDims.d[i];
    }
    std::vector<float> host_output(outputCount);

    void* dInput{nullptr};
    void* dOutput{nullptr};

    cudaMalloc(&dInput, host_input.size() * sizeof(float));
    cudaMalloc(&dOutput, host_output.size() * sizeof(float));
    cudaMemcpyAsync(dInput, host_input.data(), host_input.size() * sizeof(float),
        cudaMemcpyHostToDevice, stream);

    context->setTensorAddress(input_name, dInput);
    context->setTensorAddress(output_name, dOutput);
    if (!context->enqueueV3(stream)) {
        cudaFree(dInput);
        cudaFree(dOutput);
        cudaStreamDestroy(stream);
        delete context;
        throw std::runtime_error("enqueueV3 failed");
    }


    cudaMemcpyAsync(host_output.data(), dOutput,
        host_output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(dInput);
    cudaFree(dOutput);
    cudaStreamDestroy(stream);
    delete context;

    return host_output;
}


int main() {
    IRuntime* runtime = createInferRuntime(logger);
    
    ICudaEngine* cam_encode = load_engine("engines/cam_encode.engine", runtime);
    ICudaEngine* pointnet   = load_engine("engines/pointnet.engine", runtime);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // camera branch
    std::vector<std::string> image_paths = {
        "data/CAM_FRONT.jpg", "data/CAM_FRONT_LEFT.jpg", "data/CAM_FRONT_RIGHT.jpg",
        "data/CAM_BACK.jpg",  "data/CAM_BACK_LEFT.jpg",  "data/CAM_BACK_RIGHT.jpg"
    };
    std::vector<float> cam_input = load_and_preprocess_images(image_paths);
    Dims4 cam_dims{6, 3, IMG_H, IMG_W};
    std::vector<float> cam_output = infer(cam_encode, stream, cam_input, cam_dims);

    // lidar branch
    voxel_size vs{0.2f, 0.2f, 0.4f};
    point_cloud_range range{-50.f, 50.f, -50.f, 50.f, -3.f, 5.f};
    std::vector<float> pillar_input = run_lidar_pipeline("data/LIDAR_TOP.bin", vs, range, 32);
    Dims3 pointnet_dims{10000, 32, 9};
    std::vector<float> pointnet_output = infer(pointnet, stream, pillar_input, pointnet_dims);

    cudaStreamDestroy(stream);
}
