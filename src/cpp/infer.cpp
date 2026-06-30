#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

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

IBuilder * build_engine(const char* modelFile) {
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