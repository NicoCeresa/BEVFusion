#pragma once
#include <string>
#include <vector>

static const int N_CAMS  = 6;
static const int IMG_H   = 128;
static const int IMG_W   = 352;

// ImageNet normalization (used by EfficientNet/LSS)
static const float MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float STD[3]  = {0.229f, 0.224f, 0.225f};

std::vector<float> load_and_preprocess_images(const std::vector<std::string>& image_paths);