#pragma once
#include <string>
#include <vector>

static const int N_CAMS  = 6;
static const int IMG_H   = 128;
static const int IMG_W   = 352;

std::vector<float> load_and_preprocess_images(const std::vector<std::string>& image_paths);