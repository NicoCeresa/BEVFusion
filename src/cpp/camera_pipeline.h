#pragma once
#include <string>
#include <vector>

// Decodes, resizes and normalizes the camera images into the (N, 3, H, W)
// tensor cam_encode expects. Target size is passed in rather than fixed here
// so changing camera.image_size in config.yaml needs no C++ edit.
std::vector<float> load_and_preprocess_images(const std::vector<std::string>& image_paths,
                                              int img_h, int img_w);
