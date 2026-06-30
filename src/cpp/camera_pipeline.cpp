#include "camera_pipeline.h"
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

std::vector<float> load_and_preprocess_images(const std::vector<std::string>& image_paths) {
    if (image_paths.size() != N_CAMS)
        throw std::runtime_error("Expected 6 camera images");

    std::vector<float> out(N_CAMS * 3 * IMG_H * IMG_W);

    for (int cam = 0; cam < N_CAMS; cam++) {
        int src_w, src_h, channels;
        unsigned char* img = stbi_load(image_paths[cam].c_str(), &src_w, &src_h, &channels, 3);
        if (!img)
            throw std::runtime_error("Could not load: " + image_paths[cam]);

        std::vector<unsigned char> resized(IMG_H * IMG_W * 3);
        stbir_resize_uint8_linear(img, src_w, src_h, 0,
                                  resized.data(), IMG_W, IMG_H, 0, STBIR_RGB);
        stbi_image_free(img);

        // HWC to CHW
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < IMG_H; h++) {
                for (int w = 0; w < IMG_W; w++) {
                    float pixel = resized[(h * IMG_W + w) * 3 + c] / 255.0f;
                    int idx = cam * (3 * IMG_H * IMG_W) + c * (IMG_H * IMG_W) + h * IMG_W + w;
                    out[idx] = (pixel - MEAN[c]) / STD[c];
                }
            }
        }
    }
    return out;
}