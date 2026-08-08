#include "camera_pipeline.h"
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

std::vector<float> load_and_preprocess_images(const std::vector<std::string>& image_paths,
                                              int IMG_H, int IMG_W) {
    const int N_CAMS = static_cast<int>(image_paths.size());
    std::vector<float> out(static_cast<size_t>(N_CAMS) * 3 * IMG_H * IMG_W);

    for (int cam = 0; cam < N_CAMS; cam++) {
        int src_w, src_h, channels;
        unsigned char* img = stbi_load(image_paths[cam].c_str(), &src_w, &src_h, &channels, 3);
        if (!img)
            throw std::runtime_error("Could not load: " + image_paths[cam]);

        // Catmull-Rom to match PIL's default BICUBIC resample (the same
        // interpolating cubic, a = -0.5) that scripts/dataloader.py uses.
        std::vector<unsigned char> resized(IMG_H * IMG_W * 3);
        stbir_resize(img, src_w, src_h, 0,
                     resized.data(), IMG_W, IMG_H, 0,
                     STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM);
        stbi_image_free(img);

        // HWC to CHW. Scale to [0, 1] only — the dataloader this model was
        // trained with applies no ImageNet mean/std normalization, so
        // normalizing here would feed the network a distribution it never saw.
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < IMG_H; h++) {
                for (int w = 0; w < IMG_W; w++) {
                    float pixel = resized[(h * IMG_W + w) * 3 + c] / 255.0f;
                    int idx = cam * (3 * IMG_H * IMG_W) + c * (IMG_H * IMG_W) + h * IMG_W + w;
                    out[idx] = pixel;
                }
            }
        }
    }
    return out;
}