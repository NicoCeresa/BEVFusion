cmake_minimum_required(VERSION 3.18)
project(bevfusion_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(infer infer.cpp pillarize.cpp lidar_pipeline.cpp camera_pipeline.cpp)

target_include_directories(infer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    /mnt/BC42E05D42E01E40/TensorRT-11.0.0.114/include
)

target_link_libraries(infer PRIVATE
    /mnt/BC42E05D42E01E40/TensorRT-11.0.0.114/lib/libnvinfer.so
    /mnt/BC42E05D42E01E40/TensorRT-11.0.0.114/lib/libnvonnxparser.so
    /usr/lib/x86_64-linux-gnu/libcudart.so
)
