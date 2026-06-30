cmake_minimum_required(VERSION 3.18)
project(bevfusion_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(infer infer.cpp pillarize.cpp lidar_pipeline.cpp camera_pipeline.cpp)

target_include_directories(infer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    /media/nico/Games/TensorRT-11.0.0.114/include
)

target_link_libraries(infer PRIVATE
    /media/nico/Games/TensorRT-11.0.0.114/lib/libnvinfer.so
    /media/nico/Games/TensorRT-11.0.0.114/lib/libnvonnxparser.so
)
