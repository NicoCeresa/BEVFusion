cmake_minimum_required(VERSION 3.18)
project(bevfusion_infer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA
find_package(CUDAToolkit REQUIRED)

# TensorRT — override with -DTENSORRT_ROOT=/path/to/tensorrt if not in default paths
set(TENSORRT_ROOT "/media/nico/Games/TensorRT-11.0.0.114" CACHE PATH "TensorRT installation root")

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT}/include /usr/include
)
find_library(TENSORRT_LIB nvinfer
    HINTS ${TENSORRT_ROOT}/lib /usr/lib/x86_64-linux-gnu
)
find_library(TENSORRT_ONNX_LIB nvonnxparser
    HINTS ${TENSORRT_ROOT}/lib /usr/lib/x86_64-linux-gnu
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
    message(FATAL_ERROR "TensorRT not found. Set -DTENSORRT_ROOT=/path/to/tensorrt")
endif()

add_executable(infer
    infer.cpp
    pillarize.cpp
)

target_include_directories(infer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(infer PRIVATE
    ${TENSORRT_LIB}
    ${TENSORRT_ONNX_LIB}
    CUDA::cudart
)
