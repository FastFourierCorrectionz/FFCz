cmake_minimum_required(VERSION 3.18)

# Project name and language
project(FFTPreservation LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures (adjust to your needs)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86")

# Optimization flags
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Debug flags
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# Include paths
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files
set(PROJECTION_SOURCES
    projection_algorithm.cu
)

set(EXAMPLE_SOURCES
    projection_example.cu
)

set(UTIL_SOURCES
    fileIO.cpp
)

# Create static library for projection
add_library(projection_algorithm STATIC ${PROJECTION_SOURCES})
add_library(util STATIC ${UTIL_SOURCES})

set_target_properties(projection_algorithm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Link cuFFT for projection_algorithm
target_link_libraries(projection_algorithm
    CUDA::cufft
)

# Executable
add_executable(fftz ${EXAMPLE_SOURCES})

set_target_properties(fftz PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Add ZSTD manually
find_path(ZSTD_INCLUDE_DIR zstd.h)
find_library(ZSTD_LIBRARY zstd)
if(NOT ZSTD_INCLUDE_DIR OR NOT ZSTD_LIBRARY)
    message(FATAL_ERROR "Could not find libzstd. Please install libzstd-dev.")
endif()

target_include_directories(fftz
    PRIVATE ${ZSTD_INCLUDE_DIR}
)

target_link_libraries(fftz
    PUBLIC
        projection_algorithm
        util
        CUDA::cufft
    PRIVATE
        ${ZSTD_LIBRARY}
)

# Print configuration info
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
