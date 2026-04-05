# Copyright (c) 2024-2026 Lux Industries Inc.
# SPDX-License-Identifier: BSD-3-Clause-Eco
#
# Benchmark Suite for Lux GPU
#
# Build:
#   cmake -B build -DLUX_GPU_BUILD_BENCHMARKS=ON
#   cmake --build build
#
# Run:
#   ./build/benchmarks/bench_all

cmake_minimum_required(VERSION 3.20)

# This is meant to be included from parent CMakeLists.txt
# Can also be built standalone:
if(NOT TARGET luxgpu_core)
    project(lux-gpu-benchmarks VERSION 0.1.0 LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # Find parent library
    find_library(LUXGPU_CORE luxgpu_core HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../build)
    if(NOT LUXGPU_CORE)
        message(FATAL_ERROR "luxgpu_core not found. Build parent project first.")
    endif()

    add_library(luxgpu_core SHARED IMPORTED)
    set_target_properties(luxgpu_core PROPERTIES
        IMPORTED_LOCATION ${LUXGPU_CORE}
    )
    target_include_directories(luxgpu_core INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
endif()

# =============================================================================
# Benchmark Executables
# =============================================================================

# Common header
set(BENCH_COMMON_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/bench_common.hpp)

# Tensor benchmarks
add_executable(bench_tensor
    bench_tensor.cpp
    ${BENCH_COMMON_HEADER}
)
target_include_directories(bench_tensor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(bench_tensor PRIVATE luxgpu_core)

# Crypto benchmarks
add_executable(bench_crypto
    bench_crypto.cpp
    ${BENCH_COMMON_HEADER}
)
target_include_directories(bench_crypto PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(bench_crypto PRIVATE luxgpu_core)

# FHE benchmarks
add_executable(bench_fhe
    bench_fhe.cpp
    ${BENCH_COMMON_HEADER}
)
target_include_directories(bench_fhe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(bench_fhe PRIVATE luxgpu_core)

# Ecrecover benchmark
add_executable(bench_ecrecover
    ${CMAKE_CURRENT_SOURCE_DIR}/../test/bench_ecrecover.cpp
    ${BENCH_COMMON_HEADER}
)
target_include_directories(bench_ecrecover PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(bench_ecrecover PRIVATE luxgpu_core)

# Comprehensive benchmark suite
add_executable(bench_all
    bench_all.cpp
    ${BENCH_COMMON_HEADER}
)
target_include_directories(bench_all PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(bench_all PRIVATE luxgpu_core)

# =============================================================================
# Compiler Flags
# =============================================================================

# Enable optimizations for benchmarks
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
    target_compile_options(bench_tensor PRIVATE -O3 -DNDEBUG)
    target_compile_options(bench_crypto PRIVATE -O3 -DNDEBUG)
    target_compile_options(bench_fhe PRIVATE -O3 -DNDEBUG)
    target_compile_options(bench_ecrecover PRIVATE -O3 -DNDEBUG)
    target_compile_options(bench_all PRIVATE -O3 -DNDEBUG)
endif()

# Enable warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(BENCH_WARNINGS -Wall -Wextra -Wpedantic)
    target_compile_options(bench_tensor PRIVATE ${BENCH_WARNINGS})
    target_compile_options(bench_crypto PRIVATE ${BENCH_WARNINGS})
    target_compile_options(bench_fhe PRIVATE ${BENCH_WARNINGS})
    target_compile_options(bench_ecrecover PRIVATE ${BENCH_WARNINGS})
    target_compile_options(bench_all PRIVATE ${BENCH_WARNINGS})
endif()

# =============================================================================
# OpenMP (optional, for CPU reference)
# =============================================================================

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_tensor PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(bench_crypto PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(bench_fhe PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(bench_ecrecover PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(bench_all PRIVATE OpenMP::OpenMP_CXX)
endif()

# =============================================================================
# Custom Targets
# =============================================================================

# Run all benchmarks
add_custom_target(run_benchmarks
    COMMAND $<TARGET_FILE:bench_all>
    DEPENDS bench_all
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running comprehensive benchmark suite..."
)

# Run individual benchmark suites
add_custom_target(run_tensor_benchmarks
    COMMAND $<TARGET_FILE:bench_tensor>
    DEPENDS bench_tensor
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running tensor benchmarks..."
)

add_custom_target(run_crypto_benchmarks
    COMMAND $<TARGET_FILE:bench_crypto>
    DEPENDS bench_crypto
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running crypto benchmarks..."
)

add_custom_target(run_fhe_benchmarks
    COMMAND $<TARGET_FILE:bench_fhe>
    DEPENDS bench_fhe
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running FHE benchmarks..."
)

# =============================================================================
# Installation
# =============================================================================

install(TARGETS bench_tensor bench_crypto bench_fhe bench_ecrecover bench_all
    RUNTIME DESTINATION bin/benchmarks
)

install(FILES ${BENCH_COMMON_HEADER}
    DESTINATION include/lux/gpu/benchmarks
)
