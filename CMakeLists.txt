# Copyright (c) 2024-2026 Lux Industries Inc.
# SPDX-License-Identifier: BSD-3-Clause-Eco
#
# Lux GPU Core - Lightweight plugin-based GPU acceleration library
#
# This is the CORE library only. It provides:
#   - Stable ABI (backend_plugin.h)
#   - Plugin loader (dlopen/LoadLibrary)
#   - CPU fallback backend (builtin)
#   - Backend-agnostic tests
#
# Backend plugins are built separately:
#   - luxcpp/metal  → libluxgpu_backend_metal.dylib  (macOS arm64)
#   - luxcpp/cuda   → libluxgpu_backend_cuda.so      (Linux/Windows)
#   - luxcpp/webgpu → libluxgpu_backend_webgpu.so    (cross-platform)
#
# Usage:
#   cmake -B build
#   cmake --build build

cmake_minimum_required(VERSION 3.20)

project(lux-gpu-core
    VERSION 0.2.0
    DESCRIPTION "Lightweight GPU acceleration core with plugin architecture"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Shared library settings
set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Set install RPATH to find libraries relative to executable
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
elseif(UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
endif()

# =============================================================================
# Compiler Warnings (enabled for all builds)
# =============================================================================

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wconversion
        -Wsign-conversion
        -Wformat=2
        -Wdouble-promotion
        -Wnull-dereference
        -Wimplicit-fallthrough
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_compile_options(
            -Wduplicated-cond
            -Wduplicated-branches
            -Wlogical-op
        )
    endif()
elseif(MSVC)
    add_compile_options(/W4 /permissive-)
endif()

# =============================================================================
# Options
# =============================================================================

option(LUX_GPU_BUILD_TESTS "Build tests" ON)
option(LUX_GPU_BUILD_BENCHMARKS "Build benchmark harness" OFF)
option(LUX_GPU_CPU_BACKEND "Build CPU fallback backend (recommended)" ON)
option(LUX_GPU_CPU_USE_OPENMP "Use OpenMP for CPU backend parallelization" ON)
option(LUX_GPU_METAL_BACKEND "Build Metal backend plugin (macOS only)" ON)
option(LUX_GPU_CUDA_BACKEND "Build CUDA backend plugin" OFF)
option(LUX_GPU_DAWN_BACKEND "Build Dawn/WebGPU backend plugin" OFF)

# =============================================================================
# Core Library (Static + Shared)
# =============================================================================

# Core sources - just the essentials
set(CORE_SOURCES
    src/gpu_core.cpp
    src/kernel_loader.cpp
    src/zk_ops.cpp
)

# CPU backend (builtin, optional but recommended)
if(LUX_GPU_CPU_BACKEND)
    list(APPEND CORE_SOURCES src/cpu_backend.cpp)
endif()

# Static library (for embedding in Go/Rust)
add_library(luxgpu_core_static STATIC ${CORE_SOURCES})

target_include_directories(luxgpu_core_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(luxgpu_core_static PRIVATE
    LUX_GPU_BUILDING_CORE
    $<$<BOOL:${LUX_GPU_CPU_BACKEND}>:LUX_GPU_CPU_BACKEND>
)

set_target_properties(luxgpu_core_static PROPERTIES
    OUTPUT_NAME "luxgpu"
    POSITION_INDEPENDENT_CODE ON
)

# Shared library (for dynamic linking)
add_library(luxgpu_core SHARED ${CORE_SOURCES})

target_include_directories(luxgpu_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(luxgpu_core PRIVATE
    LUX_GPU_BUILDING_CORE
    $<$<BOOL:${LUX_GPU_CPU_BACKEND}>:LUX_GPU_CPU_BACKEND>
)

# Link dl for plugin loading (Unix)
if(NOT WIN32)
    target_link_libraries(luxgpu_core PRIVATE dl)
    target_link_libraries(luxgpu_core_static INTERFACE dl)
endif()

# OpenMP for CPU backend (optional)
if(LUX_GPU_CPU_BACKEND AND LUX_GPU_CPU_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(luxgpu_core PRIVATE OpenMP::OpenMP_CXX)
        target_link_libraries(luxgpu_core_static PRIVATE OpenMP::OpenMP_CXX)
        message(STATUS "CPU backend using OpenMP for parallelization")
    endif()
endif()

set_target_properties(luxgpu_core PROPERTIES
    OUTPUT_NAME "luxgpu"
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

# =============================================================================
# Metal Backend Plugin (macOS only)
# =============================================================================

if(LUX_GPU_METAL_BACKEND AND APPLE)
    # Verify Metal.framework is available
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)

    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
        enable_language(OBJCXX)

        add_library(luxgpu_backend_metal SHARED src/metal_backend.mm)

        target_include_directories(luxgpu_backend_metal PRIVATE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )

        target_link_libraries(luxgpu_backend_metal PRIVATE
            ${METAL_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
        )

        set_target_properties(luxgpu_backend_metal PROPERTIES
            OUTPUT_NAME "luxgpu_backend_metal"
            CXX_STANDARD 17
            OBJCXX_STANDARD 17
        )

        # Suppress -Wconversion warnings from ObjC bridge casts
        target_compile_options(luxgpu_backend_metal PRIVATE -Wno-sign-conversion)

        # Copy kernel sources next to the built library for runtime loading
        add_custom_command(TARGET luxgpu_backend_metal POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/kernels/keccak256.metal
                $<TARGET_FILE_DIR:luxgpu_backend_metal>/keccak256.metal
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/kernels/secp256k1_recover.metal
                $<TARGET_FILE_DIR:luxgpu_backend_metal>/secp256k1_recover.metal
        )

        install(TARGETS luxgpu_backend_metal
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )

        set(HAS_METAL_BACKEND TRUE)
        message(STATUS "Metal backend: ENABLED")
    else()
        set(HAS_METAL_BACKEND FALSE)
        message(STATUS "Metal backend: DISABLED (Metal.framework not found)")
    endif()
else()
    set(HAS_METAL_BACKEND FALSE)
    if(APPLE)
        message(STATUS "Metal backend: DISABLED (LUX_GPU_METAL_BACKEND=OFF)")
    else()
        message(STATUS "Metal backend: DISABLED (not macOS)")
    endif()
endif()

# =============================================================================
# CUDA Backend Plugin (guarded by LUX_GPU_CUDA_BACKEND)
# =============================================================================

if(LUX_GPU_CUDA_BACKEND)
    find_package(CUDAToolkit QUIET)

    add_library(luxgpu_backend_cuda SHARED src/cuda_backend.cpp)

    target_include_directories(luxgpu_backend_cuda PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )

    if(CUDAToolkit_FOUND)
        target_compile_definitions(luxgpu_backend_cuda PRIVATE LUX_GPU_CUDA)
        target_link_libraries(luxgpu_backend_cuda PRIVATE CUDA::cudart)
        set(HAS_CUDA_BACKEND TRUE)
        message(STATUS "CUDA backend: ENABLED (CUDA ${CUDAToolkit_VERSION})")
    else()
        set(HAS_CUDA_BACKEND TRUE)
        message(STATUS "CUDA backend: ENABLED (stub mode, no CUDA toolkit)")
    endif()

    set_target_properties(luxgpu_backend_cuda PROPERTIES
        OUTPUT_NAME "luxgpu_backend_cuda"
        CXX_STANDARD 17
    )

    install(TARGETS luxgpu_backend_cuda
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
else()
    set(HAS_CUDA_BACKEND FALSE)
    message(STATUS "CUDA backend: DISABLED (LUX_GPU_CUDA_BACKEND=OFF)")
endif()

# =============================================================================
# Dawn/WebGPU Backend Plugin (guarded by LUX_GPU_DAWN_BACKEND)
# =============================================================================

if(LUX_GPU_DAWN_BACKEND)
    add_library(luxgpu_backend_webgpu SHARED src/dawn_backend.cpp)

    target_include_directories(luxgpu_backend_webgpu PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )

    # Check for Dawn or wgpu-native
    find_package(Dawn QUIET)
    if(Dawn_FOUND)
        target_compile_definitions(luxgpu_backend_webgpu PRIVATE LUX_GPU_DAWN USE_DAWN_API)
        target_link_libraries(luxgpu_backend_webgpu PRIVATE dawn::webgpu)
        message(STATUS "Dawn backend: ENABLED (Dawn)")
    else()
        # Build without WebGPU runtime — keccak256 falls back to CPU
        message(STATUS "Dawn backend: ENABLED (stub mode, no Dawn/wgpu)")
    endif()

    set_target_properties(luxgpu_backend_webgpu PROPERTIES
        OUTPUT_NAME "luxgpu_backend_webgpu"
        CXX_STANDARD 17
    )

    # Copy WGSL kernels
    add_custom_command(TARGET luxgpu_backend_webgpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/keccak256.wgsl
            $<TARGET_FILE_DIR:luxgpu_backend_webgpu>/keccak256.wgsl
    )

    install(TARGETS luxgpu_backend_webgpu
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    set(HAS_DAWN_BACKEND TRUE)
else()
    set(HAS_DAWN_BACKEND FALSE)
    message(STATUS "Dawn backend: DISABLED (LUX_GPU_DAWN_BACKEND=OFF)")
endif()

# =============================================================================
# Tests (backend-agnostic)
# =============================================================================

if(LUX_GPU_BUILD_TESTS)
    enable_testing()

    add_executable(test_gpu_core test/test_core.cpp)
    target_link_libraries(test_gpu_core PRIVATE luxgpu_core)
    add_test(NAME test_gpu_core COMMAND test_gpu_core)

    add_executable(test_kernel_loader test/test_kernel_loader.cpp)
    target_link_libraries(test_kernel_loader PRIVATE luxgpu_core)
    add_test(NAME test_kernel_loader COMMAND test_kernel_loader)

    add_executable(test_zk_ops test/test_zk_ops.cpp)
    target_link_libraries(test_zk_ops PRIVATE luxgpu_core)
    add_test(NAME test_zk_ops COMMAND test_zk_ops)

    # Comprehensive coverage test (covers all previously untested APIs)
    add_executable(test_coverage_complete test/test_coverage_complete.cpp)
    target_link_libraries(test_coverage_complete PRIVATE luxgpu_core)
    add_test(NAME test_coverage_complete COMMAND test_coverage_complete)

    # Edge cases and error handling tests
    add_executable(test_edge_cases test/test_edge_cases.cpp)
    target_link_libraries(test_edge_cases PRIVATE luxgpu_core)
    add_test(NAME test_edge_cases COMMAND test_edge_cases)

    # Backend integration test
    add_executable(test_backends test/test_backends.cpp)
    target_link_libraries(test_backends PRIVATE luxgpu_core)
    add_test(NAME test_backends COMMAND test_backends)

    # Link Metal backend into test if available
    if(HAS_METAL_BACKEND)
        add_dependencies(test_backends luxgpu_backend_metal)
    endif()

    # Set library path for tests to find plugins
    set_tests_properties(test_gpu_core test_kernel_loader test_zk_ops
                        test_coverage_complete test_edge_cases test_backends PROPERTIES
        ENVIRONMENT "LUX_GPU_BACKEND_PATH=${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()

# =============================================================================
# Benchmarks (backend-agnostic harness)
# =============================================================================

if(LUX_GPU_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# =============================================================================
# pkg-config
# =============================================================================

include(GNUInstallDirs)

# Set private libs for static linking
if(WIN32)
    set(LUX_GPU_PC_PRIVATE_LIBS "")
else()
    set(LUX_GPU_PC_PRIVATE_LIBS "-ldl")
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/lux-gpu.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/luxgpu.pc
    @ONLY
)

# =============================================================================
# Install
# =============================================================================

install(TARGETS luxgpu_core luxgpu_core_static
    EXPORT luxgpu-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Headers (ABI + public API)
install(DIRECTORY include/lux
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# pkg-config
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/luxgpu.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# CMake exports
install(EXPORT luxgpu-targets
    FILE luxgpu-targets.cmake
    NAMESPACE lux::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/luxgpu
)

# =============================================================================
# CMake Config Export
# =============================================================================

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/luxgpu-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/luxgpu-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/luxgpu-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/luxgpu
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/luxgpu-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/luxgpu-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/luxgpu
)

# =============================================================================
# Summary
# =============================================================================

# Check OpenMP status for summary
if(LUX_GPU_CPU_BACKEND AND LUX_GPU_CPU_USE_OPENMP)
    set(_openmp_status "${OpenMP_CXX_FOUND}")
else()
    set(_openmp_status "OFF")
endif()

message(STATUS "")
message(STATUS "=== Lux GPU Core Configuration ===")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  CPU Backend: ${LUX_GPU_CPU_BACKEND}")
message(STATUS "  OpenMP: ${_openmp_status}")
message(STATUS "  Metal Backend: ${HAS_METAL_BACKEND}")
message(STATUS "  CUDA Backend: ${HAS_CUDA_BACKEND}")
message(STATUS "  Dawn Backend: ${HAS_DAWN_BACKEND}")
message(STATUS "  Tests: ${LUX_GPU_BUILD_TESTS}")
message(STATUS "  Benchmarks: ${LUX_GPU_BUILD_BENCHMARKS}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")

# CPU-mode CUDA kernel tests (no nvcc required, compiles .cu as C++)
add_executable(test_cuda_cpu test/test_cuda_cpu.cpp)
target_include_directories(test_cuda_cpu PRIVATE kernels)
target_compile_features(test_cuda_cpu PRIVATE cxx_std_17)
add_test(NAME cuda_cpu_mode COMMAND test_cuda_cpu)
