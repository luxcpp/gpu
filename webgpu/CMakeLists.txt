cmake_minimum_required(VERSION 3.28)
project(gpu)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/webgpu.cmake")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # export compile_commands.json to use with
                                      # LSP
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_LOCAL_LIBS
       "Use local libraries instead of fetching from the internet" OFF)

# Ensure the build type is set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Choose the type of build: Debug or Release" FORCE)
endif()

option(FASTBUILD "Option to enable fast builds" OFF)
if(FASTBUILD)
    set(CMAKE_BUILD_TYPE None) # Avoid default flags of predefined build types
    set(CMAKE_CXX_FLAGS "-O0")
endif()

option(DEBUG "Option to enable debug flags" OFF)
if(DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS "-O0 -g")
endif()

# Windows x64: WIN32 is defined for both 32 and 64-bit on Windows
# Use CMAKE_SIZEOF_VOID_P == 8 to detect 64-bit
if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWEBGPU_BACKEND_DAWN")
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpu.cmake")

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(
    STATUS
        "Include directories for wgpu: ${CMAKE_CURRENT_SOURCE_DIR}/third_party/headers"
)

# Note: gpu.hpp is a header-only library, gpud is an interface library
# that aggregates the dependencies for consumers
add_library(gpud INTERFACE)
target_link_libraries(gpud INTERFACE wgpu webgpu gpu)
target_include_directories(gpud INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES gpu.hpp DESTINATION include/lux/gpu/webgpu)
