# Installation Guide {#install}

Build and installation instructions for Lux GPU Core.

## Requirements

### Build Requirements

- **CMake** 3.20 or later
- **C++17** compatible compiler:
  - GCC 9+
  - Clang 10+
  - MSVC 2019+
- **Optional**: OpenMP for CPU backend parallelization

### Platform Support

| Platform | Compiler | Status |
|----------|----------|--------|
| macOS arm64 | Apple Clang 14+ | Supported |
| macOS x86_64 | Apple Clang 14+ | Supported |
| Linux x86_64 | GCC 9+ / Clang 10+ | Supported |
| Linux aarch64 | GCC 9+ / Clang 10+ | Supported |
| Windows x64 | MSVC 2019+ | Supported |

---

## Quick Start

### Clone and Build

```bash
git clone https://github.com/luxfi/gpu.git
cd gpu

# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --parallel

# Run tests
ctest --test-dir build --output-on-failure
```

### Install System-Wide

```bash
# Install to /usr/local (requires sudo on Linux/macOS)
sudo cmake --install build

# Or specify custom prefix
cmake --install build --prefix /opt/lux
```

---

## Build Options

Configure with CMake options:

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLUX_GPU_BUILD_TESTS=ON \
    -DLUX_GPU_BUILD_BENCHMARKS=OFF \
    -DLUX_GPU_CPU_BACKEND=ON \
    -DLUX_GPU_CPU_USE_OPENMP=ON
```

### Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `LUX_GPU_BUILD_TESTS` | ON | Build test suite |
| `LUX_GPU_BUILD_BENCHMARKS` | OFF | Build benchmark harness |
| `LUX_GPU_CPU_BACKEND` | ON | Build CPU fallback backend |
| `LUX_GPU_CPU_USE_OPENMP` | ON | Use OpenMP for CPU parallelization |
| `CMAKE_BUILD_TYPE` | - | Debug, Release, RelWithDebInfo |

---

## Platform-Specific Instructions

### macOS

```bash
# Install Xcode command line tools
xcode-select --install

# Install CMake via Homebrew
brew install cmake

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# For Apple Silicon (M1/M2/M3), Metal backend is recommended
# Install separately from luxcpp/metal
```

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake

# Optional: OpenMP for CPU parallelization
sudo apt-get install -y libomp-dev

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Install
sudo cmake --install build
sudo ldconfig
```

### Linux (Fedora/RHEL)

```bash
# Install dependencies
sudo dnf install -y gcc-c++ cmake

# Optional: OpenMP
sudo dnf install -y libomp-devel

# Build and install
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
sudo cmake --install build
```

### Windows

```powershell
# Using Visual Studio 2019/2022 Developer Command Prompt

# Configure with MSVC
cmake -B build -G "Visual Studio 17 2022" -A x64

# Build Release
cmake --build build --config Release

# Install
cmake --install build --prefix C:\lux
```

---

## Using the Library

### With CMake (Recommended)

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project)

# Find Lux GPU
find_package(luxgpu REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app lux::luxgpu_core)
```

Configure your project:

```bash
cmake -B build -Dluxgpu_DIR=/path/to/luxgpu/lib/cmake/luxgpu
cmake --build build
```

### With pkg-config

```bash
# Compile
gcc -o my_app main.c $(pkg-config --cflags --libs luxgpu)

# Check flags
pkg-config --cflags luxgpu   # -I/usr/local/include
pkg-config --libs luxgpu     # -L/usr/local/lib -lluxgpu -ldl
```

### Direct Linking

```bash
# Linux
gcc -o my_app main.c -I/usr/local/include -L/usr/local/lib -lluxgpu -ldl

# macOS
clang -o my_app main.c -I/usr/local/include -L/usr/local/lib -lluxgpu

# Windows (MSVC)
cl /I C:\lux\include main.c /link C:\lux\lib\luxgpu.lib
```

---

## Backend Plugins

The core library includes only the CPU backend. GPU backends are separate plugins.

### Available Backends

| Backend | Repository | Platform | Dependencies |
|---------|------------|----------|--------------|
| Metal | `luxcpp/metal` | macOS arm64 | Metal.framework, MLX |
| CUDA | `luxcpp/cuda` | Linux, Windows | CUDA Toolkit 12+ |
| WebGPU | `luxcpp/webgpu` | All | Dawn or wgpu |

### Installing Backend Plugins

Backend plugins are shared libraries placed in the plugin search path.

**Plugin naming convention**:
- Linux: `libluxgpu_backend_<name>.so`
- macOS: `libluxgpu_backend_<name>.dylib`
- Windows: `luxgpu_backend_<name>.dll`

**Search paths** (in order):
1. `LUX_GPU_BACKEND_PATH` environment variable
2. System library paths (`/usr/lib/lux-gpu`, `/usr/local/lib/lux-gpu`)
3. Relative to executable

### Example: Installing Metal Backend

```bash
# Clone Metal backend
git clone https://github.com/luxfi/metal.git
cd metal

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Install to system plugin directory
sudo mkdir -p /usr/local/lib/lux-gpu
sudo cp build/libluxgpu_backend_metal.dylib /usr/local/lib/lux-gpu/
```

### Example: Installing CUDA Backend

```bash
# Requires CUDA Toolkit 12+
git clone https://github.com/luxfi/cuda.git
cd cuda

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

sudo cp build/libluxgpu_backend_cuda.so /usr/local/lib/lux-gpu/
```

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LUX_GPU_BACKEND_PATH` | Plugin search path | `/opt/lux/backends` |
| `LUX_BACKEND` | Force specific backend | `cuda`, `metal`, `cpu` |

### Force Backend Selection

```bash
# Use CPU backend even if GPU available
export LUX_BACKEND=cpu
./my_app

# Use specific plugin path
export LUX_GPU_BACKEND_PATH=/opt/custom/backends
./my_app
```

---

## Verification

### Test Installation

```c
// test_install.c
#include <lux/gpu.h>
#include <stdio.h>

int main() {
    printf("Lux GPU version: %d.%d.%d\n",
           LUX_GPU_VERSION_MAJOR,
           LUX_GPU_VERSION_MINOR,
           LUX_GPU_VERSION_PATCH);

    LuxGPU* gpu = lux_gpu_create();
    if (gpu) {
        printf("Backend: %s\n", lux_gpu_backend_name(gpu));
        LuxDeviceInfo info;
        if (lux_gpu_device_info(gpu, &info) == LUX_OK) {
            printf("Device: %s\n", info.name);
        }
        lux_gpu_destroy(gpu);
        return 0;
    }
    return 1;
}
```

Compile and run:

```bash
gcc -o test_install test_install.c $(pkg-config --cflags --libs luxgpu)
./test_install
```

Expected output:

```
Lux GPU version: 0.2.0
Backend: cpu
Device: CPU (SIMD)
```

### Run Test Suite

```bash
cd build
ctest --output-on-failure

# Verbose output
ctest -V

# Run specific test
ctest -R test_gpu_core
```

---

## Troubleshooting

### Library Not Found

```
error: luxgpu not found
```

**Solution**: Ensure library path is set.

```bash
# Linux
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
sudo ldconfig

# macOS
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
```

### Backend Plugin Not Loading

```
Backend 'metal' not available
```

**Solution**: Check plugin path and permissions.

```bash
# Verify plugin exists
ls -la /usr/local/lib/lux-gpu/

# Set explicit path
export LUX_GPU_BACKEND_PATH=/usr/local/lib/lux-gpu
```

### OpenMP Not Found

```
CMake Warning: OpenMP not found
```

**Solution**: Install OpenMP development package.

```bash
# Ubuntu/Debian
sudo apt-get install libomp-dev

# macOS
brew install libomp

# Fedora
sudo dnf install libomp-devel
```

### ABI Version Mismatch

```
Backend ABI version mismatch: expected 2, got 1
```

**Solution**: Rebuild backend plugin with matching core library.

```bash
# Check core ABI version
grep LUX_GPU_BACKEND_ABI_VERSION include/lux/gpu/backend_plugin.h

# Rebuild backend
cd /path/to/backend
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --clean-first
```

---

## Development Build

For development with debug symbols and sanitizers:

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" \
    -DLUX_GPU_BUILD_TESTS=ON

cmake --build build
ctest --test-dir build
```

### Code Coverage

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="--coverage"

cmake --build build
ctest --test-dir build

# Generate coverage report
gcov build/CMakeFiles/luxgpu_core.dir/src/*.cpp.gcno
```
