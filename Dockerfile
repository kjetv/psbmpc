# PSB-MPC C++ development environment with GPU support
# Can be used standalone or as a devcontainer base
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    g++ \
    cmake \
    make \
    git \
    curl \
    wget \
    unzip \
    libeigen3-dev \
    libboost-all-dev \
    libgeographic-dev \
    && rm -rf /var/lib/apt/lists/*

# Create build and install directories
RUN mkdir -p /workspace/psbmpc_cxx/build

# Set working directory
WORKDIR /workspace

# Copy build script
COPY scripts/build_and_test.sh /workspace/scripts/build_and_test.sh
RUN chmod +x /workspace/scripts/build_and_test.sh

# Patch CMakeLists.txt files to use CUDA17 instead of CUDA20 (CMake 3.22 doesn't know CUDA20 dialect)
RUN find /workspace -name "CMakeLists.txt" -exec sed -i 's/set(CMAKE_CUDA_STANDARD 20)/set(CMAKE_CUDA_STANDARD 17)/g' {} +

# Default command: build and run all tests
# Use bash explicitly to avoid permission issues with volume mounts
CMD ["bash", "/workspace/scripts/build_and_test.sh"]
