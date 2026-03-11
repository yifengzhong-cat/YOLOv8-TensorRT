# =============================================================================
# YOLOv8-TensorRT API Server Dockerfile
#
# Base image: NVIDIA TensorRT 23.10 (TensorRT 8.6, CUDA 12.2, Ubuntu 22.04)
# Provides both C++ inference API and Python model conversion tools.
# =============================================================================

FROM nvcr.io/nvidia/tensorrt:23.10-py3

LABEL maintainer="YOLOv8-TensorRT"
LABEL description="YOLOv8 TensorRT inference API server with CUDA and DeepStream support"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# ---- System Dependencies ----------------------------------------------------
# Install OpenCV, FFmpeg, cmake, build tools, and header-only C++ libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        # Build essentials
        build-essential \
        cmake \
        pkg-config \
        # OpenCV and image/video processing
        libopencv-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        ffmpeg \
        # Header-only libraries for the HTTP API server
        libcpp-httplib-dev \
        nlohmann-json3-dev \
        # Networking and utilities
        curl \
        wget \
        git \
    && rm -rf /var/lib/apt/lists/*

# ---- Copy Project Source -----------------------------------------------------
WORKDIR /workspace
COPY . /workspace/

# ---- Build C++ API Server ----------------------------------------------------
# Build the HTTP API server that wraps YOLOv8 TensorRT inference
RUN cd /workspace/csrc/api \
    && mkdir -p build && cd build \
    && cmake .. \
    && make -j"$(nproc)" \
    && cp yolov8-api /usr/local/bin/yolov8-api

# ---- Python Dependencies for Model Conversion -------------------------------
# Install Python packages needed to export/convert models to TensorRT engines
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        numpy \
        opencv-python \
        onnx \
        onnxsim \
        ultralytics \
        torch \
        torchvision

# ---- Runtime Configuration --------------------------------------------------
# Create directories for models and output data
RUN mkdir -p /workspace/models_dir /workspace/output

EXPOSE 22266

# Default: launch the API server (users must volume-mount an engine file)
ENTRYPOINT ["yolov8-api"]
CMD ["--help"]
