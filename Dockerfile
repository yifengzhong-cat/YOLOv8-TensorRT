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
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        libopencv-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        ffmpeg \
        libcpp-httplib-dev \
        nlohmann-json3-dev \
        curl \
        wget \
    && rm -rf /var/lib/apt/lists/*

# ---- Copy Project Source -----------------------------------------------------
WORKDIR /workspace
COPY . /workspace/

# ---- Build C++ API Server ----------------------------------------------------
RUN cd /workspace/csrc/api \
    && mkdir -p build && cd build \
    && cmake .. \
    && make -j"$(nproc)" \
    && cp yolov8-api /usr/local/bin/yolov8-api

# ---- Python Dependencies for Model Conversion -------------------------------
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
RUN mkdir -p /workspace/models_dir /workspace/output \
    && chmod +x /workspace/scripts/entrypoint.sh

EXPOSE 22266

ENTRYPOINT ["/workspace/scripts/entrypoint.sh"]
