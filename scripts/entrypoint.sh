#!/bin/bash
# =============================================================================
# Docker Entrypoint Script for YOLOv8-TensorRT API Server
#
# This script:
# 1. Checks for .pt model files and converts them to .engine if needed
# 2. Starts the C++ API server
# =============================================================================
set -e

MODELS_DIR="${MODELS_DIR:-/workspace/models_dir}"
CONFIG_FILE="${CONFIG_FILE:-/workspace/models_config.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output}"
SERVER_PORT="${SERVER_PORT:-22266}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"

echo "============================================"
echo "  YOLOv8-TensorRT API Server"
echo "============================================"
echo "Models directory: ${MODELS_DIR}"
echo "Config file:      ${CONFIG_FILE}"
echo "Output directory:  ${OUTPUT_DIR}"
echo "Server:           ${SERVER_HOST}:${SERVER_PORT}"
echo ""

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Check if any .pt files exist and need conversion
PT_COUNT=$(find "${MODELS_DIR}" -name "*.pt" 2>/dev/null | wc -l)
ENGINE_COUNT=$(find "${MODELS_DIR}" -name "*.engine" 2>/dev/null | wc -l)

if [ "${PT_COUNT}" -gt 0 ] && [ "${ENGINE_COUNT}" -eq 0 ]; then
    echo "Found ${PT_COUNT} .pt model(s) but no .engine files."
    echo "Converting models to TensorRT engines..."
    echo ""
    python3 /workspace/scripts/convert_models.py \
        --models-dir "${MODELS_DIR}" \
        --config "${CONFIG_FILE}" \
        --fp16
    echo ""
    echo "Model conversion complete."
elif [ "${ENGINE_COUNT}" -gt 0 ]; then
    echo "Found ${ENGINE_COUNT} .engine file(s), skipping conversion."
else
    echo "WARNING: No .pt or .engine files found in ${MODELS_DIR}"
    echo "Please mount your model files to ${MODELS_DIR}"
fi

echo ""
echo "Starting API server..."
exec yolov8-api \
    --config "${CONFIG_FILE}" \
    --models-dir "${MODELS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --host "${SERVER_HOST}" \
    --port "${SERVER_PORT}" \
    "$@"
