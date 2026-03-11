#!/usr/bin/env python3
"""
Convert YOLOv8 .pt models to TensorRT .engine format.

Usage:
    python3 scripts/convert_models.py --models-dir /workspace/models_dir \
        --config /workspace/models_config.json

This script:
1. Reads model definitions from models_config.json
2. For each model, exports .pt -> .onnx (with end2end NMS)
3. Builds .onnx -> .engine using TensorRT
"""
import argparse
import json
import os
import sys

# Add repo root to path so we can import project modules
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert YOLOv8 .pt models to TensorRT .engine format"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="/workspace/models_dir",
        help="Directory containing .pt model files",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(REPO_ROOT, "models_config.json"),
        help="Path to models_config.json",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Build engines with FP16 precision"
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Confidence threshold for NMS",
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.65, help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--topk", type=int, default=100, help="Max number of detection bboxes"
    )
    parser.add_argument(
        "--input-shape",
        nargs="+",
        type=int,
        default=[1, 3, 640, 640],
        help="Model input shape",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device for export and build"
    )
    return parser.parse_args()


def export_pt_to_onnx(pt_path, args):
    """Export a .pt model to .onnx with end2end NMS."""
    from io import BytesIO

    import onnx
    import torch
    from ultralytics import YOLO

    from models.common import PostDetect, optim

    try:
        import onnxsim
    except ImportError:
        onnxsim = None

    PostDetect.conf_thres = args.conf_thres
    PostDetect.iou_thres = args.iou_thres
    PostDetect.topk = args.topk

    onnx_path = pt_path.replace(".pt", ".onnx")
    if os.path.exists(onnx_path):
        print(f"  ONNX already exists: {onnx_path}, skipping export")
        return onnx_path

    print(f"  Exporting {pt_path} -> {onnx_path}")
    b = args.input_shape[0]
    yolo_model = YOLO(pt_path)
    model = yolo_model.model.fuse().eval()
    for m in model.modules():
        optim(m)
        m.to(args.device)
    model.to(args.device)

    fake_input = torch.randn(args.input_shape).to(args.device)
    for _ in range(2):
        model(fake_input)

    with BytesIO() as f:
        torch.onnx.export(
            model,
            fake_input,
            f,
            opset_version=11,
            input_names=["images"],
            output_names=["num_dets", "bboxes", "scores", "labels"],
        )
        f.seek(0)
        onnx_model = onnx.load(f)

    onnx.checker.check_model(onnx_model)
    shapes = [b, 1, b, args.topk, 4, b, args.topk, b, args.topk]
    for i in onnx_model.graph.output:
        for j in i.type.tensor_type.shape.dim:
            j.dim_param = str(shapes.pop(0))

    if onnxsim is not None:
        try:
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, "simplify check failed"
        except Exception as e:
            print(f"  Simplifier failure: {e}")

    onnx.save(onnx_model, onnx_path)
    print(f"  ONNX export success: {onnx_path}")
    return onnx_path


def build_onnx_to_engine(onnx_path, args):
    """Build a TensorRT engine from an ONNX model."""
    from models import EngineBuilder

    engine_path = onnx_path.replace(".onnx", ".engine")
    if os.path.exists(engine_path):
        print(f"  Engine already exists: {engine_path}, skipping build")
        return engine_path

    print(f"  Building {onnx_path} -> {engine_path}")
    builder = EngineBuilder(onnx_path, args.device)
    builder.seg = False
    builder.build(
        fp16=args.fp16,
        input_shape=args.input_shape,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        topk=args.topk,
    )
    print(f"  Engine build success: {engine_path}")
    return engine_path


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    models = config.get("models", [])
    print(f"Found {len(models)} models in config")

    for model_cfg in models:
        name = model_cfg["name"]
        pt_file = model_cfg["ptFile"]
        pt_path = os.path.join(args.models_dir, pt_file)

        print(f"\nProcessing model: {name}")
        print(f"  algCode: {model_cfg['algCode']}")
        print(f"  algDesc: {model_cfg['algDesc']}")

        if not os.path.exists(pt_path):
            print(f"  WARNING: {pt_path} not found, skipping")
            continue

        # Step 1: Export .pt -> .onnx
        onnx_path = export_pt_to_onnx(pt_path, args)

        # Step 2: Build .onnx -> .engine
        build_onnx_to_engine(onnx_path, args)

    print("\nAll model conversions complete!")


if __name__ == "__main__":
    main()
