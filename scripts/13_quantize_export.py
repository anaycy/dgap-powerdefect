# -*- coding: utf-8 -*-
"""
13_quantize_export.py —— 混合精度量化 + ONNX/TensorRT 导出（模块C）
运行：
    .venv/Scripts/python.exe scripts/13_quantize_export.py \
        --weights runs/finetune/dgap_r0.5/weights/best.pt --data data/data.yaml
成功标志：
    runs/export/ 下出现 *.onnx（fp32 + fp16），打印大小和 CPU 推理 FPS。
"""
import argparse

from dgap.quantize import export_onnx, export_tensorrt_int8


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="要导出的模型 best.pt")
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--int8", action="store_true", help="额外导出 TensorRT INT8（需装 TensorRT）")
    args = p.parse_args()

    export_onnx(args.weights, imgsz=args.imgsz)
    if args.int8:
        export_tensorrt_int8(args.weights, args.data, imgsz=args.imgsz)
    print("\n导出完成。可把 ONNX 文件拷贝到边缘设备（树莓派/Jetson）做现场演示。")


if __name__ == "__main__":
    main()
