# -*- coding: utf-8 -*-
"""
quantize.py —— 混合精度量化 + 边缘模型导出（模块C）

产物：
  - ONNX fp32：基线部署文件
  - ONNX fp16：半精度（混合精度），边缘设备推荐，体积减半、提速明显
  - TensorRT INT8：需要 TensorRT（Jetson / 带 TensorRT 的 GPU），可选

说明：训练阶段已经用了 AMP（混合精度），导出用 half=True 再得到 fp16 模型，
      两者合起来就是申报书里“混合精度量化”的可落地形态。
"""
import os
import time

import numpy as np
import torch
from ultralytics import YOLO


def _onnx_speed(onnx_path, imgsz=640, iters=100):
    """用 onnxruntime 测 ONNX 推理速度（CPU 即可，用于报告里的对照）。"""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    for _ in range(5):
        sess.run(None, {name: x})
    t0 = time.time()
    for _ in range(iters):
        sess.run(None, {name: x})
    dt = (time.time() - t0) / iters
    return 1.0 / dt


def export_onnx(weights, imgsz=640, out_dir="runs/export"):
    """导出 fp32 + fp16 ONNX 到 out_dir（各自独立命名），返回产物路径和大小、速度。"""
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(weights)
    base = os.path.splitext(os.path.basename(weights))[0]

    def _relocate(src, tag):
        # ultralytics 默认把 onnx 写到权重同目录、同名（best.onnx），
        # fp32/fp16 会互相覆盖，故导出后立即移到 out_dir 并独立命名。
        dst = os.path.join(out_dir, f"{base}_{tag}.onnx")
        shutil.move(src, dst)
        return dst

    # fp32（先导出，立即移走，避免被 fp16 覆盖）
    fp32_path = _relocate(
        model.export(format="onnx", imgsz=imgsz, half=False, simplify=True), "fp32")
    # fp16
    fp16_path = _relocate(
        model.export(format="onnx", imgsz=imgsz, half=True, simplify=True), "fp16")

    report = {}
    for tag, p in [("fp32", fp32_path), ("fp16", fp16_path)]:
        size_mb = os.path.getsize(p) / 1024 / 1024
        fps = _onnx_speed(p, imgsz)
        report[tag] = {"path": p, "size_MB": round(size_mb, 2), "fps_cpu": round(fps, 1)}
        print(f"[{tag}] {p}  大小 {size_mb:.2f} MB  CPU推理 {fps:.1f} FPS")
    return report


def export_tensorrt_int8(weights, data_yaml, imgsz=640):
    """导出 TensorRT INT8（需要安装 tensorrt + 带 TensorRT 的机器）。"""
    model = YOLO(weights)
    path = model.export(format="engine", imgsz=imgsz, int8=True, data=data_yaml)
    print(f"TensorRT INT8 引擎：{path}")
    return path
