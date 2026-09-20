# -*- coding: utf-8 -*-
"""
DGAP —— Defect-Guided Adaptive Pruning
面向电力边缘设备的缺陷感知自适应轻量化检测系统

模块：
  sensitivity.py  缺陷感知通道敏感性分析（模块A 的创新点）
  pruner.py       自适应结构化剪枝（模块A）
  distill.py      缺陷区域知识蒸馏（模块B）
  quantize.py     混合精度量化 + ONNX/TensorRT 导出（模块C）
"""
from .sensitivity import DefectSensitivity, compute_sensitivity
from .pruner import prune_uniform, prune_dgap
from .quantize import export_onnx

__all__ = [
    "DefectSensitivity",
    "compute_sensitivity",
    "prune_uniform",
    "prune_dgap",
    "export_onnx",
]
