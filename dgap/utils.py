# -*- coding: utf-8 -*-
"""DGAP 公共工具函数。"""
import torch


def torch_device(d):
    """ultralytics 风格设备 '0'/'cpu' -> torch 风格 'cuda:0'/'cpu'。

    ultralytics 用 '0' 表示第 0 块 GPU，但 torch 的 tensor.to() 只认
    'cuda:0' / 'cpu'，所以这里做一次转换。
    """
    if not torch.cuda.is_available():
        return "cpu"
    d = str(d)
    if d == "cpu":
        return "cpu"
    d = d.split(",")[0].strip()      # 多卡只取第一块
    return f"cuda:{d}" if d.isdigit() else d
