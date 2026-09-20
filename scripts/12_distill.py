# -*- coding: utf-8 -*-
"""
12_distill.py —— 缺陷区域知识蒸馏（模块B）
前提：已得到剪枝后的模型（runs/pruned/dgap_r*.pt 或微调后的 best.pt）
运行：
    .venv/Scripts/python.exe scripts/12_distill.py \
        --student runs/pruned/dgap_r0.5.pt \
        --teacher runs/baseline/yolov8s/weights/best.pt \
        --data data/data.yaml --epochs 60
成功标志：
    runs/distilled/student_distilled.pt 生成。
"""
import argparse
import os
import sys

# 让 scripts/ 下的脚本能 import 到项目根的 dgap 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ultralytics import YOLO

from dgap.distill import defect_region_distill


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True, help="剪枝后的学生模型")
    p.add_argument("--teacher", required=True, help="原始高精度教师模型")
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--device", default="0")
    args = p.parse_args()

    teacher = YOLO(args.teacher).model
    student = YOLO(args.student).model

    student = defect_region_distill(teacher, student, args.data,
                                    epochs=args.epochs, device=args.device)

    os.makedirs("runs/distilled", exist_ok=True)
    ckpt = {"model": student}
    torch.save(ckpt, "runs/distilled/student_distilled.pt")
    print("\n蒸馏模型已保存到 runs/distilled/student_distilled.pt")
    print("下一步：scripts/02_metrics.py 计算蒸馏后指标")


if __name__ == "__main__":
    main()
