# -*- coding: utf-8 -*-
"""
10_sensitivity.py —— 缺陷感知通道敏感性分析（模块A 前置步骤）
运行：
    .venv/Scripts/python.exe scripts/10_sensitivity.py \
        --weights runs/baseline/yolov8s/weights/best.pt --data data/data.yaml
成功标志：
    results/sensitivity/defect_sensitivity.pkl 生成，并打印“缺陷最敏感的 5 个层”。
"""
import argparse

from ultralytics import YOLO

from dgap.sensitivity import compute_sensitivity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="原始高精度模型 best.pt")
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--num-images", type=int, default=32, help="分析用图片数，越多越准")
    p.add_argument("--device", default="0")
    args = p.parse_args()

    model = YOLO(args.weights)
    compute_sensitivity(model.model, args.data, device=args.device,
                        num_images=args.num_images)
    print("\n完成。下一步：scripts/11_prune_dgap.py 做自适应剪枝")


if __name__ == "__main__":
    main()
