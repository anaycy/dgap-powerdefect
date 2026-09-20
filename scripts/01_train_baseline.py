# -*- coding: utf-8 -*-
"""
01_train_baseline.py —— Baseline（原始高精度模型）训练
运行：
    .venv/Scripts/python.exe scripts/01_train_baseline.py --model yolov8s --epochs 100 --batch 16
成功标志：
    - runs/baseline/yolov8s/ 下出现 weights/best.pt 和 weights/last.pt
    - 训练结束打印 "训练完成，best 权重路径：..."
说明：
    - yolov8s 为推荐基线（精度和速度平衡，压缩空间大）；
      想更快可用 --model yolov8n，想更高精度用 yolov8m。
    - 首次运行会自动下载 yolov8s.pt 预训练权重（约 22MB）。
"""
import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8s", help="yolov8n / yolov8s / yolov8m")
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    args = p.parse_args()

    # 用预训练权重初始化（.pt 会自动下载到当前目录）
    model = YOLO(f"{args.model}.pt")

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        save_dir=f"runs/baseline/{args.model}",   # ultralytics 8.4 用 save_dir 指定输出目录
        exist_ok=True,
        patience=30,        # 30 轮 mAP 不涨就早停
        save=True,
        val=True,
        plots=True,         # 生成 results.png / 混淆矩阵等
        seed=42,
        optimizer="auto",   # 自适应选 SGD/AdamW
        lr0=0.01,
        amp=True,           # 混合精度训练（RTX 4060 支持，省显存提速）
        workers=4,
    )

    print(f"\n训练完成，best 权重路径：runs/baseline/{args.model}/weights/best.pt")
    print("下一步：运行 scripts/02_metrics.py 计算 Params/FLOPs/mAP/FPS")


if __name__ == "__main__":
    main()
