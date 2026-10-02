# -*- coding: utf-8 -*-
"""
25_train_ir.py —— 红外热缺陷模型训练（OverheatDL，单类 hot=发热）

数据：data_ir/data_ir.yaml（由 26_convert_ir.py 生成，nc=1, names=[hot]）
运行：
    .venv/Scripts/python.exe scripts/25_train_ir.py --model yolov8s --epochs 100
成功标志：
    runs/ir/yolov8s/weights/best.pt 生成。
"""
import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8s", help="yolov8n / yolov8s / yolov8m")
    p.add_argument("--data", default="data_ir/data_ir.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    args = p.parse_args()

    model = YOLO(f"{args.model}.pt")

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        save_dir=f"runs/ir/{args.model}",
        exist_ok=True,
        patience=30,
        save=True,
        val=True,
        plots=True,
        seed=42,
        optimizer="auto",
        lr0=0.01,
        amp=True,
        workers=4,
    )

    print(f"\n红外模型训练完成，best 权重：runs/ir/{args.model}/weights/best.pt")
    print("下一步：跑融合 demo，或 13_quantize_export.py 导出红外 ONNX 供边缘部署")


if __name__ == "__main__":
    main()
