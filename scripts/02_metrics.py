# -*- coding: utf-8 -*-
"""
02_metrics.py —— 计算四项指标：Params / FLOPs / mAP / FPS
运行：
    .venv/Scripts/python.exe scripts/02_metrics.py \
        --weights runs/baseline/yolov8s/weights/best.pt \
        --data data/data.yaml --method "Baseline(YOLOv8s)"
输出：
    results/metrics/<名称>.json        （单项指标明细）
    results/experiment_results.json   （自动追加到实验汇总表，供 03_make_table.py 用）
"""
import argparse
import json
import os
import time

import torch
from ultralytics import YOLO


def _torch_device(d):
    """ultralytics 风格设备 '0'/'cpu' -> torch 风格 'cuda:0'/'cpu'。"""
    if not torch.cuda.is_available():
        return "cpu"
    d = str(d)
    if d == "cpu":
        return "cpu"
    d = d.split(",")[0].strip()          # 多卡时只取第一块
    return f"cuda:{d}" if d.isdigit() else d


def benchmark_fps(model, device, imgsz, warmup=20, iters=100):
    """纯网络推理 FPS（fp16, batch=1），论文常用口径。"""
    model.model.to(device).half().eval()
    x = torch.randn(1, 3, imgsz, imgsz).to(device).half()
    with torch.no_grad():
        for _ in range(warmup):
            model.model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model.model(x)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
    return 1.0 / dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="模型权重 .pt")
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--method", default=None, help="方法名，写进汇总表，如 'Baseline(YOLOv8s)'")
    p.add_argument("--split", default="val", help="mAP 用哪个集：val / test")
    p.add_argument("--no-append", action="store_true", help="只算不写入汇总表")
    args = p.parse_args()

    device = _torch_device(args.device)
    model = YOLO(args.weights)

    # ---- 1) Params ----
    n_param = sum(p.numel() for p in model.model.parameters())

    # ---- 2) FLOPs（thop 返回 MACs，乘 2 才是 FLOPs）----
    from thop import profile
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz).to(device)
    macs, _ = profile(model.model.to(device).eval(), inputs=(dummy,), verbose=False)
    flops = macs * 2

    # ---- 3) mAP ----
    # split 不存在时 ultralytics 会回退到 val；这里手动兜底
    try:
        m = model.val(data=args.data, imgsz=args.imgsz, device=device,
                      split=args.split, verbose=False)
    except Exception:
        m = model.val(data=args.data, imgsz=args.imgsz, device=device,
                      split="val", verbose=False)
    map50 = float(m.box.map50)
    map50_95 = float(m.box.map)

    # ---- 4) FPS ----
    fps = benchmark_fps(model, device, args.imgsz)

    # ---- 模型文件大小 ----
    size_mb = os.path.getsize(args.weights) / 1024 / 1024

    method = args.method or os.path.basename(args.weights).replace(".pt", "")
    out = {
        "method": method,
        "params_M": round(n_param / 1e6, 2),
        "flops_G": round(flops / 1e9, 2),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "fps": round(fps, 1),
        "size_MB": round(size_mb, 2),
        "weights": args.weights,
    }

    os.makedirs("results/metrics", exist_ok=True)
    name = os.path.basename(args.weights).replace(".pt", "")
    with open(f"results/metrics/{name}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---- 追加到实验汇总表 ----
    if not args.no_append:
        agg_path = "results/experiment_results.json"
        agg = {"dataset": args.data, "methods": []}
        if os.path.exists(agg_path):
            agg = json.load(open(agg_path, encoding="utf-8"))
        # 同名方法覆盖（方便重跑）
        agg["methods"] = [r for r in agg["methods"] if r["method"] != method]
        agg["methods"].append(out)
        json.dump(agg, open(agg_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n已写入 results/metrics/{name}.json 和 results/experiment_results.json")


if __name__ == "__main__":
    main()
