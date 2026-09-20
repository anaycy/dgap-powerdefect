# -*- coding: utf-8 -*-
"""
11_prune_dgap.py —— 结构化剪枝（普通对照 vs DGAP 自适应）

运行（DGAP 自适应剪枝）：
    .venv/Scripts/python.exe scripts/11_prune_dgap.py \
        --weights runs/baseline/yolov8s/weights/best.pt --data data/data.yaml \
        --mode defect --ratio 0.5 --fine-tune
运行（普通剪枝对照）：
    .venv/Scripts/python.exe scripts/11_prune_dgap.py \
        --weights runs/baseline/yolov8s/weights/best.pt --data data/data.yaml \
        --mode uniform --ratio 0.5 --fine-tune

流程：加载 -> (defect 模式先读敏感性) -> 剪枝 -> 保存 -> 微调 -> 微调后 best.pt
成功标志：
    runs/pruned/<tag>_r<ratio>.pt 生成；
    若加 --fine-tune，runs/finetune/<tag>_r<ratio>/weights/best.pt 生成。
"""
import argparse
import os
import pickle
import sys
import torch

# 让 scripts/ 下的脚本能 import 到项目根的 dgap 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="原始模型 best.pt")
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--mode", default="defect", choices=["uniform", "defect"])
    p.add_argument("--ratio", type=float, default=0.5, help="整体剪枝比例 0~1")
    p.add_argument("--sensitivity", default="results/sensitivity/defect_sensitivity.pkl")
    p.add_argument("--fine-tune", action="store_true", help="剪枝后立即微调（推荐）")
    p.add_argument("--epochs", type=int, default=80, help="微调轮数")
    p.add_argument("--device", default="0")
    args = p.parse_args()

    model = YOLO(args.weights)
    m = model.model                       # DetectionModel
    ignored = [m.model[-1]]               # 不剪检测头，保持类别输出不变

    if args.mode == "defect":
        from dgap.pruner import prune_dgap
        if not os.path.exists(args.sensitivity):
            raise SystemExit("还没做敏感性分析：先跑 scripts/10_sensitivity.py")
        sens = pickle.load(open(args.sensitivity, "rb"))
        prune_dgap(m, sens, global_ratio=args.ratio, ignored_layers=ignored)
        tag = "dgap"
    else:
        from dgap.pruner import prune_uniform
        prune_uniform(m, ratio=args.ratio, ignored_layers=ignored)
        tag = "uniform"

    # 保存剪枝后的模型（含当前模型结构，可直接被 YOLO() 加载）
    os.makedirs("runs/pruned", exist_ok=True)
    out = f"runs/pruned/{tag}_r{args.ratio}.pt"
    ckpt = {"model": m, "train_args": getattr(model, "trainer", None) or {}}
    torch.save(ckpt, out)
    print(f"\n剪枝完成 -> {out}")
    print(f"参数量对比：剪枝前 {_count_params(YOLO(args.weights).model)/1e6:.2f}M"
          f" -> 剪枝后 {_count_params(m)/1e6:.2f}M")

    if args.fine_tune:
        pm = YOLO(out)
        pm.train(data=args.data, epochs=args.epochs, imgsz=640, device=args.device,
                 save_dir=f"runs/finetune/{tag}_r{args.ratio}", exist_ok=True)
        print(f"\n微调完成 -> runs/finetune/{tag}_r{args.ratio}/weights/best.pt")
    print("下一步：scripts/02_metrics.py 计算剪枝后指标（记得 --method 区分方法名）")


def _count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    main()
