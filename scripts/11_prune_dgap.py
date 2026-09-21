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

流程：加载 -> (defect 模式先读敏感性) -> 剪枝 -> 保存剪枝后模型 -> 微调 -> best.pt
成功标志：
    runs/pruned/<tag>_r<ratio>.pt 生成；
    若加 --fine-tune，runs/finetune/<tag>_r<ratio>/weights/best.pt 生成。

注意：剪枝改变了网络结构（通道数），ultralytics 无法通过 YOLO(path) 重建
      C2f_v2 结构的剪枝模型，因此微调直接【把剪枝后的 DetectionModel 注入
      DetectionTrainer】，而不是 `YOLO(out).train()`。
"""
import argparse
import os
import pickle
import sys
import torch

# 让 scripts/ 下的脚本能 import 到项目根的 dgap 包（含 C2f_v2，供重载剪枝权重）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

from dgap.pruner import prune_uniform, prune_dgap


def _count_params(m):
    return sum(p.numel() for p in m.parameters())


def fine_tune(model, data, epochs, imgsz, device, save_dir, batch=16, workers=0, model_path="yolov8s.pt"):
    """把已剪枝的 DetectionModel 直接注入 DetectionTrainer 做微调。

    返回 best.pt 路径。model 是剪枝后的 DetectionModel（含 C2f_v2）。
    model_path 仅作 trainer 构造时的占位（trainer.model 随后会被剪枝模型覆盖）。
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    overrides = {
        "model": model_path,          # 占位：trainer.__init__ 需要非空 model，随后被覆盖
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "device": device,
        "batch": batch,
        "workers": workers,
        "pretrained": False,          # 不加载预训练权重，直接用注入的剪枝模型
        "save_dir": save_dir,         # 直接指定输出目录（ultralytics 8.4 支持）
        "exist_ok": True,
        "plots": False,               # 省时间，不画训练曲线
    }
    trainer = DetectionTrainer(overrides=overrides)
    trainer.model = model             # nn.Module -> setup_model 跳过重建
    trainer.train()
    return str(trainer.best)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="原始模型 best.pt")
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--mode", default="defect", choices=["uniform", "defect"])
    p.add_argument("--ratio", type=float, default=0.5, help="整体剪枝比例 0~1")
    p.add_argument("--sensitivity", default="results/sensitivity/defect_sensitivity.pkl")
    p.add_argument("--fine-tune", action="store_true", help="剪枝后立即微调（推荐）")
    p.add_argument("--epochs", type=int, default=80, help="微调轮数")
    p.add_argument("--batch", type=int, default=16, help="微调 batch size")
    p.add_argument("--imgsz", type=int, default=640, help="微调图像尺寸")
    p.add_argument("--device", default="0")
    args = p.parse_args()

    model = YOLO(args.weights)
    m = model.model                       # DetectionModel
    n0 = _count_params(m)
    ignored = [m.model[-1]]               # 不剪检测头，保持类别输出不变

    if args.mode == "defect":
        if not os.path.exists(args.sensitivity):
            raise SystemExit("还没做敏感性分析：先跑 scripts/10_sensitivity.py")
        sens = pickle.load(open(args.sensitivity, "rb"))
        prune_dgap(m, sens, global_ratio=args.ratio, ignored_layers=ignored)
        tag = "dgap"
    else:
        prune_uniform(m, ratio=args.ratio, ignored_layers=ignored)
        tag = "uniform"

    n1 = _count_params(m)
    print(f"\n剪枝完成：参数量 {n0/1e6:.2f}M -> {n1/1e6:.2f}M（压缩 {(1-n1/n0)*100:.1f}%）")

    # 保存剪枝后（未微调）模型
    os.makedirs("runs/pruned", exist_ok=True)
    out = f"runs/pruned/{tag}_r{args.ratio}.pt"
    ckpt = {"model": m, "train_args": getattr(model, "trainer", None) or {}}
    torch.save(ckpt, out)
    print(f"剪枝后模型 -> {out}")

    if args.fine_tune:
        best = fine_tune(m, args.data, args.epochs, args.imgsz, args.device,
                         save_dir=f"runs/finetune/{tag}_r{args.ratio}", batch=args.batch,
                         model_path=args.weights)
        print(f"\n微调完成 -> {best}")
    print("下一步：scripts/02_metrics.py 计算剪枝后指标（记得 --method 区分方法名）")


if __name__ == "__main__":
    main()
