# -*- coding: utf-8 -*-
"""
sensitivity.py —— 缺陷感知通道敏感性分析（模块A 的核心创新点）

为什么不是普通剪枝？
  普通剪枝用 L1/L2 范数或 BN-gamma 判断通道重要性，对所有层“一视同仁”，
  是【全局统一剪枝】—— 会误删对小目标缺陷最关键的通道，导致缺陷 mAP 暴跌。

本模块做法（缺陷感知）：
  1) 跑若干张带标注的电力缺陷图，hook 每个卷积层的输出特征图；
  2) 对每个通道，计算它在【缺陷区域(标注框内)】的激活能量，得到该通道
     对缺陷的贡献度 defect_sensitivity；
  3) 通道对缺陷越敏感 → 剪枝时越要保留 → 这就是“自适应”的来源。

输出：results/sensitivity/defect_sensitivity.pkl
      {layer_name: np.ndarray(C,)  # 每个通道的缺陷敏感度(0~1)，越大越要保留}
"""
import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm


def _load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _xywh2xyxy(boxes):
    """(N,4) 归一化 cx,cy,w,h -> x1,y1,x2,y2"""
    b = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    x1 = b[:, 0] - b[:, 1] / 2
    y1 = b[:, 2] - b[:, 3] / 2
    x2 = b[:, 0] + b[:, 1] / 2
    y2 = b[:, 2] + b[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)


class DefectSensitivity:
    """给模型每个卷积层挂 hook，统计缺陷区域加权激活能量。"""

    def __init__(self, model, device="cuda"):
        self.model = model              # ultralytics DetectionModel（即 YOLO('x.pt').model）
        self.device = device
        self.hooks = []
        self.activations = {}           # name -> list[Tensor]

    def _hook_fn(self, name):
        def fn(module, inp, out):
            self.activations[name].append(out.detach())
        return fn

    def register(self):
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                self.activations[name] = []
                self.hooks.append(m.register_forward_hook(self._hook_fn(name)))
        return self

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = {}

    @staticmethod
    def _defect_mask(boxes, H, W, bg_weight=0.05):
        """把归一化标注框栅格化到 (H,W)：框内=1，背景=bg_weight。"""
        mask = np.full((H, W), bg_weight, dtype=np.float32)
        for (x1, y1, x2, y2) in boxes:
            X1, Y1 = int(x1 * W), int(y1 * H)
            X2, Y2 = int(np.ceil(x2 * W)), int(np.ceil(y2 * H))
            X1, Y1 = max(0, X1), max(0, Y1)
            X2, Y2 = min(W, X2), min(H, Y2)
            if X2 > X1 and Y2 > Y1:
                mask[Y1:Y2, X1:X2] = 1.0
        return mask

    @torch.no_grad()
    def _forward_image(self, img, boxes_xyxy):
        """跑一张图，返回每个卷积层的 (每通道缺陷激活得分)。"""
        for k in self.activations:
            self.activations[k].clear()
        self.model(img.unsqueeze(0).to(self.device))

        layer_score = {}
        for name, outs in self.activations.items():
            if not outs:
                continue
            act = outs[-1]                        # (1, C, H, W)
            C, H, W = act.shape[1], act.shape[2], act.shape[3]
            mask = self._defect_mask(boxes_xyxy, H, W)
            mask_t = torch.from_numpy(mask).to(self.device).view(1, 1, H, W)
            # 缺陷区域加权平均激活能量 = 该通道对缺陷的贡献度量
            weighted = (act * act * mask_t).sum(dim=(2, 3))   # (1, C)
            area = mask_t.sum() + 1e-6
            score = (weighted / area).squeeze(0).cpu().numpy()  # (C,)
            layer_score[name] = score
        return layer_score


def compute_sensitivity(model, data_yaml, device="cuda", imgsz=640,
                        num_images=32, out_path="results/sensitivity/defect_sensitivity.pkl"):
    """遍历 num_images 张验证图，汇总每层每通道缺陷敏感度并保存 pkl。"""
    cfg = _load_yaml(data_yaml)
    root = cfg["path"]
    val_img_dir = os.path.join(root, cfg["val"])
    # 标签目录：把 images/<split> 对应到 labels/<split>
    split_name = os.path.basename(os.path.normpath(cfg["val"]))
    val_lbl_dir = os.path.join(root, "labels", split_name)

    imgs = sorted(os.listdir(val_img_dir))
    imgs = [i for i in imgs if i.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))][:num_images]
    if not imgs:
        raise FileNotFoundError(f"在 {val_img_dir} 没找到图片，先整理数据集（见 data/README.md）")

    ds = DefectSensitivity(model, device)
    ds.register()
    model.to(device).eval()

    acc = {}   # name -> [sum_score, count]
    for fn in tqdm(imgs, desc="缺陷敏感性分析"):
        img_path = os.path.join(val_img_dir, fn)
        lbl_path = os.path.join(val_lbl_dir, os.path.splitext(fn)[0] + ".txt")
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (imgsz, imgsz))
        x = torch.from_numpy(img[:, :, ::-1].transpose(2, 0, 1)).float() / 255.0  # BGR->RGB, CHW

        boxes = []
        if os.path.exists(lbl_path):
            for line in open(lbl_path, encoding="utf-8"):
                p = line.strip().split()
                if len(p) >= 5:
                    boxes.append([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
        boxes_xyxy = _xywh2xyxy(boxes) if boxes else np.zeros((0, 4), dtype=np.float32)

        layer_score = ds._forward_image(x, boxes_xyxy)
        for name, s in layer_score.items():
            acc.setdefault(name, [0.0, 0])
            acc[name][0] += s
            acc[name][1] += 1

    ds.close()

    sensitivity = {}
    for name, (s, c) in acc.items():
        mean = s / max(c, 1)
        mx = mean.max() + 1e-8
        sensitivity[name] = (mean / mx).astype(np.float32)  # 归一化到 0~1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(sensitivity, f)

    print(f"\n敏感性分析完成：共 {len(sensitivity)} 个卷积层 -> {out_path}")
    avg = {k: float(v.mean()) for k, v in sensitivity.items()}
    top = sorted(avg.items(), key=lambda kv: -kv[1])[:5]
    print("缺陷最敏感的 5 个层（剪枝时重点保留）：")
    for n, s in top:
        print(f"  {n:40s} 平均敏感度 {s:.3f}")
    return sensitivity
