# -*- coding: utf-8 -*-
"""
distill.py —— 缺陷区域知识蒸馏（模块B）

背景：剪枝会伤小目标缺陷的精度。模块B 让剪枝后的“学生”模型在微调时，
      额外对齐“教师”模型（原始高精度模型）的中间特征，并且【缺陷区域】
      对齐权重更高 —— 从而缓解小目标缺陷的精度暴跌。

损失 = lambda_det * 检测损失(学生 vs GT)
     + lambda_feat * Σ_l 缺陷加权 ||teacher_feat_l - student_feat_l||^2

实现说明：
  - 检测损失复用 ultralytics 的 v8DetectionLoss；
  - 特征蒸馏用 forward hook 抽取教师/学生的中间特征；
  - 缺陷加权用标注框栅格化掩码（同 sensitivity.py）。

注意：本模块依赖 ultralytics 内部 API（build_dataloader / v8DetectionLoss /
DetectionModel），不同版本可能有细微差异。首次运行时若报导入错误，
我们会按当前 ultralytics 版本微调（阶段2一起调试）。
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm


def _load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _xywh2xyxy(boxes, imgsz):
    """归一化 cxcywh -> 像素 xyxy。"""
    b = np.array(boxes, dtype=np.float32).reshape(-1, 4) * imgsz
    x1 = b[:, 0] - b[:, 1] / 2
    y1 = b[:, 2] - b[:, 3] / 2
    x2 = b[:, 0] + b[:, 1] / 2
    y2 = b[:, 2] + b[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def _defect_mask(boxes, H, W, bg_weight=0.05):
    """像素级缺陷掩码 (H,W)，框内=1，背景=bg_weight。"""
    mask = np.full((H, W), bg_weight, dtype=np.float32)
    for (x1, y1, x2, y2) in boxes:
        X1, Y1 = int(x1), int(y1)
        X2, Y2 = int(np.ceil(x2)), int(np.ceil(y2))
        X1, Y1 = max(0, X1), max(0, Y1)
        X2, Y2 = min(W, X2), min(H, Y2)
        if X2 > X1 and Y2 > Y1:
            mask[Y1:Y2, X1:X2] = 1.0
    return mask


def _hook_features(model, names):
    """返回 (hook_handles, feature_dict)，抽取指定模块名对应的输出特征。"""
    feats = {}

    def make(name):
        def fn(m, inp, out):
            feats[name] = out
        return fn

    handles = []
    for n, m in model.named_modules():
        if n in names:
            handles.append(m.register_forward_hook(make(n)))
    return handles, feats


def defect_region_distill(teacher, student, data_yaml, epochs=60, batch=16,
                          imgsz=640, lambda_det=1.0, lambda_feat=5.0,
                          device="cuda", feat_names=None, lr=2e-4):
    """
    缺陷区域知识蒸馏主流程。
    teacher / student：ultralytics DetectionModel（已加载，student 为剪枝后模型）。
    """
    from ultralytics.data.build import build_dataloader
    from ultralytics.utils import RANK
    from ultralytics.utils.loss import v8DetectionLoss

    cfg = _load_yaml(data_yaml)
    root = cfg["path"]

    # 教师冻结，只评估；学生可训练
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student.to(device).train()

    # 默认蒸馏层：主干/颈部几个有代表性的卷积输出
    if feat_names is None:
        feat_names = ["model.2.cv1.conv", "model.4.cv1.conv", "model.9.cv1.conv"]

    t_handles, t_feats = _hook_features(teacher, feat_names)
    s_handles, s_feats = _hook_features(student, feat_names)

    det_loss = v8DetectionLoss(student)
    train_cfg = {"train": os.path.join(root, cfg.get("train", "images/train"))}
    loader = build_dataloader(train_cfg, batch=batch, workers=4, shuffle=True, rank=RANK)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, student.parameters()), lr=lr)

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"蒸馏 epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            img = batch["img"].to(device) / 255.0

            # 教师前向（抽特征）
            with torch.no_grad():
                teacher(img)

            # 学生前向（抽特征 + 出预测）
            preds = student(img)

            # 1) 检测损失
            loss_det, _ = det_loss(preds, batch)

            # 2) 缺陷区域特征蒸馏损失
            loss_feat = torch.tensor(0.0, device=device)
            for name in feat_names:
                t = t_feats.get(name)
                s = s_feats.get(name)
                if t is None or s is None:
                    continue
                # 对齐分辨率（教师/学生结构相同则一致；不同则插值）
                if t.shape[-2:] != s.shape[-2:]:
                    s = F.interpolate(s, size=t.shape[-2:], mode="bilinear", align_corners=False)
                # 缺陷加权掩码
                b, _, H, W = t.shape
                mask = torch.zeros(b, 1, H, W, device=device)
                for bi in range(b):
                    boxes = batch["bboxes"][bi].cpu().numpy()
                    m = _defect_mask(boxes, H, W)
                    mask[bi, 0] = torch.from_numpy(m).to(device)
                diff = (t - s) ** 2 * mask
                loss_feat = loss_feat + diff.sum() / (mask.sum() + 1e-6)

            loss = lambda_det * loss_det + lambda_feat * loss_feat
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss_det=f"{loss_det.item():.3f}", loss_feat=f"{loss_feat.item():.3f}")

    for h in t_handles + s_handles:
        h.remove()
    print("蒸馏完成，学生模型已微调。")
    return student
