# -*- coding: utf-8 -*-
"""
distill.py —— 缺陷区域知识蒸馏（模块B）

背景：剪枝会伤小目标缺陷的精度。模块B 让剪枝后的“学生”模型在微调时，
      额外对齐“教师”模型（原始高精度模型）的中间特征，并且【缺陷区域】
      对齐权重更高 —— 从而缓解小目标缺陷的精度暴跌。

损失 = lambda_det * 检测损失(学生 vs GT)
     + lambda_feat * Σ_l 缺陷加权 ||teacher_feat_l - student_feat_l||^2

实现说明：
  - 检测损失直接复用学生模型自带的 loss（ultralytics v8DetectionLoss），
    由它统一处理图像归一化 / 目标分配 / 坐标换算，避免手写一堆内部细节；
  - 特征蒸馏用 forward hook 抽取教师/学生的中间卷积输出；
  - 剪枝后学生通道数 < 教师，用 1x1 卷积做通道对齐（projection）；
  - 缺陷加权掩码：batch["bboxes"] 是【归一化 cxcywh】，先转 xyxy 再按
    特征图分辨率 H/W 缩放（不是直接用 640 像素坐标画在 80x80 图上）。

注意：本模块依赖 ultralytics 内部 API，随版本变化可能有细微差异；
      首次运行报错时按当前版本微调即可。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .utils import torch_device


def _defect_mask(boxes_xywh, H, W, bg_weight=0.05):
    """把 (N,4) 归一化 cxcywh 栅格化到特征图 (H,W)：框内=1，背景=bg_weight。

    特征图是原图下采样 H/640 倍，故坐标乘 H、W（而非 640）。
    """
    mask = np.full((H, W), bg_weight, dtype=np.float32)
    if len(boxes_xywh) == 0:
        return mask
    b = np.asarray(boxes_xywh, dtype=np.float32).reshape(-1, 4)
    x1 = (b[:, 0] - b[:, 2] / 2) * W   # cx - w/2
    y1 = (b[:, 1] - b[:, 3] / 2) * H   # cy - h/2
    x2 = (b[:, 0] + b[:, 2] / 2) * W   # cx + w/2
    y2 = (b[:, 1] + b[:, 3] / 2) * H   # cy + h/2
    for X1, Y1, X2, Y2 in zip(x1, y1, x2, y2):
        X1, Y1 = int(np.floor(X1)), int(np.floor(Y1))
        X2, Y2 = int(np.ceil(X2)), int(np.ceil(Y2))
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
                          device="cuda", feat_names=None, lr=2e-4, workers=0):
    """
    缺陷区域知识蒸馏主流程。
    teacher / student：ultralytics DetectionModel（已加载，student 为剪枝后模型）。
    workers：DataLoader 进程数。Windows 上多进程需脚本有 __main__ 保护，
             为稳妥默认 0（单进程），Linux 可调大。
    """
    from ultralytics.data.build import build_yolo_dataset, build_dataloader
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG, RANK
    from ultralytics.utils.torch_utils import initialize_weights

    device = torch_device(device)

    # 教师/学生统一重参数化为 C2f_v2，保证两者的中间层名一一对应
    # （学生是剪枝后的 C2f_v2，教师是原始 C2f，需同样替换，否则层名对不上）。
    from .c2f_v2 import replace_c2f_with_c2f_v2

    replace_c2f_with_c2f_v2(teacher)
    replace_c2f_with_c2f_v2(student)
    initialize_weights(teacher)
    initialize_weights(student)

    # 教师冻结只评估；学生可训练
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student.to(device).train()

    # 加载后的模型 args 是 dict，而 v8DetectionLoss 需要 .box/.cls/.dfl 属性访问，
    # 这里统一换成 DEFAULT_CFG（namespace），保证 loss 正常。
    student.args = DEFAULT_CFG

    # ---- 正确构建训练 DataLoader（传完整 data cfg）----
    data = check_det_dataset(data_yaml)
    args = DEFAULT_CFG
    args.batch = batch
    args.imgsz = imgsz
    args.workers = workers
    stride = int(student.stride.max()) if hasattr(student, "stride") else 32
    dataset = build_yolo_dataset(args, data["train"], batch, data, mode="train",
                                 rect=False, stride=stride)
    loader = build_dataloader(dataset, batch=batch, workers=workers,
                              shuffle=True, rank=RANK, device=device)

    # ---- 蒸馏层：主干/颈部代表性卷积输出（C2f_v2 的 cv0/cv1 + SPPF 的 cv1）----
    if feat_names is None:
        feat_names = ["model.2.cv0.conv", "model.2.cv1.conv",
                      "model.4.cv0.conv", "model.4.cv1.conv",
                      "model.9.cv1.conv"]

    t_handles, t_feats = _hook_features(teacher, feat_names)
    s_handles, s_feats = _hook_features(student, feat_names)

    # ---- 通道对齐投影层：剪枝后学生通道数 < 教师（用普通 dict，键含 "." 无法用 ModuleDict）----
    proj = {}
    for name in feat_names:
        t_ch = teacher.get_submodule(name).out_channels
        s_ch = student.get_submodule(name).out_channels
        if t_ch != s_ch:
            proj[name] = nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False).to(device)

    opt_params = list(student.parameters())
    for conv in proj.values():
        opt_params += list(conv.parameters())
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, opt_params), lr=lr)

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"蒸馏 epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            # 预处理：张量搬到设备 + 图像 uint8 -> float 0~1（等价 trainer.preprocess_batch）
            for k in list(batch.keys()):
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device, non_blocking=True)
            batch["img"] = batch["img"].float() / 255.0

            # 教师前向（只抽特征，不更新）
            with torch.no_grad():
                teacher(batch["img"])

            # 学生检测损失（内部 forward 同时填充 s_feats hook）；
            # v8DetectionLoss 返回的是 (box, cls, dfl) 三元张量，需 sum 成标量。
            loss_det = student.loss(batch)[0].sum()

            # 缺陷区域特征蒸馏损失
            loss_feat = torch.tensor(0.0, device=device)
            for name in feat_names:
                t = t_feats.get(name)
                s = s_feats.get(name)
                if t is None or s is None:
                    continue
                if name in proj:
                    s = proj[name](s)
                if t.shape[-2:] != s.shape[-2:]:
                    s = F.interpolate(s, size=t.shape[-2:], mode="bilinear",
                                      align_corners=False)
                # 缺陷加权掩码（按特征图分辨率缩放）
                b, _, H, W = t.shape
                mask = torch.zeros(b, 1, H, W, device=device)
                batch_idx = batch["batch_idx"]
                for bi in range(b):
                    boxes = batch["bboxes"][batch_idx == bi].cpu().numpy()
                    m = _defect_mask(boxes, H, W)
                    mask[bi, 0] = torch.from_numpy(m).to(device)
                diff = (t - s) ** 2 * mask
                loss_feat = loss_feat + diff.sum() / (mask.sum() + 1e-6)

            loss = lambda_det * loss_det + lambda_feat * loss_feat
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss_det=f"{loss_det.item():.3f}",
                             loss_feat=f"{loss_feat.item():.3f}")

    for h in t_handles + s_handles:
        h.remove()
    print("蒸馏完成，学生模型已微调。")
    return student
