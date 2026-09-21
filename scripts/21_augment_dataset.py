# -*- coding: utf-8 -*-
"""
21_augment_dataset.py —— 离线数据增强，重点【过采样小缺陷样本】

思路：公开数据集里"销钉缺失/破损"这类小缺陷样本占比少，直接训练会被大目标淹没。
本脚本对【含小目标】的图做更多次增强（旋转/缩放/亮度/翻转），把小缺陷样本
虚拟扩充到与大目标样本数量相当，缓解类别/尺度不均衡。

默认只增强 train 集（val/test 保持原样，避免数据泄漏）。

用法：
    .venv/Scripts/python.exe scripts/21_augment_dataset.py --split train

参数：
    --split             增强哪个 split，默认 train
    --small-area        小目标阈值：bbox 面积占图面积比例 < 该值就算小目标，默认 0.005(0.5%)
    --small-multiplier  含小目标的图额外生成几份，默认 4
    --normal-multiplier 普通图额外生成几份，默认 1（可设 0 只增强小目标图）
    --keep-original     保留原图（默认保留）
    --seed              随机种子
"""
import argparse
import os
import random

import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _load_labels(path):
    """读 YOLO 标签：返回 [(cls, cx, cy, w, h)]（归一化）。"""
    boxes = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            p = line.strip().split()
            if len(p) >= 5:
                boxes.append([float(x) for x in p[:5]])
    return np.array(boxes, dtype=np.float32).reshape(-1, 5)


def _save_labels(path, boxes):
    """写回 YOLO 标签（归一化 cxcywh）。"""
    with open(path, "w", encoding="utf-8") as f:
        for b in boxes:
            cls = int(round(b[0]))
            f.write(f"{cls} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")


def _cxcywh2xyxy(b):
    """(N,5)[cls,cx,cy,w,h] -> (N,4)[x1,y1,x2,y2] 像素坐标。"""
    x1 = b[:, 1] - b[:, 3] / 2
    y1 = b[:, 2] - b[:, 4] / 2
    x2 = b[:, 1] + b[:, 3] / 2
    y2 = b[:, 2] + b[:, 4] / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def _xyxy2cxcywh(cls, xyxy):
    """(N,4)像素xyxy + cls -> (N,5)归一化 cxcywh。"""
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    return np.stack([cls, cx, cy, w, h], axis=1)


def _clamp_boxes(xyxy, W, H, min_side=1e-3):
    """裁剪到图像内，去掉退化框。"""
    xyxy = xyxy.copy()
    xyxy[:, 0] = np.clip(xyxy[:, 0], 0, W)
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0, H)
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0, W)
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0, H)
    keep = (xyxy[:, 2] - xyxy[:, 0] >= min_side) & (xyxy[:, 3] - xyxy[:, 1] >= min_side)
    return xyxy[keep]


def _flip_h(img, boxes):
    """水平翻转。boxes: (N,5) 归一化 cxcywh。"""
    h, w = img.shape[:2]
    out = img[:, ::-1]
    nb = boxes.copy()
    nb[:, 1] = 1 - nb[:, 1]  # cx -> 1 - cx
    return out, nb


def _rotate(img, boxes, angle):
    """绕中心旋转 angle 度，bbox 四角旋转后重算外接框。"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if len(boxes) == 0:
        return out, boxes
    cls = boxes[:, 0]
    xyxy = _cxcywh2xyxy(boxes) * np.array([w, h, w, h])  # 像素
    corners = np.stack([
        xyxy[:, [0, 1]], xyxy[:, [2, 1]], xyxy[:, [2, 3]], xyxy[:, [0, 3]],
    ], axis=1).reshape(-1, 2)  # (4N, 2)
    ones = np.ones((corners.shape[0], 1))
    rot = (np.concatenate([corners, ones], axis=1) @ M.T).reshape(-1, 4, 2)  # (N,4,2)
    x1 = rot[:, :, 0].min(axis=1)
    y1 = rot[:, :, 1].min(axis=1)
    x2 = rot[:, :, 0].max(axis=1)
    y2 = rot[:, :, 1].max(axis=1)
    xyxy_rot = np.stack([x1, y1, x2, y2], axis=1)
    xyxy_rot = _clamp_boxes(xyxy_rot, w, h)
    nb = _xyxy2cxcywh(cls[:len(xyxy_rot)], xyxy_rot)
    nb[:, [1, 3]] /= w
    nb[:, [2, 4]] /= h
    return out, nb


def _scale(img, boxes, s):
    """缩放 s 倍后中心裁剪/填充回原尺寸。"""
    h, w = img.shape[:2]
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if s >= 1:  # 放大 -> 中心裁剪
        x0, y0 = (nw - w) // 2, (nh - h) // 2
        out = resized[y0:y0 + h, x0:x0 + w]
        scale_w = nw / w
        scale_h = nh / h
        if len(boxes):
            xyxy = _cxcywh2xyxy(boxes) * np.array([w, h, w, h])
            xyxy = (xyxy * np.array([scale_w, scale_h, scale_w, scale_h])
                    - np.array([x0, y0, x0, y0]))
            xyxy = _clamp_boxes(xyxy, w, h)
            nb = _xyxy2cxcywh(boxes[:, 0][:len(xyxy)], xyxy)
            nb[:, [1, 3]] /= w
            nb[:, [2, 4]] /= h
        else:
            nb = boxes
    else:  # 缩小 -> 居中贴到灰底
        out = np.full((h, w, 3), 114, dtype=np.uint8)
        x0, y0 = (w - nw) // 2, (h - nh) // 2
        out[y0:y0 + nh, x0:x0 + nw] = resized
        if len(boxes):
            xyxy = _cxcywh2xyxy(boxes) * np.array([w, h, w, h])
            xyxy = xyxy * s + np.array([x0, y0, x0, y0])
            xyxy = _clamp_boxes(xyxy, w, h)
            nb = _xyxy2cxcywh(boxes[:, 0][:len(xyxy)], xyxy)
            nb[:, [1, 3]] /= w
            nb[:, [2, 4]] /= h
        else:
            nb = boxes
    return out, nb


def _brightness_contrast(img, alpha, beta):
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out


def _augment(img, boxes, rng):
    """随机组合若干增强，返回增强后的 (img, boxes)。"""
    if rng.random() < 0.5:
        img, boxes = _flip_h(img, boxes)
    if rng.random() < 0.6:
        img, boxes = _rotate(img, boxes, rng.uniform(-12, 12))
    if rng.random() < 0.6:
        img, boxes = _scale(img, boxes, rng.uniform(0.7, 1.3))
    if rng.random() < 0.7:
        img = _brightness_contrast(img, rng.uniform(0.7, 1.3), rng.uniform(-30, 30))
    return img, boxes


def _has_small(boxes, small_area):
    """是否有 bbox 面积占比 < small_area 的小目标。"""
    for b in boxes:
        if b[3] * b[4] < small_area:
            return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train")
    p.add_argument("--data", default="data")
    p.add_argument("--small-area", type=float, default=0.005, help="小目标面积阈值(占图比例)")
    p.add_argument("--small-multiplier", type=int, default=4)
    p.add_argument("--normal-multiplier", type=int, default=1)
    p.add_argument("--keep-original", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    img_dir = os.path.join(args.data, "images", args.split)
    lbl_dir = os.path.join(args.data, "labels", args.split)
    if not os.path.isdir(img_dir):
        raise SystemExit(f"没找到 {img_dir}")

    rng = random.Random(args.seed)
    n_small = n_normal = 0
    files = sorted(os.listdir(img_dir))
    for fn in files:
        if not fn.lower().endswith(IMG_EXT):
            continue
        stem = os.path.splitext(fn)[0]
        # 跳过已经增强过的副本，避免重复增强
        if "_aug" in stem:
            continue
        img_path = os.path.join(img_dir, fn)
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        img = cv2.imread(img_path)
        if img is None:
            continue
        boxes = _load_labels(lbl_path)

        small = _has_small(boxes, args.small_area)
        k = args.small_multiplier if small else args.normal_multiplier
        if small:
            n_small += 1
        else:
            n_normal += 1

        for i in range(k):
            a_img, a_boxes = _augment(img, boxes, rng)
            out_stem = f"{stem}_aug{i}"
            cv2.imwrite(os.path.join(img_dir, out_stem + ".jpg"), a_img)
            _save_labels(os.path.join(lbl_dir, out_stem + ".txt"), a_boxes)

    print(f"完成：含小目标图 {n_small} 张（×{args.small_multiplier}），"
          f"普通图 {n_normal} 张（×{args.normal_multiplier}）")
    print(f"增强后 train 图片数 ≈ {sum(1 for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT))}")
    print("注意：ultralytics 训练时还有在线 mosaic/HSV 增强，这里是离线补充小缺陷样本。")


if __name__ == "__main__":
    main()
