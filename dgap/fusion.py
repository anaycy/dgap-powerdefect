# -*- coding: utf-8 -*-
"""DGAP 无配对跨模态决策级融合。

可见光与红外无法获得同一场景的配对图，故不做特征级融合，
改为决策级融合：两模态各自检测，同类且 IoU 重叠的框取高置信度，
互补框都保留，实现漏检互补。
"""


def iou(box_a, box_b):
    """两个 xyxy 框的 IoU。"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_dets(dets_a, dets_b, iou_thresh=0.5):
    """合并两组检测（同格式 (cls_id,x1,y1,x2,y2,conf)）。

    同类且 IoU 超过阈值的框视为同一目标，保留置信度高者；
    不重叠的框全部保留。
    """
    merged = list(dets_a)
    for det in dets_b:
        replaced = False
        for i, m in enumerate(merged):
            if int(m[0]) == int(det[0]) and iou(m[1:5], det[1:5]) >= iou_thresh:
                if det[5] > m[5]:
                    merged[i] = det
                replaced = True
                break
        if not replaced:
            merged.append(det)
    return merged


def modality_weighted_fuse(vis_dets, ir_dets, vis_weight=1.0, ir_weight=1.0,
                           iou_thresh=0.5):
    """两模态加权融合：先把置信度按模态权重缩放，再合并。

    vis_dets/ir_dets: list of (cls_id,x1,y1,x2,y2,conf)。
    """
    vis = [(*d[:5], d[5] * vis_weight) for d in vis_dets]
    ir = [(*d[:5], d[5] * ir_weight) for d in ir_dets]
    return merge_dets(vis, ir, iou_thresh)
