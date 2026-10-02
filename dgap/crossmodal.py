# -*- coding: utf-8 -*-
"""DGAP 无配对「结构-热」跨模态互补融合。

两个模态检测不同的缺陷（可见光=结构缺陷，红外=热缺陷），天然无需配对数据。
本模块在决策级做互补融合：
  1) 独立双模态诊断（复用 diagnosis.diagnose）
  2) 结构-热空间关联升级：破损绝缘子与发热在同一区域共现 -> 严重度升一档
  3) 传感器驱动的模态可信度自适应加权：光照越差，红外越可信、可见光越不可信

纯逻辑模块：不 import torch/ultralytics，可单独拷贝到边缘设备使用。
"""
from .diagnosis import diagnose, grade_defect, SUGGESTIONS
from .fusion import iou

# 可见光(结构缺陷) / 红外(热缺陷) 默认类别
VIS_CLASSES = ("broken_insulator", "missing_pin")
IR_CLASSES = ("hot",)

# 严重度升一档（"紧急"封顶）
LEVEL_UP = {"注意": "一般", "一般": "严重", "严重": "紧急", "紧急": "紧急"}


def adaptive_modality_weight(condition=None):
    """传感器驱动的模态可信度权重。

    condition 键：
      illumination: 0~1 光照强度（0=全黑，1=白天强光），默认 0.5
      hour: 0~23 小时（没有 illumination 时用它换算光照）
    返回 (w_vis, w_ir)，w 越大该模态越可信。
    """
    c = condition or {}
    illum = c.get("illumination")
    if illum is None:
        h = float(c.get("hour", 12))
        illum = 1.0 if 7 <= h <= 17 else (0.0 if h <= 5 or h >= 19 else 0.5)
    illum = max(0.0, min(1.0, float(illum)))
    w_vis = 0.4 + 0.6 * illum   # 光照对可见光影响大：黑->0.4，亮->1.0
    w_ir = 1.0                  # 红外基本不受光照影响
    return w_vis, w_ir


def structure_thermal_associate(vis_diag, ir_diag, iou_thresh=0.3):
    """结构-热空间关联：破损绝缘子与发热在同一区域共现 -> 严重度升一档。

    原地修改 vis_diag 中命中的破损绝缘子条目（升级 level + 覆盖建议 + 打关联标记），
    返回关联列表 [{structural: {...}, thermal: {...}}, ...]。
    """
    associations = []
    for v in vis_diag:
        if v["cls"] != "broken_insulator":
            continue
        for t in ir_diag:
            if iou(v["box"], t["box"]) >= iou_thresh:
                v["level"] = LEVEL_UP.get(v["level"], v["level"])
                v["suggestion"] = "结构破损叠加局部发热，疑似放电，建议优先检修"
                v["thermal_overlap"] = True
                associations.append({
                    "structural": {"cls": v["cls"], "location": v["location"]},
                    "thermal": {"cls": t["cls"], "location": t["location"]},
                })
                break
    return associations


def fuse_diagnose(vis_dets, ir_dets, condition=None,
                  classes_vis=VIS_CLASSES, classes_ir=IR_CLASSES,
                  img_shape=(640, 640), img_shape_ir=None, meta=None, iou_thresh=0.3):
    """无配对跨模态融合诊断主入口。

    vis_dets / ir_dets: list[(cls_id, x1, y1, x2, y2, conf)]
    img_shape / img_shape_ir: 两模态各自图像尺寸（红外图分辨率可能不同），
      分级用的面积占比按各自尺寸算；img_shape_ir 缺省时回退 img_shape。
    返回 dict: {diagnoses, associations, weights}
    """
    img_shape_ir = img_shape_ir or img_shape
    w_vis, w_ir = adaptive_modality_weight(condition)

    vis_diag = diagnose(vis_dets, meta=meta, img_shape=img_shape, classes=list(classes_vis))
    ir_diag = diagnose(ir_dets, meta=meta, img_shape=img_shape_ir, classes=list(classes_ir))

    # 3) 传感器驱动自适应加权：按模态权重缩放置信度后重打分级
    for d in vis_diag:
        d["conf"] = round(d["conf"] * w_vis, 4)
        d["level"] = grade_defect(d["cls"], d["area_ratio"], d["conf"])
        d["suggestion"] = SUGGESTIONS[d["level"]]
    for d in ir_diag:
        d["conf"] = round(d["conf"] * w_ir, 4)
        d["level"] = grade_defect(d["cls"], d["area_ratio"], d["conf"])
        d["suggestion"] = SUGGESTIONS[d["level"]]

    # 2) 结构-热关联升级
    associations = structure_thermal_associate(vis_diag, ir_diag, iou_thresh)

    return {
        "diagnoses": vis_diag + ir_diag,
        "associations": associations,
        "weights": {"vis": w_vis, "ir": w_ir},
    }
