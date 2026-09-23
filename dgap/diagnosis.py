# -*- coding: utf-8 -*-
"""DGAP 闭环诊断引擎：检测结果 -> 分级 / 定位 / 预警 / 工单。

纯逻辑模块：不 import torch/ultralytics，输入输出都是普通数据结构，
可单独拷贝到树莓派（只有 onnxruntime）使用。
"""
from datetime import datetime

DEFAULT_CLASSES = ["broken_insulator", "missing_pin"]

BASE_RISK = {
    "missing_pin": 4,       # 销钉缺失 = 直接危及运行
    "broken_insulator": 3,  # 绝缘子破损 = 按面积分级
}

SUGGESTIONS = {
    "紧急": "立即安排检修，建议 24 小时内处理",
    "严重": "建议 3 日内安排检修",
    "一般": "纳入下轮巡检重点复查",
    "注意": "登记台账，持续观察",
}

LEVEL_COLORS = {  # BGR
    "紧急": (0, 0, 255),
    "严重": (0, 165, 255),
    "一般": (0, 255, 255),
    "注意": (255, 0, 0),
}


def _area_ratio(box, img_shape):
    w = max(box[2] - box[0], 0.0)
    h = max(box[3] - box[1], 0.0)
    return (w * h) / max(img_shape[0] * img_shape[1], 1.0)


def grade_defect(cls_name, area_ratio, conf):
    """按 类别基础风险 + 面积 + 置信度 打等级。

    - 销钉缺失基础分高，默认"严重"；
    - 绝缘子破损靠面积抬分（大面积破损更危险）；
    - 置信度过低降一档，避免误报抬高等级。
    """
    score = BASE_RISK.get(cls_name, 2)
    if area_ratio > 0.02:
        score += 1
    if conf < 0.5:
        score -= 1
    if score >= 5:
        return "紧急"
    if score >= 4:
        return "严重"
    if score >= 3:
        return "一般"
    return "注意"


def locate_defect(meta, box, img_shape):
    """定位：优先用元信息（杆塔号/线路/GPS），否则给图内相对位置。"""
    if meta:
        line = str(meta.get("line", ""))
        tower = str(meta.get("tower", ""))
        gps = meta.get("gps", "")
        if line or tower:
            return (line + tower).strip()
        if gps:
            return str(gps)
    cx = (box[0] + box[2]) / 2 / max(img_shape[1], 1)
    cy = (box[1] + box[3]) / 2 / max(img_shape[0], 1)
    return f"图内相对位置({cx:.2f}, {cy:.2f})"


def build_alert(diagnosis, meta=None):
    """把一条诊断转成结构化告警 JSON。"""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "category": diagnosis["cls"],
        "level": diagnosis["level"],
        "confidence": round(diagnosis["conf"], 4),
        "location": diagnosis["location"],
        "box": [int(v) for v in diagnosis["box"]],
        "suggestion": diagnosis["suggestion"],
        "source": (meta or {}).get("source", "camera"),
    }


def diagnose(detections, meta=None, img_shape=(640, 640), classes=None):
    """主入口。

    detections: list of (cls_id, x1, y1, x2, y2, conf)，像素坐标。
    返回 list[dict]，每项：cls/level/box/conf/area_ratio/location/suggestion。
    """
    classes = classes or DEFAULT_CLASSES
    out = []
    for det in detections:
        cls_id, x1, y1, x2, y2, conf = det
        cls_name = classes[int(cls_id)] if int(cls_id) < len(classes) else f"cls{int(cls_id)}"
        box = (float(x1), float(y1), float(x2), float(y2))
        ar = _area_ratio(box, img_shape)
        level = grade_defect(cls_name, ar, float(conf))
        out.append({
            "cls": cls_name,
            "level": level,
            "box": box,
            "conf": float(conf),
            "area_ratio": round(ar, 5),
            "location": locate_defect(meta, box, img_shape),
            "suggestion": SUGGESTIONS[level],
        })
    return out


def render_report(img, diagnoses):
    """在图上画分级框 + 标签，返回标注后的图（BGR）。"""
    import cv2
    img = img.copy()
    for d in diagnoses:
        color = LEVEL_COLORS[d["level"]]
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{d['cls']} [{d['level']}] {d['conf']:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img
