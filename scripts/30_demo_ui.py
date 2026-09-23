# -*- coding: utf-8 -*-
"""30_demo_ui.py —— 闭环诊断演示：上传图 -> 检测 -> 分级/定位/预警/工单。

运行：.venv/Scripts/python.exe scripts/30_demo_ui.py
依赖：gradio（可选，.venv/Scripts/pip.exe install gradio）
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gradio as gr
from ultralytics import YOLO

from dgap.diagnosis import diagnose, render_report
import dgap.c2f_v2  # noqa: F401  注册 C2f_v2，供重载剪枝模型

CLASSES = ["broken_insulator", "missing_pin"]

MODELS = {
    "Baseline(YOLOv8s)": "runs/baseline/yolov8s/weights/best.pt",
    "普通剪枝": "runs/finetune/uniform_r0.5/weights/best.pt",
    "DGAP自适应剪枝": "runs/finetune/dgap_r0.5/weights/best.pt",
    "剪枝+蒸馏": "runs/distilled/student_distilled.pt",
}

_cache = {}


def get_model(name):
    if name not in _cache:
        _cache[name] = YOLO(MODELS[name])
    return _cache[name]


def diagnose_image(image, model_name, conf):
    if image is None:
        return None, "请先上传图片"
    model = get_model(model_name)
    t0 = time.time()
    res = model.predict(image, conf=conf, imgsz=640, verbose=False)[0]
    fps = 1 / (time.time() - t0 + 1e-6)

    img = image if isinstance(image, np.ndarray) else np.array(image)
    dets = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cf = res.boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            dets.append((int(cls[i]), *xyxy[i].tolist(), float(cf[i])))

    h, w = img.shape[:2]
    diag = diagnose(dets, meta=None, img_shape=(h, w), classes=CLASSES)
    img_bgr = img[:, :, ::-1].copy()   # RGB -> BGR 供 cv2 画框
    plot_bgr = render_report(img_bgr, diag)
    plot = plot_bgr[:, :, ::-1]        # 回 RGB 供 Gradio

    lines = [f"模型: {model_name}", f"推理: {fps:.1f} FPS", f"检出: {len(diag)} 处缺陷", "---"]
    for d in diag:
        lines.append(f"[{d['level']}] {d['cls']} conf={d['conf']:.2f} @ {d['location']}")
        lines.append(f"   建议: {d['suggestion']}")
    return plot, "\n".join(lines)


with gr.Blocks(title="DGAP 闭环诊断") as demo:
    gr.Markdown("# 电力缺陷边缘智能闭环诊断系统")
    with gr.Row():
        inp = gr.Image(type="numpy", label="上传巡检图")
        with gr.Column():
            model_name = gr.Dropdown(list(MODELS), value=list(MODELS)[0], label="模型")
            conf = gr.Slider(0.05, 0.9, 0.25, label="置信度阈值")
            btn = gr.Button("检测并诊断")
    out_img = gr.Image(type="numpy", label="诊断结果")
    out_txt = gr.Textbox(label="诊断报告 / 工单")
    btn.click(diagnose_image, [inp, model_name, conf], [out_img, out_txt])

if __name__ == "__main__":
    demo.launch()
