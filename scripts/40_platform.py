# -*- coding: utf-8 -*-
"""40_platform.py —— 边缘AI轻量化部署平台：一键 训练→压缩→导出→报告。

用 subprocess 调用现有脚本，复用已调通流程（在 PC 上运行，需 GPU）。
运行：.venv/Scripts/python.exe scripts/40_platform.py
"""
import os, subprocess
import gradio as gr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")


def _run(script, *args):
    cmd = [PY, os.path.join(ROOT, "scripts", script), *args]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _fmt(tag, code, out):
    return f"[{'OK' if code == 0 else 'FAIL'}] {tag}\n\n{out[-3000:]}"


def train(epochs):
    c, o = _run("01_train_baseline.py", "--epochs", str(epochs))
    return _fmt("训练 baseline", c, o)


def sensitivity():
    c, o = _run("10_sensitivity.py", "--weights", "runs/baseline/yolov8s/weights/best.pt",
                "--data", "data/data.yaml")
    return _fmt("敏感性分析", c, o)


def prune(mode):
    c, o = _run("11_prune_dgap.py", "--weights", "runs/baseline/yolov8s/weights/best.pt",
                "--data", "data/data.yaml", "--mode", mode, "--ratio", "0.5", "--fine-tune")
    return _fmt(f"剪枝({mode})", c, o)


def distill():
    c, o = _run("12_distill.py", "--student", "runs/finetune/dgap_r0.5/weights/best.pt",
                "--teacher", "runs/baseline/yolov8s/weights/best.pt", "--data", "data/data.yaml")
    return _fmt("蒸馏", c, o)


def quantize():
    c, o = _run("13_quantize_export.py", "--weights", "runs/finetune/dgap_r0.5/weights/best.pt",
                "--data", "data/data.yaml")
    return _fmt("量化导出", c, o)


def report():
    c, o = _run("03_make_table.py")
    md = ""
    p = os.path.join(ROOT, "results", "实验对比表.md")
    if os.path.exists(p):
        md = open(p, encoding="utf-8").read()
    return _fmt("生成报告", c, o) + "\n\n" + md


with gr.Blocks(title="DGAP 部署平台") as app:
    gr.Markdown("# 边缘AI轻量化部署平台（一键 训练→压缩→导出→报告）")
    with gr.Row():
        epochs = gr.Slider(10, 200, 100, step=10, label="训练轮数")
        b_train = gr.Button("1. 训练 baseline")
    with gr.Row():
        b_sens = gr.Button("2. 缺陷敏感性分析")
        b_prune = gr.Button("3. DGAP 自适应剪枝")
        b_uni = gr.Button("3b. 普通剪枝(对照)")
    with gr.Row():
        b_dist = gr.Button("4. 缺陷区域蒸馏")
        b_quant = gr.Button("5. 量化 + ONNX 导出")
    with gr.Row():
        b_rep = gr.Button("6. 生成评估报告")
    log = gr.Textbox(label="运行日志 / 报告", lines=25)

    b_train.click(train, [epochs], log)
    b_sens.click(sensitivity, [], log)
    b_prune.click(lambda: prune("defect"), [], log)
    b_uni.click(lambda: prune("uniform"), [], log)
    b_dist.click(distill, [], log)
    b_quant.click(quantize, [], log)
    b_rep.click(report, [], log)

if __name__ == "__main__":
    app.launch()
