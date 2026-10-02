# -*- coding: utf-8 -*-
"""
27_fusion_demo.py —— 无配对「结构-热」跨模态融合演示（CLI）

输入：可见光图 + 红外图 + 场景条件（光照/时间）
流程：可见光模型(结构缺陷) + 红外模型(热缺陷) -> fuse_diagnose 融合诊断

运行：
    .venv/Scripts/python.exe scripts/27_fusion_demo.py \
        --vis data/images/test/cplid_013.jpg \
        --ir data_ir/images/val/0213.jpg \
        --illumination 0.3

说明：
  「结构-热共现升级」只在同一区域同时被双模态拍到时才触发（需采集板卡实拍对齐），
  公开数据无配对图，故共现逻辑已由 tests/test_crossmodal.py 覆盖；
  本脚本演示独立双模态检测 + 传感器驱动自适应加权。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from ultralytics import YOLO

import dgap.c2f_v2  # noqa: F401  重载压缩后的可见光模型需要
from dgap.diagnosis import render_report
from dgap.crossmodal import fuse_diagnose

VIS_MODEL = "runs/finetune/dgap_r0.5/weights/best.pt"
IR_MODEL = "runs/finetune/dgap_ir_r0.5/weights/best.pt"
VIS_CLASSES = ("broken_insulator", "missing_pin")
IR_CLASSES = ("hot",)


def detect(model, img):
    res = model.predict(img, conf=0.25, imgsz=640, verbose=False)[0]
    dets = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cf = res.boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            dets.append((int(cls[i]), *xyxy[i].tolist(), float(cf[i])))
    return dets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vis", required=True, help="可见光图路径")
    p.add_argument("--ir", required=True, help="红外图路径")
    p.add_argument("--illumination", type=float, default=None, help="光照 0~1")
    p.add_argument("--hour", type=float, default=None, help="时间 0~23")
    args = p.parse_args()

    vis_img = cv2.imread(args.vis)
    ir_img = cv2.imread(args.ir)
    if vis_img is None or ir_img is None:
        raise SystemExit("图片读取失败，检查路径")

    vis_model = YOLO(VIS_MODEL)
    ir_model = YOLO(IR_MODEL)

    vis_dets = detect(vis_model, vis_img)
    ir_dets = detect(ir_model, ir_img)

    condition = {}
    if args.illumination is not None:
        condition["illumination"] = args.illumination
    if args.hour is not None:
        condition["hour"] = args.hour

    out = fuse_diagnose(vis_dets, ir_dets, condition=condition,
                        classes_vis=VIS_CLASSES, classes_ir=IR_CLASSES,
                        img_shape=vis_img.shape[:2], img_shape_ir=ir_img.shape[:2])

    print(f"可见光检出 {len(vis_dets)} 处，红外检出 {len(ir_dets)} 处")
    print(f"模态权重: 可见光 {out['weights']['vis']:.2f} / 红外 {out['weights']['ir']:.2f}")
    print(f"结构-热关联: {len(out['associations'])} 处")
    print("--- 融合诊断 ---")
    for d in out["diagnoses"]:
        flag = "  [结构+热关联]" if d.get("thermal_overlap") else ""
        print(f"[{d['level']}] {d['cls']} conf={d['conf']:.2f} @ {d['location']}{flag}")
        print(f"    建议: {d['suggestion']}")

    os.makedirs("runs", exist_ok=True)
    vis_annot = render_report(vis_img, [d for d in out["diagnoses"] if d["cls"] in VIS_CLASSES])
    cv2.imwrite("runs/fusion_demo_vis.jpg", vis_annot)
    ir_annot = render_report(ir_img, [d for d in out["diagnoses"] if d["cls"] in IR_CLASSES])
    cv2.imwrite("runs/fusion_demo_ir.jpg", ir_annot)
    print("\n标注图已存 runs/fusion_demo_vis.jpg 和 runs/fusion_demo_ir.jpg")


if __name__ == "__main__":
    main()
