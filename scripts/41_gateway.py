# -*- coding: utf-8 -*-
"""41_gateway.py —— 树莓派边缘网关：采集(摄像头+板卡) -> 检测 -> 闭环诊断 -> 上报。

--dry-run：不连摄像头/MQTT，伪造一条检测 + 一条板卡数据走通全链路（PC 可跑）。
完整采集→检测→诊断→上报的接线与循环见 docs/成员2行动指南.md。
运行：.venv/Scripts/python.exe scripts/41_gateway.py --dry-run
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dgap.diagnosis import diagnose, build_alert

CLASSES = ["broken_insulator", "missing_pin"]


def infer(session, img, imgsz=640):
    """letterbox -> ONNX -> 返回 dets [(cls_id,x1,y1,x2,y2,conf)]。"""
    import cv2
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    dw, dh = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = blob[None]
    out = session.run(None, {session.get_inputs()[0].name: blob})[0]
    pred = out[0].transpose()          # (8400, 4+nc)
    boxes, scores = pred[:, :4], pred[:, 4:]
    dets = []
    for c in range(len(CLASSES)):
        for i in np.where(scores[:, c] > 0.25)[0]:
            x1, y1, x2, y2 = boxes[i]
            x1 = (x1 - dw) / r; y1 = (y1 - dh) / r
            x2 = (x2 - dw) / r; y2 = (y2 - dh) / r
            dets.append((c, float(x1), float(y1), float(x2), float(y2), float(scores[i, c])))
    return dets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/export/best_fp32.onnx")
    p.add_argument("--dry-run", action="store_true", help="不连硬件，跑一条样例")
    args = p.parse_args()

    if args.dry_run:
        dets = [(1, 100, 120, 200, 240, 0.91)]  # cls=1 销钉缺失
        meta = {"line": "10kV 明珠线", "tower": "T102", "source": "camera"}
        diag = diagnose(dets, meta=meta, img_shape=(480, 640), classes=CLASSES)
        for d in diag:
            print(json.dumps(build_alert(d, meta), ensure_ascii=False))
        print("[网关] dry-run 走通：检测 -> 诊断 -> 告警")
        return

    print("[网关] 完整采集->检测->诊断->上报 流程见 docs/成员2行动指南.md（含摄像头/板卡接线）")


if __name__ == "__main__":
    main()
