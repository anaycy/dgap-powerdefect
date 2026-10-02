# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dgap.diagnosis import diagnose
from dgap.crossmodal import (
    adaptive_modality_weight,
    structure_thermal_associate,
    fuse_diagnose,
)


def test_adaptive_weight_night_ir_more_reliable():
    w_vis, w_ir = adaptive_modality_weight({"illumination": 0.0})
    assert w_ir > w_vis


def test_adaptive_weight_day_visible_ok():
    w_vis, w_ir = adaptive_modality_weight({"illumination": 1.0})
    assert w_vis >= w_ir


def test_adaptive_weight_hour_night():
    w_vis, w_ir = adaptive_modality_weight({"hour": 2})
    assert w_ir > w_vis


def test_associate_escalates_overlap():
    vis_diag = diagnose([(0, 0, 0, 100, 100, 0.9)], img_shape=(200, 200),
                        classes=["broken_insulator"])
    ir_diag = diagnose([(0, 10, 10, 110, 110, 0.8)], img_shape=(200, 200), classes=["hot"])
    assert vis_diag[0]["level"] == "严重"
    assoc = structure_thermal_associate(vis_diag, ir_diag, iou_thresh=0.3)
    assert len(assoc) == 1
    assert vis_diag[0]["level"] == "紧急"          # 升级了
    assert vis_diag[0]["thermal_overlap"] is True
    assert "叠加局部发热" in vis_diag[0]["suggestion"]


def test_associate_no_overlap():
    vis_diag = diagnose([(0, 0, 0, 10, 10, 0.9)], img_shape=(200, 200),
                        classes=["broken_insulator"])
    ir_diag = diagnose([(0, 100, 100, 110, 110, 0.8)], img_shape=(200, 200), classes=["hot"])
    assoc = structure_thermal_associate(vis_diag, ir_diag, iou_thresh=0.3)
    assert len(assoc) == 0
    assert "thermal_overlap" not in vis_diag[0]


def test_associate_ignores_non_insulator():
    # 销钉缺失即使和 hot 重叠也不升级（只升级破损绝缘子）
    vis_diag = diagnose([(1, 0, 0, 100, 100, 0.9)], img_shape=(200, 200),
                        classes=["broken_insulator", "missing_pin"])
    ir_diag = diagnose([(0, 10, 10, 110, 110, 0.8)], img_shape=(200, 200), classes=["hot"])
    assoc = structure_thermal_associate(vis_diag, ir_diag, iou_thresh=0.3)
    assert len(assoc) == 0


def test_fuse_diagnose_combines_and_escalates():
    out = fuse_diagnose(
        vis_dets=[(0, 0, 0, 100, 100, 0.9), (1, 150, 150, 160, 160, 0.7)],
        ir_dets=[(0, 10, 10, 110, 110, 0.8)],
        condition={"illumination": 1.0},
        img_shape=(200, 200),
    )
    assert len(out["diagnoses"]) == 3          # 2 可见光 + 1 红外
    assert len(out["associations"]) == 1       # 破损绝缘子 + hot 重叠
    assert out["weights"]["vis"] >= out["weights"]["ir"]
    # 找到被升级的破损绝缘子
    ins = [d for d in out["diagnoses"] if d["cls"] == "broken_insulator"][0]
    assert ins["level"] == "紧急"
    assert ins["thermal_overlap"] is True
