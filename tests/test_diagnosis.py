# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dgap.diagnosis import diagnose, grade_defect, build_alert, render_report


def test_grade_missing_pin():
    # 销钉缺失 base=4，正常置信度 -> 严重
    assert grade_defect("missing_pin", area_ratio=0.001, conf=0.9) == "严重"


def test_grade_insulator_by_area():
    small = grade_defect("broken_insulator", area_ratio=0.001, conf=0.9)  # base3 -> 一般
    big = grade_defect("broken_insulator", area_ratio=0.05, conf=0.9)      # base3+1 -> 严重
    assert big == "严重"
    assert small == "一般"


def test_low_conf_downgrade():
    # conf<0.5 降一档：missing_pin base4 -> score3 -> 一般
    assert grade_defect("missing_pin", area_ratio=0.001, conf=0.3) == "一般"


def test_diagnose_returns_fields():
    dets = [(1, 100, 120, 200, 240, 0.91)]  # cls=1 销钉缺失
    out = diagnose(dets, img_shape=(480, 640))
    assert len(out) == 1
    d = out[0]
    for k in ("cls", "level", "box", "conf", "area_ratio", "location", "suggestion"):
        assert k in d
    assert d["cls"] == "missing_pin"
    assert d["level"] in ("紧急", "严重", "一般", "注意")


def test_locate_uses_meta():
    dets = [(0, 10, 10, 50, 50, 0.8)]
    meta = {"line": "10kV 明珠线", "tower": "T102"}
    d = diagnose(dets, meta=meta, img_shape=(480, 640))[0]
    assert "T102" in d["location"]


def test_locate_fallback_relative():
    dets = [(0, 10, 10, 50, 50, 0.8)]
    d = diagnose(dets, meta=None, img_shape=(480, 640))[0]
    assert "相对位置" in d["location"]


def test_build_alert_json():
    d = diagnose([(1, 1, 1, 9, 9, 0.9)], img_shape=(100, 100))[0]
    a = build_alert(d, meta={"source": "camera"})
    assert a["category"] == "missing_pin"
    assert a["level"] == d["level"]
    assert a["source"] == "camera"


def test_render_report():
    import numpy as np
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    d = diagnose([(1, 10, 10, 40, 40, 0.9)], img_shape=(100, 100))
    out = render_report(img, d)
    assert out.shape == img.shape
