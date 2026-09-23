# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dgap.fusion import iou, merge_dets, modality_weighted_fuse


def test_iou_identical():
    assert abs(iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-6


def test_iou_disjoint():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_merge_keeps_higher_conf():
    a = [(1, 0, 0, 10, 10, 0.6)]
    b = [(1, 0, 0, 10, 10, 0.9)]
    out = merge_dets(a, b, iou_thresh=0.5)
    assert len(out) == 1 and out[0][5] == 0.9


def test_merge_keeps_complement():
    a = [(1, 0, 0, 10, 10, 0.8)]
    b = [(1, 50, 50, 60, 60, 0.7)]
    out = merge_dets(a, b, iou_thresh=0.5)
    assert len(out) == 2


def test_fuse_scales_confidence():
    vis = [(1, 0, 0, 10, 10, 0.8)]
    ir = [(1, 0, 0, 10, 10, 0.5)]
    out = modality_weighted_fuse(vis, ir, vis_weight=1.0, ir_weight=0.7, iou_thresh=0.5)
    assert len(out) == 1 and abs(out[0][5] - 0.8) < 1e-6
