# -*- coding: utf-8 -*-
"""
26_convert_ir.py —— OverheatDL 红外数据集 COCO -> YOLO 转换

数据说明（重要）：
  OverheatDL 的图片是连续序列（0000.jpg ~ 1246.jpg），train/test 靠 json 里的
  file_name 划分，物理 zip 文件夹划分对不上——图片实际分散在
  raw_datasets/overheatdl/images/{train,test}/ 两个文件夹里。
  本脚本按 json 的 file_name 去两个文件夹里找图，按 json 的划分产出 YOLO 数据。

COCO 类别（category_id）：
  0=plate(引流板,设备)  1=insulator(绝缘子,设备)  2=point(热斑,缺陷)  3=hot(发热,缺陷)
默认只保留 3=hot(发热) 一类（热斑 point 仅 38 实例太少，设备类非缺陷）。

运行：
    .venv/Scripts/python.exe scripts/26_convert_ir.py
    .venv/Scripts/python.exe scripts/26_convert_ir.py --keep 3,2   # 若坚持要发热+热斑两类
"""
import argparse
import json
import os
import shutil
from collections import Counter

SRC = "raw_datasets/overheatdl"
CATEGORY_NAMES = {0: "plate", 1: "insulator", 2: "point", 3: "hot"}


def build_path_map():
    """file_name -> 磁盘绝对路径（train/test 两个文件夹都要搜）。"""
    m = {}
    for sub in ("train", "test"):
        d = os.path.join(SRC, "images", sub)
        for fn in os.listdir(d):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                m[fn] = os.path.join(d, fn)
    return m


def convert(json_path, keep_ids, path_map, out_img, out_lbl):
    data = json.load(open(json_path, encoding="utf-8"))
    id2class = {cid: i for i, cid in enumerate(keep_ids)}  # COCO id -> YOLO 类下标

    img_meta = {im["id"]: im for im in data["images"]}
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    copied = 0
    skipped_noimg = 0
    skipped_cat = 0
    for im in data["images"]:
        fn = im["file_name"]
        src = path_map.get(fn)
        if src is None:
            skipped_noimg += 1
            continue
        shutil.copy(src, os.path.join(out_img, fn))
        copied += 1

    # 按 image 分组写标签
    anns_by_img = {}
    for a in data["annotations"]:
        if a["category_id"] not in id2class:
            skipped_cat += 1
            continue
        anns_by_img.setdefault(a["image_id"], []).append(a)

    for img_id, anns in anns_by_img.items():
        meta = img_meta[img_id]
        w, h = meta["width"], meta["height"]
        fn = meta["file_name"]
        base = os.path.splitext(fn)[0]
        lines = []
        for a in anns:
            x, y, bw, bh = a["bbox"]  # COCO 是像素 x,y,w,h
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{id2class[a['category_id']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        with open(os.path.join(out_lbl, base + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    return copied, skipped_noimg, skipped_cat, Counter(
        a["category_id"] for a in data["annotations"] if a["category_id"] in id2class)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keep", default="3", help="保留的 COCO category_id，逗号分隔（默认 3=hot）")
    p.add_argument("--out", default="data_ir", help="输出根目录")
    args = p.parse_args()

    keep_ids = [int(x) for x in args.keep.split(",")]
    names = {i: CATEGORY_NAMES[cid] for i, cid in enumerate(keep_ids)}
    path_map = build_path_map()
    print(f"磁盘红外图总数: {len(path_map)}")
    print(f"保留类别: {names}")

    summary = {}
    for split, json_name in (("train", "train.json"), ("val", "test.json")):
        out_img = os.path.join(args.out, "images", split)
        out_lbl = os.path.join(args.out, "labels", split)
        c, noimg, nocat, dist = convert(
            os.path.join(SRC, "annotations", json_name),
            keep_ids, path_map, out_img, out_lbl)
        summary[split] = (c, noimg, nocat, dist)
        print(f"[{split}] 复制图 {c} 张，缺图 {noimg}，跳过非目标类标注 {nocat}，类别分布 {dict(dist)}")

    # 生成 data_ir.yaml
    yaml_path = os.path.join(args.out, "data_ir.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"# 红外数据集（OverheatDL 转换，仅热缺陷）\n")
        f.write(f"path: {os.path.abspath(args.out)}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val   # 用 test.json 当验证集\n")
        f.write(f"names:\n")
        for i, name in names.items():
            f.write(f"  {i}: {name}\n")
        f.write(f"nc: {len(names)}\n")
    print(f"\n完成。数据 -> {args.out}/，配置 -> {yaml_path}")


if __name__ == "__main__":
    main()
