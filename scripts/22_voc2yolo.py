# -*- coding: utf-8 -*-
"""
22_voc2yolo.py —— Pascal VOC(xml) 标注 → YOLO(txt) 标注转换

很多公开电力缺陷数据集（腾讯云/CSDN 的"销钉缺失""绝缘子破损"等）给的是
VOC 格式 xml 标注。本脚本把它们转成 YOLO 归一化 cxcywh txt，并（可选）把
图片拷到统一目录，方便合并进 data/images + data/labels。

用法：
    .venv/Scripts/python.exe scripts/22_voc2yolo.py \
        --xml-dir /path/to/Annotations --img-dir /path/to/JPEGImages \
        --classes "DefectPin,NormalPin,DefectInsulator,NormalInsulator"

参数：
    --xml-dir    存放 .xml 的目录
    --img-dir    存放图片的目录（可选；给了就顺带把有标注的图片拷过去）
    --out-img    输出图片目录，默认 data/images/_raw
    --out-label  输出标签目录，默认 data/labels/_raw
    --classes    类别名（逗号分隔，顺序即 class_id）。不填则自动按字母序收集，
                 但强烈建议显式指定，保证合并多个数据集时 id 一致。
"""
import argparse
import os
import shutil
import xml.etree.ElementTree as ET

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_xml(path, class_map):
    """解析单个 VOC xml，返回 (filename, [(cls_id, x1, y1, x2, y2) 像素])。"""
    tree = ET.parse(path)
    root = tree.getroot()
    filename = root.findtext("filename", "")
    size = root.find("size")
    w = int(float(size.findtext("width", "0")))
    h = int(float(size.findtext("height", "0")))
    objs = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "")
        if name not in class_map:
            continue  # 跳过未知类别
        bnd = obj.find("bndbox")
        x1 = float(bnd.findtext("xmin"))
        y1 = float(bnd.findtext("ymin"))
        x2 = float(bnd.findtext("xmax"))
        y2 = float(bnd.findtext("ymax"))
        objs.append((class_map[name], x1, y1, x2, y2))
    return filename, w, h, objs


def to_yolo_line(cls_id, x1, y1, x2, y2, w, h):
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml-dir", required=True)
    p.add_argument("--img-dir", default=None)
    p.add_argument("--out-img", default="data/images/_raw")
    p.add_argument("--out-label", default="data/labels/_raw")
    p.add_argument("--classes", default=None, help="逗号分隔类别名，顺序即 id")
    args = p.parse_args()

    xmls = [f for f in os.listdir(args.xml_dir) if f.lower().endswith(".xml")]
    if not xmls:
        raise SystemExit(f"{args.xml_dir} 里没找到 xml")

    # 类别映射
    if args.classes:
        class_map = {name: i for i, name in enumerate(args.classes.split(","))}
    else:
        names = set()
        for f in xmls:
            root = ET.parse(os.path.join(args.xml_dir, f)).getroot()
            for obj in root.findall("object"):
                names.add(obj.findtext("name", ""))
        class_map = {name: i for i, name in enumerate(sorted(names))}
        print("未指定 --classes，自动按字母序生成类别映射：")
        for name, i in class_map.items():
            print(f"  {i}: {name}")
        print("（强烈建议合并多个数据集时显式指定 --classes 以统一 id）")

    os.makedirs(args.out_label, exist_ok=True)
    if args.img_dir:
        os.makedirs(args.out_img, exist_ok=True)

    done = 0
    for f in sorted(xmls):
        filename, w, h, objs = parse_xml(os.path.join(args.xml_dir, f), class_map)
        if not objs or w == 0 or h == 0:
            continue
        stem = os.path.splitext(filename)[0]
        # 写 YOLO 标签
        with open(os.path.join(args.out_label, stem + ".txt"), "w", encoding="utf-8") as fo:
            for cls_id, x1, y1, x2, y2 in objs:
                fo.write(to_yolo_line(cls_id, x1, y1, x2, y2, w, h) + "\n")
        # 拷贝图片
        if args.img_dir:
            src = None
            for ext in IMG_EXT:
                cand = os.path.join(args.img_dir, filename)
                if os.path.exists(cand):
                    src = cand
                    break
                cand = os.path.join(args.img_dir, stem + ext)
                if os.path.exists(cand):
                    src = cand
                    break
            if src:
                shutil.copy2(src, os.path.join(args.out_img, os.path.basename(src)))
        done += 1

    print(f"转换完成：{done} 张。类别映射 = {class_map}")
    print("下一步：把 _raw 里的图片/标签并进 data/images|labels/train，再跑 20_split_dataset.py 划分，")
    print("并更新 data/data.yaml 的 names/nc 与此处类别一致。")


if __name__ == "__main__":
    main()
