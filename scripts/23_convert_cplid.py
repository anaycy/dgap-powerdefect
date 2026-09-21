# -*- coding: utf-8 -*-
"""
23_convert_cplid.py —— CPLID 绝缘子数据集 → YOLO 格式并合并进 _raw

CPLID 结构（特殊，不是标准 VOC）：
  Defective_Insulators/images/*.jpg           破损绝缘子图片（248 张）
  Defective_Insulators/labels/defect/*.xml    缺陷区域标注（name="defect"）
  Defective_Insulators/labels/insulator/*.xml 整只绝缘子标注（name="insulator"）
  Normal_Insulators/images/*.jpg              正常绝缘子图片（600 张）
  Normal_Insulators/labels/*.xml              正常绝缘子标注（name="insulator"）

转换规则（缺陷检测任务）：
  - 破损图的 "defect" 标注 → class 0（broken_insulator），忽略 "insulator" 标注
    （缺陷检测只关心"缺陷在哪"，整只绝缘子框只是上下文）。
  - 正常图没有缺陷 → 生成【空标签】txt，作为负样本（让模型学会不把正常绝缘子误报成缺陷）。

输出：data/images/_raw + data/labels/_raw（与其它数据集合并的中间目录）

用法：
    .venv/Scripts/python.exe scripts/23_convert_cplid.py
"""
import argparse
import os
import shutil
import xml.etree.ElementTree as ET


def _parse_box(xml_path):
    """解析 xml，返回 (filename, width, height, [(name, x1, y1, x2, y2)])。"""
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename", "")
    size = root.find("size")
    w = int(float(size.findtext("width", "0")))
    h = int(float(size.findtext("height", "0")))
    objs = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "")
        bnd = obj.find("bndbox")
        x1 = float(bnd.findtext("xmin"))
        y1 = float(bnd.findtext("ymin"))
        x2 = float(bnd.findtext("xmax"))
        y2 = float(bnd.findtext("ymax"))
        objs.append((name, x1, y1, x2, y2))
    return filename, w, h, objs


def _write_yolo(txt_path, objs, w, h, class_map):
    with open(txt_path, "w", encoding="utf-8") as f:
        for name, x1, y1, x2, y2 in objs:
            if name not in class_map:
                continue
            cid = class_map[name]
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--defective", default="raw_datasets/Defective_Insulators")
    p.add_argument("--normal", default="raw_datasets/Normal_Insulators")
    p.add_argument("--out-img", default="data/images/_raw")
    p.add_argument("--out-label", default="data/labels/_raw")
    p.add_argument("--class-name", default="broken_insulator", help="defect 映射到的类别名")
    args = p.parse_args()

    # 统一类别表：defect -> 0（破损绝缘子）
    class_map = {args.class_name: 0}
    # 兼容 CPLID 里可能出现的不同写法
    class_map["defect"] = 0
    class_map["Defect"] = 0

    os.makedirs(args.out_img, exist_ok=True)
    os.makedirs(args.out_label, exist_ok=True)

    n_defect = n_normal = 0

    # 1) 破损绝缘子：用 labels/defect 目录的 xml
    defect_xml_dir = os.path.join(args.defective, "labels", "defect")
    img_dir = os.path.join(args.defective, "images")
    if os.path.isdir(defect_xml_dir):
        for fn in sorted(os.listdir(defect_xml_dir)):
            if not fn.lower().endswith(".xml"):
                continue
            filename, w, h, objs = _parse_box(os.path.join(defect_xml_dir, fn))
            stem = os.path.splitext(os.path.basename(filename))[0]
            # 只保留 defect 类别的框
            defect_objs = [o for o in objs if o[0].lower() == "defect"]
            if not defect_objs:
                continue
            # 找图片
            img = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                cand = os.path.join(img_dir, filename)
                if os.path.exists(cand):
                    img = cand
                    break
                cand = os.path.join(img_dir, stem + ext)
                if os.path.exists(cand):
                    img = cand
                    break
            if img is None:
                continue
            # 防重名：加 cplid_ 前缀
            dst_img = os.path.join(args.out_img, "cplid_" + os.path.basename(img))
            shutil.copy2(img, dst_img)
            _write_yolo(os.path.join(args.out_label, os.path.splitext(os.path.basename(dst_img))[0] + ".txt"),
                        defect_objs, w, h, class_map)
            n_defect += 1

    # 2) 正常绝缘子：生成空标签（负样本）
    normal_img_dir = os.path.join(args.normal, "images")
    if os.path.isdir(normal_img_dir):
        for fn in sorted(os.listdir(normal_img_dir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            dst_img = os.path.join(args.out_img, "cplid_n_" + fn)
            shutil.copy2(os.path.join(normal_img_dir, fn), dst_img)
            # 空标签文件（无缺陷）
            stem = os.path.splitext(fn)[0]
            open(os.path.join(args.out_label, "cplid_n_" + stem + ".txt"), "w", encoding="utf-8").close()
            n_normal += 1

    print(f"完成：破损绝缘子(含缺陷框) {n_defect} 张，正常绝缘子(负样本) {n_normal} 张")
    print(f"已输出到 {args.out_img} / {args.out_label}")
    print("类别映射：defect -> 0 (broken_insulator)；正常图 = 空标签（负样本）")


if __name__ == "__main__":
    main()
