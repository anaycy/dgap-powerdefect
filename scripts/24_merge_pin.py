# -*- coding: utf-8 -*-
"""
24_merge_pin.py —— 销钉数据集（verified）→ 合并进 _raw

源结构（YOLO 格式，2 类）：
  Pin_datasets/{images,labels}/{train,val}
  Pin_low_quality_dataset/{images,labels}/{train,val}
  类别 names: ['missing', 'normal']   (源 id 0=缺失, 1=正常)

映射规则（缺陷检测，最终 2 类）：
  - missing(源 id 0) -> 我们的 class 1 = missing_pin（销钉缺失）
  - normal(源 id 1)  -> 丢弃（正常销钉 = 负样本，只留空标签）
  最终类别：
    0 = broken_insulator（已有 iraset + CPLID）
    1 = missing_pin（本脚本加入）

用法：
    .venv/Scripts/python.exe scripts/24_merge_pin.py
"""
import argparse
import glob
import os
import shutil

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def _convert_labels(src_label_dir, src_img_dir, out_img, out_label, prefix):
    """把一组 images/labels 转成统一类别并复制到 _raw。返回加入的图片数。"""
    n = 0
    for img in sorted(glob.glob(os.path.join(src_img_dir, "**", "*"), recursive=True)):
        if not img.lower().endswith(IMG_EXT):
            continue
        stem = os.path.splitext(os.path.basename(img))[0]
        # 源标签可能就在同级 labels 目录（按子目录对应）
        rel = os.path.relpath(img, src_img_dir)
        rel_dir = os.path.dirname(rel)
        lbl = os.path.join(src_label_dir, rel_dir, stem + ".txt")

        out_lines = []
        if os.path.exists(lbl):
            for line in open(lbl, encoding="utf-8"):
                p = line.strip().split()
                if len(p) < 5:
                    continue
                cid = int(float(p[0]))
                if cid == 0:  # missing -> 我们的 class 1 (missing_pin)
                    out_lines.append(f"1 {p[1]} {p[2]} {p[3]} {p[4]}\n")
                # cid == 1 (normal) -> 丢弃（负样本）

        dst_stem = f"{prefix}_{stem}"
        shutil.copy2(img, os.path.join(out_img, dst_stem + os.path.splitext(img)[1]))
        with open(os.path.join(out_label, dst_stem + ".txt"), "w", encoding="utf-8") as f:
            f.writelines(out_lines)
        n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="raw_datasets/SADL–YOLOv11_verified dataset")
    p.add_argument("--out-img", default="data/images/_raw")
    p.add_argument("--out-label", default="data/labels/_raw")
    args = p.parse_args()

    os.makedirs(args.out_img, exist_ok=True)
    os.makedirs(args.out_label, exist_ok=True)

    total = 0
    for ds, prefix in [("Pin_datasets", "pin"), ("Pin_low_quality_dataset", "plqd")]:
        img_dir = os.path.join(args.src, ds, "images")
        lbl_dir = os.path.join(args.src, ds, "labels")
        n = _convert_labels(lbl_dir, img_dir, args.out_img, args.out_label, prefix)
        print(f"{ds}: 加入 {n} 张")
        total += n

    # 统计最终 _raw 类别分布
    from collections import Counter
    cnt = Counter()
    for txt in glob.glob(os.path.join(args.out_label, "*.txt")):
        for line in open(txt, encoding="utf-8"):
            line = line.strip()
            if line:
                cnt[int(float(line.split()[0]))] += 1
    n_img = len(glob.glob(os.path.join(args.out_img, "*")))
    print(f"\n完成：共加入 {total} 张销钉图")
    print(f"_raw 现共 {n_img} 张，各类别框数: {dict(cnt)}（0=破损绝缘子, 1=销钉缺失）")


if __name__ == "__main__":
    main()
