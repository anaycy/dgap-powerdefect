# -*- coding: utf-8 -*-
"""
20_split_dataset.py —— 把数据集按 7:2:1 划分 train / val / test（可分层，保证稀有类不缺席）

两种用法：

1) 重划分项目里现有的 data（把现有 train+val(+test) 合起来重新 7:2:1 分，补出 test 集）：
    .venv/Scripts/python.exe scripts/20_split_dataset.py --stratify

2) 把某个刚下载的平铺数据集（images/ + labels/ 同名）划分进 data：
    .venv/Scripts/python.exe scripts/20_split_dataset.py \
        --src-img /path/to/dataset/images --src-label /path/to/dataset/labels --stratify

参数：
    --ratio    训练:验证:测试 比例，默认 7:2:1
    --stratify 按类别分布分层（推荐，保证"销钉缺失"等稀有类在三个集里都有）
    --seed     随机种子，保证可复现
    --dry-run  只打印计划，不写任何文件（也不会复制图片）
"""
import argparse
import os
import random
import shutil
from collections import Counter

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _collect_from_dir(src_img, src_label):
    """从单个平铺目录收集 [(img_path, label_path, class_signature)]，只保留有同名标签的图片。"""
    pairs = []
    if not os.path.isdir(src_img):
        return pairs
    for fn in sorted(os.listdir(src_img)):
        if not fn.lower().endswith(IMG_EXT):
            continue
        stem = os.path.splitext(fn)[0]
        lbl = os.path.join(src_label, stem + ".txt")
        if not os.path.exists(lbl):
            continue
        classes = set()
        for line in open(lbl, encoding="utf-8"):
            line = line.strip()
            if line:
                classes.add(int(line.split()[0]))
        pairs.append((os.path.join(src_img, fn), lbl, tuple(sorted(classes))))
    return pairs


def _split(pairs, ratio, stratify, seed):
    r = [float(x) for x in ratio.split(":")]
    r = [x / sum(r) for x in r]
    n = len(pairs)

    if stratify:
        from collections import defaultdict
        buckets = defaultdict(list)
        for p in pairs:
            buckets[p[2]].append(p)
        train, val, test = [], [], []
        for items in buckets.values():
            random.Random(seed).shuffle(items)
            k = len(items)
            nt = int(round(k * r[0]))
            nv = int(round(k * r[1]))
            train += items[:nt]
            val += items[nt:nt + nv]
            test += items[nt + nv:]
        for lst in (train, val, test):
            random.Random(seed).shuffle(lst)
    else:
        random.Random(seed).shuffle(pairs)
        n_train = int(round(n * r[0]))
        n_val = int(round(n * r[1]))
        train, val, test = pairs[:n_train], pairs[n_train:n_train + n_val], pairs[n_train + n_val:]

    return train, val, test


def _move(items, dst_img, dst_label):
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_label, exist_ok=True)
    for img, lbl, _ in items:
        shutil.copy2(img, os.path.join(dst_img, os.path.basename(img)))
        shutil.copy2(lbl, os.path.join(dst_label, os.path.basename(lbl)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-img", default=None, help="源图片目录（默认直接读 data/images/{train,val,test}）")
    p.add_argument("--src-label", default=None, help="源标签目录（默认直接读 data/labels/{train,val,test}）")
    p.add_argument("--dst", default="data")
    p.add_argument("--ratio", default="7:2:1")
    p.add_argument("--stratify", action="store_true", help="按类别分层")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    # 收集所有成对样本（默认模式不复制文件，直接读现有目录）
    if args.src_img is None:
        pairs = []
        for split in ("train", "val", "test"):
            pairs += _collect_from_dir(
                os.path.join(args.dst, "images", split),
                os.path.join(args.dst, "labels", split),
            )
    else:
        pairs = _collect_from_dir(args.src_img, args.src_label)

    if not pairs:
        raise SystemExit("没找到成对的图片+标签，检查路径（--src-img / --src-label）")

    cls_count = Counter()
    for _, lbl, _ in pairs:
        for line in open(lbl, encoding="utf-8"):
            line = line.strip()
            if line:
                cls_count[int(line.split()[0])] += 1

    train, val, test = _split(pairs, args.ratio, args.stratify, args.seed)
    print(f"共 {len(pairs)} 张（含标签），类别框数: {dict(cls_count)}")
    print(f"划分 -> train {len(train)} / val {len(val)} / test {len(test)}")

    if args.dry_run:
        print("[dry-run] 未写任何文件。去掉 --dry-run 再跑一次即可真正划分。")
        return

    # 清空目标目录，写入新划分
    for split in ("train", "val", "test"):
        shutil.rmtree(os.path.join(args.dst, "images", split), ignore_errors=True)
        shutil.rmtree(os.path.join(args.dst, "labels", split), ignore_errors=True)
    _move(train, os.path.join(args.dst, "images", "train"), os.path.join(args.dst, "labels", "train"))
    _move(val, os.path.join(args.dst, "images", "val"), os.path.join(args.dst, "labels", "val"))
    _move(test, os.path.join(args.dst, "images", "test"), os.path.join(args.dst, "labels", "test"))

    print("完成。已写入 data/images/{train,val,test} 与 data/labels/{train,val,test}")
    print("提醒：确认 data/data.yaml 已含 test 字段（默认已含）。")


if __name__ == "__main__":
    main()
