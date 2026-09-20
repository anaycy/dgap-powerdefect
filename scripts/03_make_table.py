# -*- coding: utf-8 -*-
"""
03_make_table.py —— 生成实验结果对比表（Markdown + CSV）
数据源：results/experiment_results.json（由 02_metrics.py 和各剪枝/蒸馏脚本写入）
运行：
    .venv/Scripts/python.exe scripts/03_make_table.py
输出：
    results/实验对比表.md
    results/实验对比表.csv
第一行永远是 Baseline，其余行自动计算「参数量压缩%」和「mAP下降%」。
"""
import csv
import json
import os


def render():
    path = "results/experiment_results.json"
    if not os.path.exists(path):
        print("还没结果：请先运行 02_metrics.py 生成 results/experiment_results.json")
        return

    data = json.load(open(path, encoding="utf-8"))
    methods = data.get("methods", [])
    if not methods:
        print("experiment_results.json 里还没有任何方法，先跑 02_metrics.py")
        return

    base = methods[0]  # 第一行是 Baseline
    cols = ["方法", "Params(M)↓", "FLOPs(G)↓", "mAP@50↑", "mAP@50-95↑",
            "FPS↑", "大小(MB)↓", "参数量压缩%", "mAP@50下降%"]

    rows = []
    for r in methods:
        param_compress = (1 - r["params_M"] / base["params_M"]) * 100 if base["params_M"] else 0
        map_drop = (base["map50"] - r["map50"]) * 100
        rows.append([
            r["method"],
            f'{r["params_M"]:.2f}',
            f'{r["flops_G"]:.2f}',
            f'{r["map50"]:.4f}',
            f'{r["map50_95"]:.4f}',
            f'{r["fps"]:.1f}',
            f'{r["size_MB"]:.2f}',
            f'{param_compress:.1f}%' if r is not base else "—",
            f'{map_drop:.2f}' if r is not base else "—",
        ])

    # Markdown
    md = ["# 实验结果对比表", "",
          f"数据集：`{data.get('dataset','')}`", "",
          "| " + " | ".join(cols) + " |",
          "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    md.append("> 参数量压缩% = (1 - 当前Params / Baseline Params) × 100；"
              "mAP@50下降% = Baseline mAP@50 - 当前 mAP@50（单位是百分点）。")
    md.append("> 关键指标目标（申报书）：参数量压缩 ≥60%，FPS 提升 ≥2.2 倍，mAP 下降 ≤3%。")

    os.makedirs("results", exist_ok=True)
    with open("results/实验对比表.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # CSV（方便贴进 Excel）
    with open("results/实验对比表.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)

    print("\n".join(md))


if __name__ == "__main__":
    render()
