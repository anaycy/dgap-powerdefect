# DGAP —— 面向电力边缘设备的缺陷感知自适应轻量化检测系统

> 挑战杯 / 明理杯 · 科技发明制作 B 类 · 信息技术（计算机）
>
> 一句话：把 YOLO 目标检测模型针对**电力缺陷**做**自适应剪枝 + 缺陷区域蒸馏 + 混合精度量化**，
> 在 mAP 几乎不掉的前提下，参数量压缩 ≥60%、推理提速 ≥2.2 倍，并导出 ONNX 部署到边缘设备。

---

## 0. 项目目录结构

```
dgap-powerdefect/
├── README.md                  ← 本文件（完整手把手教程）
├── 01_创建环境.bat            ← 一键装环境
├── 02_下载数据集.md           ← 数据集来源 + 整理方法
├── requirements.txt
├── configs/dgap.yaml          ← 剪枝/蒸馏/量化总配置
├── data/
│   ├── data.yaml              ← 数据集配置（改类别）
│   └── images/ labels/        ← 数据放这里（自己整理）
├── dgap/                      ← 核心算法包
│   ├── sensitivity.py         ← 缺陷感知敏感性分析（模块A创新点）
│   ├── pruner.py              ← 自适应结构化剪枝（模块A）
│   ├── distill.py             ← 缺陷区域知识蒸馏（模块B）
│   └── quantize.py            ← 量化 + ONNX/TensorRT 导出（模块C）
├── scripts/                   ← 每一步的运行脚本
│   ├── 00_check_env.py        ← 环境自检
│   ├── 01_train_baseline.py   ← Baseline 训练
│   ├── 02_metrics.py          ← 算 Params/FLOPs/mAP/FPS
│   ├── 03_make_table.py       ← 生成实验对比表
│   ├── 10_sensitivity.py      ← 敏感性分析
│   ├── 11_prune_dgap.py       ← 剪枝（普通 vs DGAP）
│   ├── 12_distill.py          ← 蒸馏
│   └── 13_quantize_export.py  ← 量化导出
├── runs/                      ← 训练/剪枝/微调/导出产物（自动生成）
└── results/                   ← 指标 json + 实验表（自动生成）
```

**总路线**（对照申报书 B3 表）：

```
电力缺陷数据集 → 原始高精度模型 → 缺陷敏感性分析 → 自适应剪枝
→ 缺陷区域蒸馏 → 混合精度量化 → ONNX/TensorRT → 边缘部署
```

---

## 1. 阶段一：装环境

**目标**：在 `.venv` 里装好 CUDA 版 PyTorch + ultralytics。

双击运行 `01_创建环境.bat`（或命令行执行）：

```bash
cd /e/大学/111明理杯/dgap-powerdefect
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

**成功标志**（自检）：

```bash
.venv/Scripts/python.exe scripts/00_check_env.py
# 最后一行必须出现：  ===== 环境检查全部通过 =====
# 且能看到：  [OK] CUDA 可用，GPU = NVIDIA GeForce RTX 4060 ...
```

**报错排查**：
- `CUDA 不可用` → 你装的是 CPU 版 torch，删掉 `.venv` 重跑上面的 `--index-url` 那行。
- `torch_pruning 未安装` → `pip install torch_pruning`。
- `ModuleNotFoundError: cv2` → `pip install opencv-python`。

---

## 2. 阶段二：准备数据集

见 **`02_下载数据集.md`**。整理好后目录长这样：

```
data/images/train/*.jpg   data/labels/train/*.txt
data/images/val/*.jpg     data/labels/val/*.txt
```

改好 `data/data.yaml` 里的 `names` 和 `nc`。

---

## 3. 阶段三：训练 Baseline（原始高精度模型）

```bash
.venv/Scripts/python.exe scripts/01_train_baseline.py --model yolov8s --epochs 100 --batch 16
```

- 首次会自动下载 `yolov8s.pt`（约 22MB）。
- **成功标志**：`runs/baseline/yolov8s/weights/best.pt` 和 `last.pt` 出现；
  同时 `runs/baseline/yolov8s/` 里有 `results.png`（loss/mAP 曲线）、`confusion_matrix.png`。
- **报错排查**：
  - `CUDA out of memory` → 把 `--batch` 从 16 降到 8 或 4。
  - `Dataset not found` / `images/train 空` → data.yaml 的 path 写错或图片没放对，见 `data/README.md`。
  - 类别数报错 `nc mismatch` → data.yaml 的 nc 和实际标签的最大 class_id 不符。

---

## 4. 阶段四：算指标（Params / FLOPs / mAP / FPS）

```bash
.venv/Scripts/python.exe scripts/02_metrics.py \
  --weights runs/baseline/yolov8s/weights/best.pt \
  --data data/data.yaml --method "Baseline(YOLOv8s)"
```

**成功标志**：终端打印类似

```json
{
  "method": "Baseline(YOLOv8s)",
  "params_M": 11.13,
  "flops_G": 28.5,
  "map50": 0.8734,
  "map50_95": 0.6122,
  "fps": 156.4,
  "size_MB": 22.5
}
```

- Params（M）：参数量，百万为单位。
- FLOPs（G）：算力消耗，十亿次浮点运算。
- mAP@50：IoU=0.5 的平均精度（缺陷检测常用指标）。
- mAP@50-95：更严格的平均精度。
- FPS：GPU 上 fp16 单张推理帧率。

---

## 5. 阶段五：生成第一张实验表

```bash
.venv/Scripts/python.exe scripts/03_make_table.py
```

**成功标志**：生成 `results/实验对比表.md` 和 `results/实验对比表.csv`，内容像这样：

| 方法 | Params(M)↓ | FLOPs(G)↓ | mAP@50↑ | mAP@50-95↑ | FPS↑ | 大小(MB)↓ | 参数量压缩% | mAP@50下降% |
|---|---|---|---|---|---|---|---|---|
| Baseline(YOLOv8s) | 11.13 | 28.50 | 0.8734 | 0.6122 | 156.4 | 22.50 | — | — |

> 这就是你要的**第一张实验结果表**。之后每做一个方法，跑一次 `02_metrics.py`
> （带不同的 `--method` 名），再跑 `03_make_table.py`，表格就会自动多一行，
> 并自动算出「参数量压缩%」「mAP下降%」。

**到这里，Baseline 阶段全部完成。** 下面进入 DGAP 三个模块。

---

## 6. 阶段六：DGAP 自适应剪枝

### 6.1 缺陷敏感性分析（模块A 的创新点）

```bash
.venv/Scripts/python.exe scripts/10_sensitivity.py \
  --weights runs/baseline/yolov8s/weights/best.pt --data data/data.yaml
```

**成功标志**：生成 `results/sensitivity/defect_sensitivity.pkl`，并打印：
```
缺陷最敏感的 5 个层（剪枝时重点保留）：
  model.2.cv1.conv: 平均敏感度 0.873
  ...
```

### 6.2 自适应剪枝（含微调）

```bash
# DGAP 自适应剪枝
.venv/Scripts/python.exe scripts/11_prune_dgap.py \
  --weights runs/baseline/yolov8s/weights/best.pt --data data/data.yaml \
  --mode defect --ratio 0.5 --fine-tune

# 普通剪枝（对照实验）
.venv/Scripts/python.exe scripts/11_prune_dgap.py \
  --weights runs/baseline/yolov8s/weights/best.pt --data data/data.yaml \
  --mode uniform --ratio 0.5 --fine-tune
```

**成功标志**：
- `runs/pruned/dgap_r0.5.pt`（剪枝后的裸模型）
- `runs/finetune/dgap_r0.5/weights/best.pt`（微调后的最终模型）

剪枝后再各跑一次 `02_metrics.py`（method 名分别写 `"普通剪枝"`、`"DGAP自适应剪枝"`），
再跑 `03_make_table.py`，表格就出现三行了。

> **阶段六 / 七 / 八的代码是完整写好的，但属于“阶段2”，第一次跑可能会遇到
> torch_pruning / ultralytics 版本的细微 API 差异。报错了把完整报错贴给我，我们一起修。**

---

## 7. 阶段七：缺陷区域知识蒸馏（模块B）

```bash
.venv/Scripts/python.exe scripts/12_distill.py \
  --student runs/finetune/dgap_r0.5/weights/best.pt \
  --teacher runs/baseline/yolov8s/weights/best.pt \
  --data data/data.yaml --epochs 60
```

**成功标志**：生成 `runs/distilled/student_distilled.pt`，随后照旧跑 `02_metrics.py`
（method 名 `"自适应剪枝+缺陷蒸馏"`）。

---

## 8. 阶段八：混合精度量化 + 导出（模块C）

```bash
.venv/Scripts/python.exe scripts/13_quantize_export.py \
  --weights runs/finetune/dgap_r0.5/weights/best.pt --data data/data.yaml
```

**成功标志**：`runs/export/` 下出现 `.onnx`（fp32 和 fp16），打印各自大小和 CPU 推理 FPS。
有 Jetson / 装了 TensorRT 的机器，加 `--int8` 再导一个 INT8 引擎。

---

## 9. 完整实验对比表（最终目标）

跑完所有方法后，`results/实验对比表.md` 长这样（示例，数字会因数据集不同而变化）：

| 方法 | Params(M)↓ | FLOPs(G)↓ | mAP@50↑ | mAP@50-95↑ | FPS↑ | 大小(MB)↓ | 参数量压缩% | mAP@50下降% |
|---|---|---|---|---|---|---|---|---|
| Baseline(YOLOv8s) | 11.13 | 28.50 | 0.8734 | 0.6122 | 156.4 | 22.50 | — | — |
| 普通剪枝 | 4.02 | 9.80 | 0.8411 | 0.5703 | 265.9 | 9.10 | 63.9% | 3.23 |
| DGAP自适应剪枝 | 4.15 | 10.05 | 0.8620 | 0.5940 | 258.1 | 9.40 | 62.7% | 1.14 |
| 自适应剪枝+缺陷蒸馏 | 4.15 | 10.05 | 0.8693 | 0.6033 | 258.1 | 9.40 | 62.7% | 0.41 |

**这一张表就是申报书 B3 表的核心证据**：普通剪枝 mAP 掉 3.23（超标），
DGAP 自适应 + 缺陷蒸馏把 mAP 下降压到 0.41（≤3%），同时参数量压缩 ≥60%。

---

## 10. 附：申报书 B3 表填写要点

- **组别**：信息技术；**学科领域**：计算机。
- **作品设计目的/思路/创新点/技术指标**：直接抄 `要求.txt` 的三个创新点 + 关键技术指标。
- **作品可展示形式**：勾选 □实物、产品 □现场演示 □图片 □录像（软件系统 + 树莓派/Jetson 终端）。
- **作品所处阶段**：○实验室阶段。
- **科学性先进性**：和现有“统一剪枝”对比，说明你的“缺陷感知自适应”的实质性进步，附参考文献。
