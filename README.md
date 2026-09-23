# 软硬结合·边端自治·闭环诊断的电力缺陷边缘智能系统

> 挑战杯 / 明理杯 · 科技发明制作 B 类 · 信息技术（计算机）
>
> 一句话：软硬结合，把电力缺陷检测模型轻量化后下沉到边缘设备，
> 并形成「采集 → 检测 → 分级 → 预警 → 工单」的闭环诊断，而非单个检测器。

---

## 0. 项目定位（对标全国铜奖「云协智端」）

获奖项目「云协智端」的骨架是「硬件板卡 + 网关 → 边缘算力 → 平台」三层。
本项目复刻并扩展这一骨架，每一层都有独立的技术内容：

| 层 | 云协智端 | 本项目 | 状态 |
|---|---|---|---|
| 硬件层 | 智能采集板卡 + 泛兼容网关 | 自研多源采集板卡 + 树莓派边缘网关 | 软件实做 / 硬件设计见成员2指南 |
| 边缘算力层 | 算力下沉 + 嵌入型检测 | 缺陷感知自适应轻量化（剪枝/蒸馏/量化） | 已有 |
| 平台/应用层 | 行业 AI 模型 + 监测平台 | 闭环诊断终端 + 一键部署平台 | 本项目实做 |
| 跨模态层 | — | 无配对可见光 + 红外决策级融合 | 本项目实做 |

---

## 1. 要解决的四个痛点（对应四个创新点）

1. **边缘算力受限**：高精度检测模型在树莓派等低算力设备上跑不动。
2. **通用压缩"一刀切"**：全局统一剪枝/量化对小目标缺陷（销钉缺失、绝缘子破损）"一视同仁"，压缩后精度暴跌。
3. **跨模态数据缺失**：可见光 + 红外融合需要同一场景的配对图，但配对数据集不存在（纯红外数据集多、配对图没有）。
4. **缺闭环**：现有方案多是"检测器"，检测完就结束，缺"检测 → 诊断 → 决策 → 执行"。

---

## 2. 创新点

1. **缺陷感知自适应轻量化**：分层特征贡献度评估 + 混合粒度剪枝量化联合优化 + 缺陷区域知识蒸馏，保护小目标缺陷通道。
2. **闭环诊断引擎**：检测结果 → 分级（紧急/严重/一般/注意）→ 定位（杆塔号/线路/图内位置）→ 预警 → 工单。
3. **无配对跨模态决策级融合**：可见光与红外各自检测、决策级合并，不依赖不存在的配对数据，互补漏检。
4. **软硬结合**：自研多源采集板卡（可见光/红外测温/温湿度/振动）+ 树莓派边缘网关，边端自治。
5. **可视化一键部署平台**：训练 → 压缩 → 导出 → 评估，网页点按钮走完。

---

## 3. 技术路线

```
电力缺陷数据集 → 高精度模型(教师) → 缺陷感知剪枝 → 缺陷区域蒸馏 → 混合精度量化
   → ONNX 导出 → 边缘网关(采集 + 检测 + 诊断) → 告警 / 工单
```

---

## 4. 关键指标（实测为准，见 `results/实验对比表.md`）

- 参数量压缩 ≥60%、推理提速 ≥2.2 倍、mAP@0.5 下降 ≤3%；
- 边缘端到端时延 / 整机功耗（成员2 在树莓派上实测）。

---

## 5. 目录结构

```
dgap-powerdefect/
├── README.md                     ← 本文件（项目门面）
├── requirements.txt              ← 依赖（含可选的 gradio/mqtt/serial）
├── configs/dgap.yaml             ← 剪枝/蒸馏/量化总配置
├── data/
│   ├── data.yaml                 ← 数据集配置（nc=2，两类）
│   └── images/ labels/           ← 数据（gitignore，按 docs/数据采集方案.md 获取）
├── dgap/                         ← 核心算法包
│   ├── sensitivity.py            ← 缺陷感知敏感性分析（创新点1）
│   ├── pruner.py                 ← 自适应结构化剪枝（创新点1）
│   ├── c2f_v2.py                 ← C2f 重参数化（torch_pruning 兼容）
│   ├── distill.py                ← 缺陷区域知识蒸馏（创新点1）
│   ├── quantize.py               ← 量化 + ONNX/TensorRT 导出（创新点1）
│   ├── diagnosis.py              ← 闭环诊断引擎（创新点2，纯逻辑）
│   ├── fusion.py                 ← 跨模态决策级融合（创新点3，纯逻辑）
│   └── utils.py
├── scripts/                      ← 运行脚本
│   ├── 00_check_env.py           ← 环境自检
│   ├── 01_train_baseline.py      ← Baseline 训练
│   ├── 02_metrics.py             ← 算 Params/FLOPs/mAP/FPS
│   ├── 03_make_table.py          ← 生成实验对比表
│   ├── 10_sensitivity.py         ← 敏感性分析
│   ├── 11_prune_dgap.py          ← 剪枝（普通 vs DGAP）
│   ├── 12_distill.py             ← 蒸馏
│   ├── 13_quantize_export.py     ← 量化导出
│   ├── 20~24_*.py                ← 数据：划分/增强/转换
│   ├── 30_demo_ui.py             ← 闭环诊断演示界面（创新点2）
│   ├── 40_platform.py            ← 一键部署平台（创新点5）
│   ├── 41_gateway.py             ← 边缘网关（创新点4）
│   └── 42_board_simulator.py     ← 采集板卡模拟器（创新点4）
├── tests/                        ← 单元测试（diagnosis / fusion）
├── docs/                         ← 项目文档
│   ├── 分工与计划.md             ← 三人分工 + 时间线
│   ├── 成员2行动指南.md          ← 硬件 + 采集板卡 + 部署
│   ├── 成员3行动指南.md          ← 界面 + 申报书 + PPT
│   ├── 演示界面与申报书模板.md   ← 申报书 B3/C/D 表模板
│   ├── 数据采集方案.md           ← 数据集来源与整理
│   └── ...
├── raw_datasets/                 ← 原始下载数据（gitignore）
├── runs/ results/                ← 训练/结果产物（自动生成，gitignore）
└── yolov8s.pt                    ← 预训练权重（gitignore）
```

---

## 6. 快速开始

### 6.1 装环境

```bash
cd /e/大学/111明理杯/dgap-powerdefect
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
# 自检
.venv/Scripts/python.exe scripts/00_check_env.py
```

### 6.2 备数据 + 训 baseline

数据集：3095 张、2 类（破损绝缘子 + 销钉缺失），来源见 `docs/数据采集方案.md`。

```bash
.venv/Scripts/python.exe scripts/01_train_baseline.py --epochs 100
.venv/Scripts/python.exe scripts/02_metrics.py --weights runs/baseline/yolov8s/weights/best.pt --data data/data.yaml --method "Baseline(YOLOv8s)"
```

### 6.3 跑闭环诊断 demo（无需训练完也能看效果）

```bash
.venv/Scripts/pip.exe install gradio
.venv/Scripts/python.exe scripts/30_demo_ui.py   # 浏览器打开 http://127.0.0.1:7860
```

上传巡检图 → 选模型 → 检出缺陷 + 分级 + 定位 + 运维建议工单。

### 6.4 一键部署平台

```bash
.venv/Scripts/python.exe scripts/40_platform.py  # 网页点按钮：训练→压缩→导出→报告
```

---

## 7. 硬件清单

| 状态 | 硬件 | 用途 |
|---|---|---|
| 已购 | 树莓派 5（8GB） | 边缘网关（本地推理 + 闭环诊断） |
| 已购 | 普通 USB 摄像头 | 可见光采集 |
| 待购 | MLX90640 红外测温模块 | 采集板卡热成像 |
| 待购 | DHT22 温湿度传感器 | 环境监测 |
| 待购 | SW-420 振动传感器 | 杆塔振动监测 |
| 待购 | ESP32-DevKitC | 采集板卡 MCU |

> 待购硬件的接口、接线、固件与打样流程见 `docs/成员2行动指南.md`。

---

## 8. 团队分工

| 成员 | 角色 | 核心产出 |
|---|---|---|
| 成员1 | 算法 | 剪枝/蒸馏/量化 + 消融实验对比表 |
| 成员2 | 部署 + 硬件 | 采集板卡 + 树莓派网关 + 性能测试 |
| 成员3 | 界面 + 材料 | 演示界面 + 申报书/研究报告/PPT |

详见 `docs/分工与计划.md`。

---

## 9. 软件著作权 / 知识产权

一键部署工具链与闭环诊断引擎为自主开发，可申请软件著作权（受理通知书可作为成果支撑）。
