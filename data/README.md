# data/ 目录说明 —— 数据集如何整理

YOLOv8 要求的标准目录结构如下（图片和标签**分开放**，同名对应）：

```
data/
├── data.yaml              ← 数据集配置（改类别名）
├── images/
│   ├── train/  xxx.jpg    ← 训练图片
│   ├── val/    yyy.jpg    ← 验证图片
│   └── test/   zzz.jpg    ← 测试图片（可选）
└── labels/
    ├── train/  xxx.txt    ← 训练标签（和图片同名）
    ├── val/    yyy.txt
    └── test/   zzz.txt
```

## 标签文件格式（每张图一个 .txt，同名）

每一行一个目标，格式为：
```
class_id  x_center  y_center  width  height
```
其中 x_center / y_center / width / height 都是**归一化到 0~1** 的小数。

例：`0 0.5123 0.4321 0.2340 0.1560` 表示第 0 类（破损绝缘子）中心在 (0.51, 0.43)、宽 0.234、高 0.156。

## 你只需要做三步

1. 下载数据集（见 `docs/数据采集方案.md`）。
2. 把图片塞进 `images/train|val|test`，把同名的 `.txt` 标签塞进 `labels/train|val|test`。
3. 改 `data.yaml` 里的 `names` 和 `nc`，让它们和数据集的实际类别一致。

## 常见坑

- 图片和标签**必须同名**（`a.jpg` ↔ `a.txt`）。
- 标签框坐标必须**归一化**（0~1），不是像素坐标。Roboflow 导出的 YOLO 格式已经归一化，直接用即可。
- 没有 test 集时，把 `data.yaml` 里的 `test:` 那一行删掉，指标脚本会自动回退到 val。
