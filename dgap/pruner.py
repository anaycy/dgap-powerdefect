# -*- coding: utf-8 -*-
"""
pruner.py —— 结构化剪枝引擎（模块A，基于 torch_pruning）

两个入口：
  prune_uniform(model, ratio)          普通全局剪枝（对照实验：L2 范数，统一比例）
  prune_dgap(model, sensitivity, ...)  DGAP 缺陷感知自适应剪枝（模块A 创新点）

原理：
  torch_pruning 自动解析 YOLOv8 的依赖图（残差 / concat / Detect 头），
  保证剪掉通道后网络仍能前向跑通。我们只负责：
    1) 每个通道的重要性打分（缺陷感知）；
    2) 每个层的自适应剪枝比例（缺陷敏感层少剪、冗余层多剪）。

关键正确性约定（torch_pruning 1.6.0）：
  - MetaPruner 的 pruning_ratio_dict 键必须是 nn.Module（不是字符串名），
    get_target_pruning_ratio(module) 是按 module 对象查表的；
  - importance 回调收到的是 Group 对象，取根模块用 group[0].dep.target.module，
    取通道下标用 group[0].idxs（_HybridIndex，需取 .idx）；
  - example_inputs 必须与模型在同一设备（模型在 CUDA 时不能再在 CPU 建 tensor）。
"""
import numpy as np
import torch
import torch.nn as nn

from .c2f_v2 import replace_c2f_with_c2f_v2


def _prepare_model(model):
    """剪枝前的模型预处理：C2f -> C2f_v2 重参数化 + 初始化 BN + 打开梯度。

    - C2f_v2 把 `cv1(x).chunk(2)` 拆成 cv0/cv1 两个显式卷积，是 torch_pruning
      能正确解析依赖图的必要前提（见 dgap/c2f_v2.py）。
    - YOLO(...) 加载的模型参数 requires_grad=False，torch_pruning 依赖前向
      的 grad_fn 做图 tracing，因此这里统一打开 requires_grad。
    """
    from ultralytics.utils.torch_utils import initialize_weights

    replace_c2f_with_c2f_v2(model)
    initialize_weights(model)          # 新 cv0/cv1 的 BN 设为 eps=1e-3 / momentum=0.03
    for p in model.parameters():
        p.requires_grad = True
    return model


def _meta_prune(model, example_inputs, importance, ratio, ratio_dict=None, ignored_layers=None):
    """调用 torch_pruning 的 MetaPruner 做一次性结构化剪枝。"""
    import torch_pruning as tp

    model.eval()
    pruner = tp.pruner.MetaPruner(
        model=model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=ratio,
        pruning_ratio_dict=ratio_dict,
        iterative_steps=1,
        ignored_layers=ignored_layers or [],
    )
    pruner.step()
    return model


def _example_inputs(model, imgsz=640):
    """构造与模型同设备的示例输入（torch_pruning 用它 trace 依赖图）。"""
    device = next(model.parameters()).device
    return torch.rand(1, 3, imgsz, imgsz, device=device)


def prune_uniform(model, ratio=0.5, ignored_layers=None):
    """普通全局剪枝：Magnitude(L2) 重要性，所有层统一比例（对照实验）。"""
    import torch_pruning as tp

    _prepare_model(model)
    example_inputs = _example_inputs(model)
    importance = tp.importance.MagnitudeImportance(p=2)
    return _meta_prune(model, example_inputs, importance, ratio, None, ignored_layers)


def prune_dgap(model, sensitivity, global_ratio=0.5, beta=0.6,
               ratio_min=0.1, ratio_max=0.8, ignored_layers=None):
    """
    DGAP 缺陷感知自适应剪枝：
      - 层级自适应：某层缺陷敏感度越高，剪枝比例越小（ratio_dict，键为模块对象）。
      - 通道级自适应：某通道缺陷敏感度越高，重要性越大（越保留）。
    """
    _prepare_model(model)
    example_inputs = _example_inputs(model)

    # 模块对象 -> 名字 的反查表（torch_pruning 的 group 只给 module）
    name_of = {m: n for n, m in model.named_modules()}

    # 1) 层级自适应比例：某层缺陷敏感度越高，剪枝比例越小。
    #    以全体卷积层的平均敏感度 imp_mean 为中性点：高于均值的敏感层少剪
    #    （比例 < global_ratio），低于均值的冗余层多剪（比例 > global_ratio），
    #    保证整体压缩率仍围绕 global_ratio，同时让"敏感层低剪枝"名副其实。
    name_imp = {}
    matched = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            s = sensitivity.get(name)
            if s is not None:
                name_imp[name] = float(np.clip(s.mean(), 0.0, 1.0))
                matched += 1
            else:
                name_imp[name] = 0.5
    imp_mean = float(np.mean(list(name_imp.values()))) if name_imp else 0.5
    print(f"[DGAP] 敏感性匹配 {matched}/{len(name_imp)} 个卷积层"
          f"（未匹配的按 0.5 处理），平均敏感度 {imp_mean:.3f}")
    if matched == 0:
        print("[DGAP] 警告：没有任何层匹配到敏感性分析结果，DGAP 将退化为普通剪枝！"
              "请先跑 scripts/10_sensitivity.py，并确认层名一致。")

    ratio_dict = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            imp = name_imp[name]
            r = global_ratio * (1.0 + beta * (imp_mean - imp))
            ratio_dict[m] = max(ratio_min, min(ratio_max, r))

    # 2) 通道级缺陷感知重要性：敏感度越高越保留
    def importance(group):
        module = group[0].dep.target.module          # 剪枝组的根模块
        name = name_of.get(module)
        idxs = group[0].idxs                          # _HybridIndex 列表
        plain = [int(i.idx) if hasattr(i, "idx") else int(i) for i in idxs]
        n = len(plain)
        if name is not None and name in sensitivity:
            s = sensitivity[name]
            scores = torch.tensor(
                [float(s[i]) if i < len(s) else 0.0 for i in plain],
                dtype=torch.float32,
            )
            # 加微小扰动，避免全等导致剪枝器报错
            return scores + 1e-6 * torch.rand(n)
        return torch.ones(n, dtype=torch.float32)

    return _meta_prune(model, example_inputs, importance, global_ratio,
                       ratio_dict, ignored_layers)
