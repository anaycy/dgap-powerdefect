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
"""
import numpy as np
import torch
import torch.nn as nn


def _meta_prune(model, example_inputs, importance, ratio, ratio_dict=None, ignored_layers=None):
    """调用 torch_pruning 的 MetaPruner，兼容新旧版本参数名。"""
    import torch_pruning as tp

    model.eval()
    tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

    kwargs = dict(
        model=model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=1,
        ignored_layers=ignored_layers or [],
    )
    try:
        # 新版本：pruning_ratio / pruning_ratio_dict（支持每层自适应比例）
        kwargs["pruning_ratio"] = ratio
        if ratio_dict:
            kwargs["pruning_ratio_dict"] = ratio_dict
        pruner = tp.pruner.MetaPruner(**kwargs)
    except TypeError:
        # 旧版本：channel_sparsity
        kwargs.pop("pruning_ratio", None)
        kwargs.pop("pruning_ratio_dict", None)
        kwargs["channel_sparsity"] = ratio
        pruner = tp.pruner.MetaPruner(**kwargs)

    pruner.step()
    return model


def prune_uniform(model, ratio=0.5, ignored_layers=None):
    """普通全局剪枝：Magnitude(L2) 重要性，所有层统一比例。"""
    import torch_pruning as tp
    example_inputs = torch.randn(1, 3, 640, 640)
    importance = tp.importance.MagnitudeImportance(p=2)
    return _meta_prune(model, example_inputs, importance, ratio, None, ignored_layers)


def prune_dgap(model, sensitivity, global_ratio=0.5, beta=0.6,
               ratio_min=0.1, ratio_max=0.8, ignored_layers=None):
    """
    DGAP 缺陷感知自适应剪枝：
      - 层级自适应：某层缺陷敏感度越高，剪枝比例越小（ratio_dict）。
      - 通道级自适应：某通道缺陷敏感度越高，重要性越大（越保留）。
    """
    import torch_pruning as tp
    example_inputs = torch.randn(1, 3, 640, 640)

    # 模块对象 -> 名字 的反查表（torch_pruning 的 group 只给 module）
    name_of = {m: n for n, m in model.named_modules()}

    # 1) 层级自适应比例：ratio_i = clamp(global * (1 + beta*(1 - importance_i)))
    ratio_dict = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            s = sensitivity.get(name)
            imp = float(np.clip(s.mean(), 0.0, 1.0)) if s is not None else 0.5
            r = global_ratio * (1.0 + beta * (1.0 - imp))
            ratio_dict[name] = max(ratio_min, min(ratio_max, r))

    # 2) 通道级缺陷感知重要性：敏感度越高越保留
    def importance(group):
        module = getattr(getattr(group, "dep", None), "target", None)
        module = getattr(module, "module", module)
        name = name_of.get(module)
        idxs = list(group.idxs)
        if name is not None and name in sensitivity:
            s = sensitivity[name]
            scores = torch.tensor(
                [float(s[i]) if i < len(s) else 0.0 for i in idxs],
                dtype=torch.float32,
            )
            # 加微小扰动，避免全等导致剪枝器报错
            return scores + 1e-6 * torch.rand(len(idxs))
        return torch.ones(len(idxs), dtype=torch.float32)

    return _meta_prune(model, example_inputs, importance, global_ratio,
                       ratio_dict, ignored_layers)
