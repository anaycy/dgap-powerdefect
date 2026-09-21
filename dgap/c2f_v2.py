# -*- coding: utf-8 -*-
"""
c2f_v2.py —— C2f -> C2f_v2 重参数化（结构化剪枝的必要前置步骤）

为什么需要？
  YOLOv8 的 C2f 模块里 `self.cv1(x).chunk(2, 1)` 是一个【隐式 split】，
  torch_pruning 的依赖图无法正确解析这种 split→fan-out→concat 的通道映射，
  剪枝时会报越界索引（CUDA device-side assert / IndexError）。

解法（来自 Torch-Pruning 官方 Issue #147 的 C2f_v2 方案）：
  把 C2f 里的单个 `cv1`（输出 2c 通道再 chunk 成两半）显式拆成两个卷积
  `cv0`（前半 c 通道）和 `cv1`（后半 c 通道），并做等价的权重搬运。
  重参数化后网络前向结果完全一致，但依赖图变清晰，可被 torch_pruning 正确剪枝。

注意：C2f_v2 会被 torch.save 一起序列化进剪枝后的 checkpoint，因此本模块必须
可被 import（项目根在 sys.path 里即可），否则重载剪枝权重会报 ModuleNotFoundError。
"""
import torch
import torch.nn as nn

from ultralytics.nn.modules import Bottleneck, C2f, Conv


def infer_shortcut(bottleneck):
    """判断 Bottleneck 是否带残差连接（shortcut=True 时 c1==c2 且有 add）。"""
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, "add") and bottleneck.add


class C2f_v2(nn.Module):
    """CSP Bottleneck with 2 convolutions（把 chunk(2) 拆成 cv0/cv1 两个显式卷积）。"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def transfer_weights(c2f, c2f_v2):
    """把原 C2f 的 cv1 权重按通道一分为二，搬到 C2f_v2 的 cv0 / cv1。"""
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # cv1 输出通道的前一半 -> cv0，后一半 -> cv1
    old_weight = state_dict["cv1.conv.weight"]
    half = old_weight.shape[0] // 2
    state_dict_v2["cv0.conv.weight"] = old_weight[:half]
    state_dict_v2["cv1.conv.weight"] = old_weight[half:]

    # cv1 的 BN 参数/缓冲同样按通道拆分
    for bn_key in ["weight", "bias", "running_mean", "running_var"]:
        old_bn = state_dict[f"cv1.bn.{bn_key}"]
        state_dict_v2[f"cv0.bn.{bn_key}"] = old_bn[:half]
        state_dict_v2[f"cv1.bn.{bn_key}"] = old_bn[half:]

    # 其余参数（cv2 / bottleneck 等）原样搬
    for key in state_dict:
        if not key.startswith("cv1."):
            state_dict_v2[key] = state_dict[key]

    # 搬非方法属性（f / i / type / np 等），这些是 ultralytics 在 parse_model 里
    # 给每层打的“来自第几层/第几个模块”标记，C2f_v2 新建时没有，前向会报 AttributeError。
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and "_" not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)


def replace_c2f_with_c2f_v2(module):
    """递归把模型里所有 C2f 替换为 C2f_v2（权重等价搬运）。"""
    for name, child in module.named_children():
        if isinstance(child, C2f):
            shortcut = infer_shortcut(child.m[0])
            c2f_v2 = C2f_v2(
                child.cv1.conv.in_channels,
                child.cv2.conv.out_channels,
                n=len(child.m),
                shortcut=shortcut,
                g=child.m[0].cv2.conv.groups,
                e=child.c / child.cv2.conv.out_channels,
            )
            transfer_weights(child, c2f_v2)
            # 新建的 cv0/cv1 默认在 CPU，需与原模型对齐设备
            c2f_v2 = c2f_v2.to(next(child.parameters()).device)
            setattr(module, name, c2f_v2)
        else:
            replace_c2f_with_c2f_v2(child)
