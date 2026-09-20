# -*- coding: utf-8 -*-
"""
00_check_env.py —— 环境自检
运行：.venv/Scripts/python.exe scripts/00_check_env.py
成功标志：最后一行打印 "===== 环境检查全部通过 ====="
"""
import sys


def main():
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  [OK] " if cond else "  [FAIL] ") + msg)
        if not cond:
            ok = False

    print("Python:", sys.version.split()[0])
    print("=" * 50)

    # 1) torch + CUDA
    try:
        import torch
        print(f"torch: {torch.__version__}")
        check(torch.cuda.is_available(),
              f"CUDA 可用，GPU = {torch.cuda.get_device_name(0)}" if torch.cuda.is_available()
              else "CUDA 不可用 —— 你装的是 CPU 版 torch！必须重装 CUDA 版（见 01_创建环境.bat）")
        if torch.cuda.is_available():
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    except ImportError:
        check(False, "torch 未安装")

    # 2) ultralytics
    try:
        import ultralytics
        print(f"ultralytics: {ultralytics.__version__}")
        check(True, "ultralytics 已安装")
    except ImportError:
        check(False, "ultralytics 未安装 —— 运行 pip install -r requirements.txt")

    # 3) 其它依赖
    for mod in ["numpy", "yaml", "cv2", "thop", "torch_pruning"]:
        try:
            __import__(mod)
            check(True, f"{mod} 已安装")
        except ImportError:
            check(False, f"{mod} 未安装")

    print("=" * 50)
    if ok:
        print("===== 环境检查全部通过 =====")
    else:
        print("存在失败项，按上面提示修复后再继续。")
        sys.exit(1)


if __name__ == "__main__":
    main()
