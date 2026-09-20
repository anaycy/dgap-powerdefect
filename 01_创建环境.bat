@echo off
REM ============================================================
REM  DGAP 一键环境：venv + CUDA 版 torch + 全部依赖
REM  用法：双击本文件，或在项目目录命令行运行它
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0"

echo [1/3] 创建虚拟环境 .venv ...
python -m venv .venv
if errorlevel 1 (
    echo 创建 venv 失败，请确认已安装 Python 并加入 PATH。
    pause & exit /b 1
)

echo [2/3] 安装 CUDA 版 PyTorch ^(约 2.5GB，请耐心等待^)...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

echo [3/3] 安装其余依赖 ...
.venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo ============================================================
echo  环境安装完成！请运行自检：
echo  .venv\Scripts\python.exe scripts\00_check_env.py
echo ============================================================
pause
