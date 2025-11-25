#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
install.py
安装联合 CNN 图像分类 + 回归所需的依赖
"""

import subprocess
import sys

def install(package):
    """安装 Python 包"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# =========================
# 依赖列表
# =========================
packages = [
    "tensorflow>=2.12",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "pandas"   # 可选，如果后续处理数据需要
]

# =========================
# 安装依赖
# =========================
for pkg in packages:
    try:
        __import__(pkg.split("==")[0].split(">=")[0])
        print(f"{pkg} 已安装")
    except ImportError:
        print(f"{pkg} 未安装，正在安装...")
        install(pkg)

print("\n所有依赖安装完成！你可以运行联合 CNN 脚本了。")
