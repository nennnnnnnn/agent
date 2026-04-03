"""
快速测试脚本 - 仅测试基础功能
"""

import sys

print("Python 版本:", sys.version)

# 测试基础导入
try:
    import numpy as np

    print("✅ numpy 已安装")
except:
    print("❌ numpy 未安装")

try:
    from pathlib import Path

    print("✅ pathlib 可用")
except:
    print("❌ pathlib 不可用")

# 检查目录
print("\n检查项目结构:")
dirs = ["models", "uploads", "vector_store", "config", "src"]
for d in dirs:
    exists = "✅" if Path(d).exists() else "❌"
    print(f"  {exists} {d}/")

# 检查模型
print("\n检查模型文件:")
models = ["models/Qwen1.5-1.8B-Chat", "models/bge-small-zh-v1.5"]
for m in models:
    exists = "✅" if Path(m).exists() else "❌"
    print(f"  {exists} {m}")

print("\n" + "=" * 50)
print("下一步:")
print("1. 安装依赖：pip install -r requirements.txt")
print("2. 下载模型：python download_models.py")
print("3. 运行程序：python src/main_simple.py")
print("=" * 50)
