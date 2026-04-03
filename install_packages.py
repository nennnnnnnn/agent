"""
在虚拟环境中安装所有依赖包
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """运行命令"""
    print(f"\n{'=' * 60}")
    print(f"正在：{description}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def main():
    print("=" * 60)
    print("本地知识问答机器人 - 依赖包安装脚本")
    print("=" * 60)

    # 检查虚拟环境
    venv_path = os.path.join(os.path.dirname(__file__), "venv")
    if not os.path.exists(venv_path):
        print("\n创建虚拟环境...")
        run_command(f'"{sys.executable}" -m venv "{venv_path}"', "创建虚拟环境")

    # 虚拟环境的 pip 路径
    venv_pip = os.path.join(venv_path, "Scripts", "pip.exe")
    venv_python = os.path.join(venv_path, "Scripts", "python.exe")

    if not os.path.exists(venv_pip):
        print(f"错误：虚拟环境 pip 不存在：{venv_pip}")
        return

    print(f"\n使用虚拟环境：{venv_path}")

    # 升级 pip
    run_command(
        f'"{venv_python}" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple',
        "升级 pip",
    )

    # 安装 PyTorch CPU 版本
    run_command(
        f'"{venv_pip}" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -i https://pypi.tuna.tsinghua.edu.cn/simple',
        "安装 PyTorch (CPU 版本)",
    )

    # 安装 Transformers 相关
    run_command(
        f'"{venv_pip}" install transformers sentence-transformers accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple',
        "安装 Transformers 相关",
    )

    # 安装 LangChain 相关
    run_command(
        f'"{venv_pip}" install chromadb langchain langchain-community langchain-core -i https://pypi.tuna.tsinghua.edu.cn/simple',
        "安装 LangChain 相关",
    )

    # 安装 FastAPI 相关
    run_command(
        f'"{venv_pip}" install fastapi uvicorn python-multipart pydantic pydantic-settings -i https://pypi.tuna.tsinghua.edu.cn/simple',
        "安装 FastAPI 相关",
    )

    # 安装文档处理相关
    run_command(
        f'"{venv_pip}" install pypdf python-docx markdown -i https://pypi.tuna.tsinghua.edu.cn/simple',
        "安装文档处理包",
    )

    # 安装其他依赖
    run_command(
        f'"{venv_pip}" install pillow numpy pandas tqdm huggingface-hub einops tiktoken -i https://pypi.tuna.tsinghua.edu.cn/simple',
        "安装其他依赖",
    )

    # 创建必要目录
    base_dir = os.path.dirname(__file__)
    for dir_name in ["models", "uploads", "vector_store", "logs"]:
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"创建目录：{dir_path}")

    # 验证安装
    print("\n" + "=" * 60)
    print("验证安装...")
    print("=" * 60)

    packages = [
        "torch",
        "transformers",
        "sentence_transformers",
        "fastapi",
        "langchain",
        "chromadb",
        "pypdf",
        "docx",
    ]

    for pkg in packages:
        try:
            subprocess.run(
                [venv_python, "-c", f'import {pkg}; print(f"✅ {pkg} 已安装")'],
                check=True,
            )
        except subprocess.CalledProcessError:
            print(f"❌ {pkg} 安装失败")

    print("\n" + "=" * 60)
    print("安装完成！")
    print("=" * 60)
    print(f"\n虚拟环境位置：{venv_path}")
    print("\n下一步:")
    print("1. 下载模型：运行 download.bat")
    print("2. 启动服务：运行 start.bat")
    print("\n激活虚拟环境命令:")
    print(f"  call {venv_path}\\Scripts\\activate.bat")
    print()


if __name__ == "__main__":
    main()
