"""
模型下载脚本
下载 Qwen1.5-1.8B-Chat 和 BGE 向量模型
"""

import os
from pathlib import Path


def download_model(model_name: str, save_dir: str):
    """
    下载 HuggingFace 模型

    Args:
        model_name: 模型名称
        save_dir: 保存目录
    """
    from huggingface_hub import snapshot_download

    print(f"📥 下载模型：{model_name}")
    print(f"💾 保存到：{save_dir}")

    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=save_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✅ 模型下载完成：{save_dir}")
    except Exception as e:
        print(f"❌ 下载失败：{str(e)}")
        # 使用镜像站点重试
        print("🔄 尝试使用镜像站点...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        snapshot_download(
            repo_id=model_name,
            local_dir=save_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✅ 模型下载完成：{save_dir}")


def main():
    """主函数"""
    print("=" * 60)
    print("模型下载脚本")
    print("=" * 60)

    # 创建 models 目录
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)

    # 下载 Qwen 模型
    print("\n1️⃣ 下载 Qwen1.5-1.8B-Chat 模型...")
    download_model(
        model_name="Qwen/Qwen1.5-1.8B-Chat",
        save_dir=str(models_dir / "Qwen1.5-1.8B-Chat"),
    )

    # 下载向量模型
    print("\n2️⃣ 下载 BGE 向量模型...")
    download_model(
        model_name="BAAI/bge-small-zh-v1.5",
        save_dir=str(models_dir / "bge-small-zh-v1.5"),
    )

    print("\n" + "=" * 60)
    print("✅ 所有模型下载完成")
    print("=" * 60)

    # 显示模型信息
    print("\n📁 模型目录结构:")
    for model_path in models_dir.iterdir():
        if model_path.is_dir():
            print(f"   📂 {model_path.name}")


if __name__ == "__main__":
    main()
