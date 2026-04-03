"""
本地知识问答机器人 - 简化版本
基于 Qwen + 向量检索的本地 RAG 系统
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging
import json
import pickle
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleEmbedding:
    """简易向量模型包装器"""

    def __init__(self, model_name="BAAI/bge-small-zh-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"加载向量模型：{model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info("✅ 向量模型加载完成")
        except Exception as e:
            logger.error(f"向量模型加载失败：{e}")
            raise

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)


class SimpleVectorStore:
    """简易向量存储"""

    def __init__(self, persist_dir: str = "./vector_store"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.documents: List[str] = []
        self.embeddings: np.ndarray = None
        self.metadata: List[Dict] = []

        # 加载已保存的数据
        self._load()

    def _load(self):
        data_file = self.persist_dir / "data.pkl"
        if data_file.exists():
            try:
                with open(data_file, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data.get("documents", [])
                    self.embeddings = data.get("embeddings")
                    self.metadata = data.get("metadata", [])
                logger.info(f"✅ 已加载 {len(self.documents)} 个文档")
            except Exception as e:
                logger.warning(f"加载数据失败：{e}")

    def _save(self):
        data_file = self.persist_dir / "data.pkl"
        with open(data_file, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "embeddings": self.embeddings,
                    "metadata": self.metadata,
                },
                f,
            )
        logger.info(f"💾 已保存向量数据")

    def add_documents(
        self, texts: List[str], embeddings: np.ndarray, metadata: List[Dict] = None
    ):
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        self.documents.extend(texts)
        self.metadata.extend(metadata or [{} for _ in texts])
        self._save()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # 计算余弦相似度
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "content": self.documents[idx],
                    "similarity": float(similarities[idx]),
                    "metadata": self.metadata[idx],
                }
            )

        return results


class SimpleQwenLLM:
    """简易 Qwen LLM"""

    def __init__(self, model_path: str = "./models/Qwen1.5-1.8B-Chat"):
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                GenerationConfig,
            )
            import torch

            logger.info(f"加载 Qwen 模型：{model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )

            self.generation_config = GenerationConfig.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.generation_config.update(
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            logger.info("✅ Qwen 模型加载完成")

        except Exception as e:
            logger.error(f"模型加载失败：{e}")
            raise

    def generate(self, prompt: str, context: str = "") -> str:
        import torch

        system_prompt = "你是一个有帮助的助手。请基于以下信息回答问题。"
        if context:
            system_prompt += f"\n\n参考信息:\n{context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, generation_config=self.generation_config
            )

        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
        )

        return response


class LocalKnowledgeBot:
    """本地知识问答机器人"""

    def __init__(
        self,
        model_path: str = "./models/Qwen1.5-1.8B-Chat",
        embedding_path: str = "./models/bge-small-zh-v1.5",
        persist_directory: str = "./vector_store",
    ):
        self.persist_directory = persist_directory

        # 初始化向量模型
        logger.info("初始化向量模型...")
        self.embeddings = SimpleEmbedding(embedding_path)

        # 初始化向量存储
        logger.info("初始化向量存储...")
        self.vectorstore = SimpleVectorStore(persist_directory)

        # 初始化 LLM
        logger.info("初始化 Qwen 模型...")
        self.llm = SimpleQwenLLM(model_path)

        logger.info("✅ 机器人初始化完成")

    def process_document(
        self, file_path: str, chunk_size: int = 512, chunk_overlap: int = 50
    ) -> bool:
        """处理文档"""
        try:
            logger.info(f"处理文档：{file_path}")

            # 读取文件
            file_ext = Path(file_path).suffix.lower()
            content = ""

            if file_ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            elif file_ext == ".md":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            elif file_ext == ".pdf":
                try:
                    from pypdf import PdfReader

                    reader = PdfReader(file_path)
                    content = "\n".join([page.extract_text() for page in reader.pages])
                except ImportError:
                    logger.error("请安装 pypdf: pip install pypdf")
                    return False
            elif file_ext == ".docx":
                try:
                    from docx import Document

                    doc = Document(file_path)
                    content = "\n".join([para.text for para in doc.paragraphs])
                except ImportError:
                    logger.error("请安装 python-docx: pip install python-docx")
                    return False
            else:
                logger.error(f"不支持的文件格式：{file_ext}")
                return False

            # 分块
            chunks = self._split_text(content, chunk_size, chunk_overlap)

            # 生成向量
            logger.info(f"生成 {len(chunks)} 个文本块的向量...")
            chunk_embeddings = self.embeddings.encode(chunks)

            # 添加到存储
            metadata = [
                {"source": str(file_path), "chunk": i} for i in range(len(chunks))
            ]
            self.vectorstore.add_documents(chunks, chunk_embeddings, metadata)

            logger.info(f"✅ 文档处理完成")
            return True

        except Exception as e:
            logger.error(f"处理文档失败：{e}")
            return False

    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """文本分块"""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]

            # 尝试在句子边界处分割
            if end < text_length:
                for sep in ["。", "！", "？", "\n", " "]:
                    last_sep = chunk.rfind(sep)
                    if last_sep > chunk_size // 2:
                        chunk = chunk[: last_sep + 1]
                        break

            chunks.append(chunk.strip())
            start += chunk_size - chunk_overlap

        return [c for c in chunks if c]

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """查询问题"""
        logger.info(f"查询：{question}")

        # 向量检索
        query_embedding = self.embeddings.encode([question])[0]
        results = self.vectorstore.search(query_embedding, top_k)

        if not results:
            return {"answer": "抱歉，没有找到相关信息。请先上传文档。", "sources": []}

        # 构建上下文
        context = "\n\n".join([r["content"] for r in results])

        # 生成答案
        answer = self.llm.generate(question, context)

        return {"answer": answer, "sources": results}

    def chat(self, message: str) -> str:
        """简单聊天"""
        result = self.query(message)
        return result["answer"]


def main():
    """主函数"""
    print("=" * 60)
    print("本地知识问答机器人")
    print("=" * 60)

    # 检查模型
    if not Path("./models/Qwen1.5-1.8B-Chat").exists():
        print("\n❌ 模型不存在，请先运行: python download_models.py")
        return

    # 初始化
    bot = LocalKnowledgeBot(
        model_path="./models/Qwen1.5-1.8B-Chat",
        embedding_path="./models/bge-small-zh-v1.5",
        persist_directory="./vector_store",
    )

    print("\n💬 输入 'quit' 退出，'upload <文件>' 上传文档")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n你：").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("再见！")
                break

            if user_input.lower().startswith("upload "):
                file_path = user_input[7:].strip()
                if os.path.exists(file_path):
                    success = bot.process_document(file_path)
                    print(f"{'✅' if success else '❌'} 处理完成")
                else:
                    print(f"❌ 文件不存在：{file_path}")
                continue

            # 查询
            print("\n🤖 机器人:", end=" ", flush=True)
            answer = bot.chat(user_input)
            print(answer)

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"❌ 错误：{e}")


if __name__ == "__main__":
    main()
