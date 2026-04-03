"""
本地知识问答机器人 - 主应用
基于 FluxGym + LangChain + LlamaFactory + RAGFlow 技术栈
纯 CPU 环境 Windows 系统部署
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LocalKnowledgeBot:
    """本地知识问答机器人"""

    def __init__(
        self,
        model_path: str = "./models/Qwen1.5-1.8B-Chat",
        embedding_path: str = "./models/bge-small-zh-v1.5",
        persist_directory: str = "./vector_store",
        upload_dir: str = "./uploads",
    ):
        """
        初始化机器人

        Args:
            model_path: Qwen 模型路径
            embedding_path: 向量模型路径
            persist_directory: 向量数据库持久化目录
            upload_dir: 文档上传目录
        """
        self.model_path = model_path
        self.embedding_path = embedding_path
        self.persist_directory = persist_directory
        self.upload_dir = upload_dir

        # 创建必要的目录
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)

        # 初始化组件
        logger.info("初始化向量模型...")
        self.embeddings = self._init_embeddings()

        logger.info("初始化文本分块器...")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        )

        logger.info("初始化向量数据库...")
        self.vectorstore = self._init_vectorstore()

        logger.info("初始化 LLM...")
        self.llm = self._init_llm()

        logger.info("初始化检索问答链...")
        self.qa_chain = self._init_qa_chain()

        logger.info("✅ 本地知识问答机器人初始化完成")

    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        """初始化向量模型"""
        model_kwargs = {"device": "cpu", "trust_remote_code": True}
        encode_kwargs = {"normalize_embeddings": True, "batch_size": 8}

        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            show_progress=True,
        )
        return embeddings

    def _init_vectorstore(self) -> Chroma:
        """初始化向量数据库"""
        vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="local_knowledge",
        )
        return vectorstore

    def _init_llm(self) -> LLM:
        """
        初始化 Qwen LLM
        使用 LlamaFactory 进行本地推理
        """
        try:
            from llamafactory.chat import ChatModel

            # LlamaFactory 配置
            args = {
                "model_name_or_path": self.model_path,
                "template": "qwen",
                "infer_dtype": "int8",  # CPU 量化
                "max_length": 2048,
                "max_new_tokens": 1024,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
            }

            llm = ChatModel(args)
            return llm

        except ImportError:
            logger.warning("LlamaFactory 未安装，使用 transformers 直接加载")
            return self._init_llm_transformers()

    def _init_llm_transformers(self) -> LLM:
        """使用 transformers 初始化 LLM"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        import torch

        logger.info(f"从 {self.model_path} 加载模型...")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
            load_in_8bit=True,
            low_cpu_mem_usage=True,
        )

        generation_config = GenerationConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        generation_config.update(
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

        class QwenLLM(LLM):
            """Qwen LLM 包装类"""

            def __init__(self, model, tokenizer, generation_config, **kwargs):
                super().__init__(**kwargs)
                self.model = model
                self.tokenizer = tokenizer
                self.generation_config = generation_config

            @property
            def _llm_type(self) -> str:
                return "qwen"

            def _call(
                self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any
            ) -> str:
                messages = [
                    {"role": "system", "content": "你是一个有帮助的助手。"},
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

        return QwenLLM(
            model=model, tokenizer=tokenizer, generation_config=generation_config
        )

    def _init_qa_chain(self) -> RetrievalQA:
        """初始化检索问答链"""
        prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，
不允许在答案中添加编造成分，答案请使用中文。

已知信息：
{context}

问题：{question}
答案："""

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

        return qa_chain

    def upload_document(self, file_path: str) -> bool:
        """
        上传并处理文档

        Args:
            file_path: 文档路径

        Returns:
            bool: 是否成功
        """
        try:
            logger.info(f"处理文档：{file_path}")

            # 根据文件类型选择加载器
            file_ext = Path(file_path).suffix.lower()

            if file_ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_ext == ".docx":
                loader = Docx2txtLoader(file_path)
            elif file_ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_ext == ".md":
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                logger.error(f"不支持的文件格式：{file_ext}")
                return False

            # 加载文档
            documents = loader.load()

            # 分割文本
            texts = self.text_splitter.split_documents(documents)

            # 添加到向量数据库
            self.vectorstore.add_documents(texts)

            # 持久化
            self.vectorstore.persist()

            logger.info(f"✅ 文档处理完成：{file_path}")
            logger.info(f"   - 分割为 {len(texts)} 个文本块")

            return True

        except Exception as e:
            logger.error(f"处理文档失败：{str(e)}")
            return False

    def upload_documents_batch(self, file_paths: List[str]) -> Dict[str, bool]:
        """
        批量上传文档

        Args:
            file_paths: 文档路径列表

        Returns:
            Dict[str, bool]: 每个文件的处理结果
        """
        results = {}
        for file_path in file_paths:
            results[file_path] = self.upload_document(file_path)
        return results

    def query(self, question: str) -> Dict[str, Any]:
        """
        查询问题

        Args:
            question: 问题

        Returns:
            Dict: 包含答案和源文档
        """
        logger.info(f"查询：{question}")

        result = self.qa_chain({"query": question})

        response = {
            "answer": result["result"],
            "source_documents": [
                {"content": doc.page_content[:200], "metadata": doc.metadata}
                for doc in result["source_documents"]
            ],
        }

        logger.info("✅ 查询完成")
        return response

    def chat(self, message: str) -> str:
        """
        简单的聊天接口

        Args:
            message: 用户消息

        Returns:
            str: 机器人回复
        """
        result = self.query(message)
        return result["answer"]


def main():
    """主函数 - 演示用法"""
    print("=" * 60)
    print("本地知识问答机器人")
    print("基于 FluxGym + LangChain + LlamaFactory + RAGFlow")
    print("=" * 60)
    print()

    # 初始化机器人
    bot = LocalKnowledgeBot(
        model_path="./models/Qwen1.5-1.8B-Chat",
        embedding_path="./models/bge-small-zh-v1.5",
        persist_directory="./vector_store",
        upload_dir="./uploads",
    )

    # 交互式对话
    print("\n💬 输入 'quit' 退出，'upload <file>' 上传文档")
    print("-" * 60)

    while True:
        user_input = input("\n你：").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("再见！")
            break

        if user_input.lower().startswith("upload "):
            file_path = user_input[7:].strip()
            if os.path.exists(file_path):
                success = bot.upload_document(file_path)
                print(f"{'✅' if success else '❌'} 处理完成")
            else:
                print(f"❌ 文件不存在：{file_path}")
            continue

        # 查询
        print("\n🤖 机器人:", end=" ")
        try:
            answer = bot.chat(user_input)
            print(answer)
        except Exception as e:
            print(f"❌ 错误：{str(e)}")


if __name__ == "__main__":
    main()
