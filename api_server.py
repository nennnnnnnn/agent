"""
FastAPI Web 服务接口
提供文档上传、检索、问答的 RESTful API
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from main import LocalKnowledgeBot


# 数据模型
class QueryRequest(BaseModel):
    """查询请求"""

    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    """查询响应"""

    answer: str
    sources: List[dict]


class ChatRequest(BaseModel):
    """聊天请求"""

    message: str


class ChatResponse(BaseModel):
    """聊天响应"""

    reply: str


# 全局变量
bot: Optional[LocalKnowledgeBot] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global bot

    # 启动时初始化
    print("🚀 初始化本地知识问答机器人...")
    bot = LocalKnowledgeBot(
        model_path="./models/Qwen1.5-1.8B-Chat",
        embedding_path="./models/bge-small-zh-v1.5",
        persist_directory="./vector_store",
        upload_dir="./uploads",
    )
    print("✅ 服务就绪")

    yield

    # 关闭时清理
    print("👋 关闭服务...")


# 创建 FastAPI 应用
app = FastAPI(
    title="本地知识问答机器人 API",
    description="基于 FluxGym + LangChain + LlamaFactory + RAGFlow 的本地 RAG 系统",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """健康检查"""
    return {
        "status": "ok",
        "message": "本地知识问答机器人 API 运行中",
        "endpoints": {
            "docs": "/docs",
            "upload": "/api/upload",
            "query": "/api/query",
            "chat": "/api/chat",
        },
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档

    - **file**: 文档文件 (支持 pdf, docx, txt, md)
    """
    if bot is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    # 验证文件类型
    allowed_extensions = {".pdf", ".docx", ".txt", ".md"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式，支持：{', '.join(allowed_extensions)}",
        )

    # 保存文件
    upload_dir = Path("./uploads")
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / file.filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 处理文档
        success = bot.upload_document(str(file_path))

        if success:
            return JSONResponse(
                {
                    "status": "success",
                    "message": f"文档 {file.filename} 处理完成",
                    "file": file.filename,
                }
            )
        else:
            raise HTTPException(status_code=500, detail="文档处理失败")

    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload/batch")
async def upload_documents_batch(files: List[UploadFile] = File(...)):
    """
    批量上传文档
    """
    if bot is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    results = []
    for file in files:
        try:
            upload_dir = Path("./uploads")
            upload_dir.mkdir(exist_ok=True)
            file_path = upload_dir / file.filename

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            success = bot.upload_document(str(file_path))
            results.append({"file": file.filename, "success": success})
        except Exception as e:
            results.append({"file": file.filename, "success": False, "error": str(e)})

    return JSONResponse({"status": "completed", "results": results})


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    知识查询

    - **question**: 问题内容
    - **top_k**: 返回的参考文档数量
    """
    if bot is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    try:
        result = bot.query(request.question)

        return QueryResponse(
            answer=result["answer"], sources=result["source_documents"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天接口

    - **message**: 用户消息
    """
    if bot is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    try:
        reply = bot.chat(request.message)
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def get_status():
    """获取系统状态"""
    return {
        "status": "running",
        "model": "./models/Qwen1.5-1.8B-Chat",
        "embedding": "./models/bge-small-zh-v1.5",
        "vector_store": "./vector_store",
        "uploads": "./uploads",
    }


def main():
    """启动 Web 服务"""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
