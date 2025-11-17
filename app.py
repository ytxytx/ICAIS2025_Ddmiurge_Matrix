import os
import json
import logging
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.config import config, TimeoutConfig
from src.models import AgentConfig, HealthCheck
from src.agents.ideation import IdeationAgent
from src.agents.review import ReviewAgent
from src.services.document_processor import DocumentProcessor
from src.services.academic_data import AcademicDataService
from src.services.embedding_service import EmbeddingService

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="AI Scientist Challenge - Professional Version",
    description="智能学术研究助手，提供文献综述、论文问答、研究构思和论文评审功能",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务
document_processor = DocumentProcessor()
academic_service = AcademicDataService()
embedding_service = EmbeddingService()

# 初始化智能体
ideation_agent = IdeationAgent(
    config=AgentConfig(
        name="ideation_agent",
        model=config.SCI_LLM_MODEL,
        temperature=0.8,
        max_tokens=2048,
        timeout=TimeoutConfig.IDEATION
    )
)

review_agent = ReviewAgent(
    config=AgentConfig(
        name="review_agent",
        model=config.SCI_LLM_MODEL,
        temperature=0.7,
        max_tokens=2048,
        timeout=TimeoutConfig.PAPER_REVIEW
    )
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logging.error(f"全局异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc)
        }
    )


@app.post("/literature_review")
async def literature_review(request: Request):
    """
    文献综述端点 - 对研究主题进行全面的文献综述
    
    Request body:
    {
        "query": "请帮我全面梳理扩散语言模型领域的最新进展"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "").strip()

        if not query:
            raise HTTPException(
                status_code=400,
                detail={"error": "Bad Request", "message": "Query is required"}
            )

        logging.info(f"[literature_review] 接收查询: {query}")

        async def generate():
            try:
                # 使用研究构思智能体进行文献综述
                async for chunk in ideation_agent.execute(query, task_type="literature_review"):
                    yield chunk
                    
            except Exception as e:
                logging.error(f"文献综述流式生成失败: {str(e)}")
                error_chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "delta": {
                            "content": f"\n\n❌ 文献综述过程中出现错误: {str(e)}\n\n"
                        }
                    }]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"文献综述端点错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/paper_qa")
async def paper_qa(request: Request):
    """
    论文问答端点 - 基于PDF内容回答论文相关问题
    
    Request body:
    {
        "query": "请仔细分析并解释本文中使用的强化学习训练方法",
        "pdf_content": "base64_encoded_pdf_content"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "").strip()
        pdf_content = body.get("pdf_content", "")

        if not query:
            query = "请分析这篇论文的主要贡献和方法"

        if not pdf_content:
            raise HTTPException(
                status_code=400,
                detail={"error": "Bad Request", "message": "pdf_content is required"}
            )

        logging.info(f"[paper_qa] 接收查询: {query}")

        async def generate():
            try:
                # 提取PDF文本
                text = document_processor.extract_text(pdf_content)
                if not text:
                    raise ValueError("无法从PDF中提取文本内容")

                # 使用推理模型进行深度分析
                prompt = f"""
                请基于以下论文内容，仔细分析并回答用户的问题。

                论文内容：
                {text[:8000]}  # 限制文本长度

                问题：{query}

                要求：
                1. 基于论文内容提供准确的回答
                2. 引用论文中的具体内容支持你的分析
                3. 如果论文中没有相关信息，请明确指出
                4. 提供深入的技术分析
                """

                # 使用推理模型
                from openai import AsyncOpenAI
                client = AsyncOpenAI(
                    base_url=config.SCI_MODEL_BASE_URL,
                    api_key=config.SCI_MODEL_API_KEY
                )

                stream = await client.chat.completions.create(
                    model=config.SCI_LLM_REASONING_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2048,
                    temperature=0.7,
                    stream=True
                )

                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta_content = chunk.choices[0].delta.content
                        if delta_content:
                            response_data = {
                                "object": "chat.completion.chunk",
                                "choices": [{
                                    "delta": {
                                        "content": delta_content
                                    }
                                }]
                            }
                            yield f"data: {json.dumps(response_data)}\n\n"

            except Exception as e:
                logging.error(f"论文问答流式生成失败: {str(e)}")
                error_chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "delta": {
                            "content": f"\n\n❌ 论文问答过程中出现错误: {str(e)}\n\n"
                        }
                    }]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"论文问答端点错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ideation")
async def ideation(request: Request):
    """
    研究构思端点 - 生成创新性研究想法
    
    Request body:
    {
        "query": "请帮我提出使用LLM技术进行时空数据预测的创新想法"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "").strip()

        if not query:
            raise HTTPException(
                status_code=400,
                detail={"error": "Bad Request", "message": "Query is required"}
            )

        logging.info(f"[ideation] 接收查询: {query}")

        async def generate():
            try:
                # 使用研究构思智能体
                async for chunk in ideation_agent.execute(query):
                    yield chunk
                    
            except Exception as e:
                logging.error(f"研究构思流式生成失败: {str(e)}")
                error_chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "delta": {
                            "content": f"\n\n❌ 研究构思过程中出现错误: {str(e)}\n\n"
                        }
                    }]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"研究构思端点错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/paper_review")
async def paper_review(request: Request):
    """
    论文评审端点 - 对论文进行结构化评审
    
    Request body:
    {
        "query": "请对这篇论文进行简要评审",
        "pdf_content": "base64_encoded_pdf_content"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "请对这篇论文进行评审").strip()
        pdf_content = body.get("pdf_content", "")

        if not pdf_content:
            raise HTTPException(
                status_code=400,
                detail={"error": "Bad Request", "message": "pdf_content is required"}
            )

        logging.info(f"[paper_review] 接收查询: {query}")

        async def generate():
            try:
                # 使用论文评审智能体
                async for chunk in review_agent.execute(query, pdf_content=pdf_content):
                    yield chunk
                    
            except Exception as e:
                logging.error(f"论文评审流式生成失败: {str(e)}")
                error_chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "delta": {
                            "content": f"\n\n❌ 论文评审过程中出现错误: {str(e)}\n\n"
                        }
                    }]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"论文评审端点错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """健康检查端点"""
    from datetime import datetime
    
    checks = {
        "llm_service": True,  # 简化检查
        "embedding_service": True,
        "document_processor": True,
        "academic_service": True
    }
    
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now(),
        checks=checks
    )


@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "欢迎使用 AI Scientist Challenge 专业版",
        "description": "智能学术研究助手系统",
        "version": "1.0.0",
        "endpoints": {
            "/literature_review": "文献综述 - 对研究主题进行全面文献梳理",
            "/paper_qa": "论文问答 - 基于PDF内容回答论文相关问题", 
            "/ideation": "研究构思 - 生成创新性研究想法",
            "/paper_review": "论文评审 - 对论文进行结构化评审"
        },
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level=config.LOG_LEVEL.lower()
    )
