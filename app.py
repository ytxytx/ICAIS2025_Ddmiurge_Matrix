import os
import json
import base64
from typing import Optional, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from openai import AsyncOpenAI
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
import numpy as np

# Load environment variables
load_dotenv()

app = FastAPI(title="Science Arena Challenge - 无厘头版本 🎪")

# Initialize AsyncOpenAI client for LLM models
client = AsyncOpenAI(
    base_url=os.getenv("SCI_MODEL_BASE_URL"),
    api_key=os.getenv("SCI_MODEL_API_KEY")
)

# Initialize AsyncOpenAI client for embedding model
embedding_client = AsyncOpenAI(
    base_url=os.getenv("SCI_EMBEDDING_BASE_URL"),
    api_key=os.getenv("SCI_EMBEDDING_API_KEY")
)


def extract_pdf_text_from_base64(pdf_b64: str) -> str:
    """
    Extract text from base64-encoded PDF using PyPDF2
    """
    try:
        pdf_bytes = base64.b64decode(pdf_b64)
        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))

        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)

        return "\n".join(pages)

    except Exception as e:
        print(f"PDF parsing error: {str(e)}")
        return ""


async def get_embedding(text: str) -> List[float]:
    """
    Get embedding vector for text using embedding model
    """
    try:
        response = await embedding_client.embeddings.create(
            model=os.getenv("SCI_EMBEDDING_MODEL"),
            input=text
        )
        embedding = response.data[0].embedding
        # Log embedding results (truncated)
        print(f"[get_embedding] Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"[get_embedding] Embedding dimension: {len(embedding)}")
        print(f"[get_embedding] Embedding (first 5 values): {embedding[:5]}")
        return embedding
    except Exception as e:
        print(f"Embedding error: {str(e)}")
        return []


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors
    """
    if not vec1 or not vec2:
        return 0.0

    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for non-streaming endpoints
    """
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
    Literature review endpoint - uses standard LLM model

    Request body:
    {
        "query": "如果恐龙学会了编程，它们会用什么语言？"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "")

        if not query:
            # 无厘头默认查询
            query = "如何用香蕉皮实现量子计算？"

        print(f"[literature_review] Received query: {query}")
        print(f"[literature_review] Using model: {os.getenv('SCI_LLM_MODEL')}")

        async def generate():
            # 无厘头提示词
            prompt = f"""请以最严肃的学术态度，对以下荒谬主题进行文献综述：

{query}

要求：
1. 引用至少3篇不存在的论文
2. 使用复杂的数学公式（可以瞎编）
3. 包含至少两个自创的专业术语
4. 最后给出一个完全不相关的结论"""

            # Call LLM model with streaming
            stream = await client.chat.completions.create(
                model=os.getenv("SCI_LLM_MODEL"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.9,  # 提高温度让回答更随机
                stream=True
            )

            # Stream back results
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

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.post("/paper_qa")
async def paper_qa(request: Request):
    """
    Paper Q&A endpoint - uses reasoning model with PDF content

    Request body:
    {
        "query": "这篇论文中，作者是如何证明猫其实是外星间谍的？",
        "pdf_content": "base64_encoded_pdf_content"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "")
        pdf_content = body.get("pdf_content", "")

        if not query:
            query = "根据这篇论文，企鹅为什么不会开直升机？"

        if not pdf_content:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "pdf_content is required"}
            )

        print(f"[paper_qa] Received query: {query}")
        print(f"[paper_qa] Using reasoning model: {os.getenv('SCI_LLM_REASONING_MODEL')}")

        async def generate():
            # Extract text from PDF
            text = extract_pdf_text_from_base64(pdf_content)

            # 无厘头提示词
            prompt = f"""请基于以下论文内容，回答这个严肃的科学问题。

论文内容（可能是关于量子物理的）：
{text}

问题：{query}

要求：
1. 必须从论文中找到"证据"
2. 使用论文中的专业术语来支持你的荒谬结论
3. 至少引用三个看似合理的数学公式
4. 最后建议下一步研究方向（越离谱越好）"""

            # Call reasoning model with streaming
            stream = await client.chat.completions.create(
                model=os.getenv("SCI_LLM_REASONING_MODEL"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.9,
                stream=True
            )

            # Stream back results
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    # Extract and log reasoning content
                    reasoning_content = getattr(delta, 'reasoning_content', None)
                    if reasoning_content:
                        print(f"[paper_qa] 荒谬推理: {reasoning_content}", flush=True)

                    # Stream regular content to client
                    delta_content = delta.content
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

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.post("/ideation")
async def ideation(request: Request):
    """
    Ideation endpoint - uses embedding model for similarity and LLM for generation

    Request body:
    {
        "query": "如何用洗衣机研究暗物质？"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "")

        if not query:
            query = "如何训练金鱼成为数据科学家？"

        # 无厘头参考想法
        reference_ideas = [
            "用微波炉观测黑洞蒸发的实验设计",
            "基于泡面弹性模量的新材料研究",
            "利用扫地机器人进行城市地形测绘",
            "通过分析猫咪打哈欠预测股市走势",
            "使用香蕉皮作为量子比特载体",
            "基于打喷嚏频率的情感识别系统",
            "用洗衣机离心力模拟引力波探测",
            "通过分析云朵形状进行天气预报的深度学习模型"
        ]

        print(f"[ideation] Received query: {query}")
        print(f"[ideation] Using {len(reference_ideas)} 个荒谬参考想法进行嵌入相似度分析")
        print(f"[ideation] Using LLM model: {os.getenv('SCI_LLM_MODEL')}")
        print(f"[ideation] Using embedding model: {os.getenv('SCI_EMBEDDING_MODEL')}")

        async def generate():
            # 无厘头提示词
            prompt = f"""请为以下荒谬研究主题生成创新性的研究想法：

研究主题：{query}

要求：
1. 每个想法都要听起来很科学但实际上完全不可行
2. 包含假想的实验装置描述
3. 预测一些不可能的研究结果
4. 建议申请哪些根本不存在的科研基金"""

            # Use embedding model to find similarities with hardcoded reference ideas
            print("[ideation] 正在计算荒谬想法的嵌入相似度...")

            # Get embedding for query
            query_embedding = await get_embedding(query)

            # Get embeddings for reference ideas and compute similarities
            similarities = []
            for idx, idea in enumerate(reference_ideas):
                idea_embedding = await get_embedding(idea)
                similarity = cosine_similarity(query_embedding, idea_embedding)
                similarities.append((idx, idea, similarity))

            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[2], reverse=True)

            # Add similarity analysis to prompt
            prompt += f"\n\n相关荒谬想法参考（按相似度排序）：\n"
            for idx, idea, sim in similarities[:3]:  # 只取前3个最相似的
                prompt += f"\n{idx+1}. (荒谬相似度: {sim:.3f}) {idea}"

            prompt += "\n\n基于以上参考，请生成更加创新（且更加荒谬）的研究想法！"

            # Call LLM model with streaming
            stream = await client.chat.completions.create(
                model=os.getenv("SCI_LLM_MODEL"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=1.0,  # 最高温度，让回答最随机
                stream=True
            )

            # Stream back results
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

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.post("/paper_review")
async def paper_review(request: Request):
    """
    Paper review endpoint - uses LLM model with PDF content

    Request body:
    {
        "query": "请用莎士比亚的风格评审这篇论文",
        "pdf_content": "base64_encoded_pdf_content"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "请用说唱的方式给这篇论文写评审意见")
        pdf_content = body.get("pdf_content", "")

        if not pdf_content:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "pdf_content is required"}
            )

        print(f"[paper_review] Received query: {query}")
        print(f"[paper_review] Using model: {os.getenv('SCI_LLM_MODEL')}")

        async def generate():
            # Extract text from PDF
            text = extract_pdf_text_from_base64(pdf_content)

            # 无厘头评审提示词
            prompt = f"""请按照以下特殊要求评审这篇论文：

论文内容：
{text}

评审要求：{query}

额外指示：
1. 评审意见要严肃但内容要荒谬
2. 指出论文中不存在的"重大缺陷"
3. 建议一些不可能实现的改进方案
4. 用专业术语包装毫无意义的建议
5. 最后给出一个戏剧性的总体评价"""

            # Call LLM model with streaming
            stream = await client.chat.completions.create(
                model=os.getenv("SCI_LLM_MODEL"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.9,
                stream=True
            )

            # Stream back results
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

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.get("/health")
async def health():
    """健康检查端点 - 也改成无厘头版本"""
    return {
        "status": "极度健康", 
        "message": "系统正在愉快地生成荒谬内容",
        "absurdity_level": 99.9,
        "warning": "请不要在喝水时使用本系统"
    }


@app.get("/")
async def root():
    """根端点 - 无厘头欢迎信息"""
    return {
        "message": "欢迎来到科学竞技场无厘头版本！🎪",
        "description": "这里的一切都很科学（才怪）",
        "endpoints": {
            "/literature_review": "为荒谬主题撰写'严肃'文献综述",
            "/paper_qa": "从正经论文中找出荒谬答案", 
            "/ideation": "生成不可能实现的研究想法",
            "/paper_review": "用各种奇怪风格评审论文"
        },
        "disclaimer": "本系统输出内容纯属娱乐，如有人当真，那一定是在做梦"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
