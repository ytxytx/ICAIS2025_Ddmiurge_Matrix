import os
import json
import base64
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from openai import AsyncOpenAI
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
import aiohttp
import numpy as np
import asyncio

load_dotenv()

app = FastAPI(title="Science Arena Challenge API")
# 获取 Semantic Scholar API 配置
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
SEMANTIC_SCHOLAR_API_BASE_URL = "https://api.semanticscholar.org/graph/v1"

client = AsyncOpenAI(
    base_url=os.getenv("SCI_MODEL_BASE_URL"),
    api_key=os.getenv("SCI_MODEL_API_KEY")
)

embedding_client = AsyncOpenAI(
    base_url=os.getenv("SCI_EMBEDDING_BASE_URL"),
    api_key=os.getenv("SCI_EMBEDDING_API_KEY")
)


def extract_pdf_text_from_base64(pdf_b64: str) -> str:
    try:
        pdf_bytes = base64.b64decode(pdf_b64)
        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


async def get_embedding(text: str) -> List[float]:
    try:
        response = await embedding_client.embeddings.create(
            model=os.getenv("SCI_EMBEDDING_MODEL"),
            input=text
        )
        return response.data[0].embedding
    except Exception:
        return []


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    a, b = np.array(vec1), np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

async def generate_search_keywords(query: str) -> List[str]:
    """
    使用LLM生成适合Semantic Scholar搜索的关键词
    """
    try:
        prompt = f"""
        根据以下科研查询，生成1个最适合在学术搜索引擎 Semantic Scholar 中搜索的关键词。
        要求：
        1. 使用英文关键词
        2. 包含具体的技术术语和领域术语
        3. 优先使用在学术论文中常见的表达方式
        4. 返回格式：纯文本，仅一行，一个关键词
        
        用户查询：{query}
        
        请直接返回关键词，不要额外解释：
        """
        
        response = await client.chat.completions.create(
            model=os.getenv("SCI_LLM_MODEL"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3
        )
        
        keywords_text = response.choices[0].message.content.strip()
        # 解析返回的关键词，每行一个
        keywords = [k.strip() for k in keywords_text.split('\n') if k.strip()]
        
        # 如果LLM返回格式不对，回退到基于查询的简单处理
        if not keywords:
            # 简单的关键词提取：取前几个有意义的词
            words = query.split()
            important_words = [w for w in words if len(w) > 4][:3]
            keywords = important_words if important_words else [query]
            
        return keywords
        
    except Exception as e:
        print(f"生成关键词时出错: {str(e)}")
        # 回退方案：使用查询中的主要词汇
        words = query.split()
        return words[:3] if len(words) >= 3 else [query]

async def get_related_papers_from_keywords(keywords: List[str], max_papers: int = 20) -> List[Dict[str, Any]]:
    """
    使用多个关键词从 Semantic Scholar 获取相关论文
    """
    all_papers = []
    
    try:
        headers = {}
        if SEMANTIC_SCHOLAR_API_KEY:
            headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
        
        async with aiohttp.ClientSession() as session:
            for keyword in keywords[:3]:  # 最多使用前3个关键词
                try:
                    params = {
                        "query": f'"{keyword}"',  # 使用引号确保精确匹配
                        "limit": 10,  # 每个关键词获取10篇
                        "fields": "title,authors,year,venue,publicationTypes,citationCount,url,abstract",
                        "year": "2018-",
                        "fieldsOfStudy": "Computer Science,Engineering,Mathematics,Physics,Biology,Chemistry,Medicine"  # 限制在科学领域
                    }
                    
                    async with session.get(
                        f"{SEMANTIC_SCHOLAR_API_BASE_URL}/paper/search",
                        params=params,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        
                        if response.status == 200:
                            data = await response.json()
                            papers = data.get("data", [])
                            all_papers.extend(papers)
                            
                            # 短暂暂停，避免频繁请求
                            await asyncio.sleep(0.5)
                            
                except Exception as e:
                    print(f"搜索关键词 '{keyword}' 时出错: {str(e)}")
                    continue
        
        # 去重并排序
        seen_paper_ids = set()
        unique_papers = []
        
        for paper in all_papers:
            paper_id = paper.get("paperId")
            if paper_id and paper_id not in seen_paper_ids:
                seen_paper_ids.add(paper_id)
                unique_papers.append(paper)
        
        # 按引用量排序并限制数量
        sorted_papers = sorted(
            unique_papers, 
            key=lambda x: x.get("citationCount", 0), 
            reverse=True
        )[:max_papers]
        
        return sorted_papers
        
    except Exception as e:
        print(f"获取论文时出错: {str(e)}")
        return []

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "Internal Server Error", "message": str(exc)})


@app.post("/literature_review")
async def literature_review(request: Request):
    try:
        body = await request.json()
        query = body.get("query", "Please conduct a literature review on an unconventional topic.")
        prompt = f"Provide a rigorous academic literature review on the following topic:\n\n{query}"
        stream = await client.chat.completions.create(
            model=os.getenv("SCI_LLM_MODEL"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.7,
            stream=True
        )

        async def generate():
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'choices':[{'delta':{'content': chunk.choices[0].delta.content}}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/paper_qa")
async def paper_qa(request: Request):
    try:
        body = await request.json()
        query = body.get("query", "Summarize the key insights from this paper.")
        pdf_content = body.get("pdf_content")
        if not pdf_content:
            return JSONResponse(status_code=400, content={"error": "pdf_content is required"})

        text = extract_pdf_text_from_base64(pdf_content)
        prompt = f"Based on the following paper, answer the question.\n\nPaper Content:\n{text}\n\nQuestion: {query}"
        stream = await client.chat.completions.create(
            model=os.getenv("SCI_LLM_REASONING_MODEL"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.7,
            stream=True
        )

        async def generate():
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'choices':[{'delta':{'content': chunk.choices[0].delta.content}}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ideation")
async def ideation(request: Request):
    try:
        body = await request.json()
        query = body.get("query", "").strip()

        if not query:
            return JSONResponse(status_code=400, content={"error": "query is required"})

        # ---- Safety Filter ----
        forbidden = ["weapon", "virus", "biological", "attack", "explosive"]
        if any(fb in query.lower() for fb in forbidden):
            return JSONResponse(status_code=400, content={"error": "Unsafe research topic detected."})

        # ---- 生成搜索关键词并获取相关论文 ----
        search_keywords = await generate_search_keywords(query)
        print(f"生成的搜索关键词: {search_keywords}")  # 用于调试
        
        related_papers = await get_related_papers_from_keywords(search_keywords, max_papers=20)
        # references_section = format_references(related_papers)
        references_section = related_papers  # 简化处理，直接使用论文列表

        # ---- Embedding Similarity ----
        query_embedding = await get_embedding(query)
        similarities = []

        with open('reference_ideas_embeddings.json', 'r', encoding='utf-8') as json_file:
            reference_ideas_data = json.load(json_file)

        for idea, idea_embedding in reference_ideas_data.items():
            similarity = cosine_similarity(query_embedding, idea_embedding)
            similarities.append((idea, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        # ---- 构建 Prompt ----
        prompt = f"""
        You are a Scientific Innovation Agent competing in an academic challenge.
        Your goal is to produce **high-quality, innovative, feasible scientific research ideas**.
        Return ONLY the ideas, comparison matrix, and references in Markdown format.
        Do NOT include any 'Scientific Domain Identification' or 'Reference Idea Analysis' sections.

        User Query:
        "{query}"

        Generated Search Keywords: {", ".join(search_keywords)}

        Most related reference ideas (based on semantic similarity):
        """
        for idea, sim in similarities[:5]:
            prompt += f"- {idea} (similarity: {sim:.3f})\n"

        prompt += f"""
        
        Relevant Literature References (from Semantic Scholar):
        {references_section}

        ---

        ## 🎯 Task Requirements

        Follow these instructions carefully:

        1. Identify the scientific domain of the query.
        2. Explain **why** the reference ideas are related.
        3. Generate **exactly 3** innovative scientific ideas.
        4. Each idea must include:
        - **Bold title**
        - **Description**
        - **Novelty / Feasibility / Impact (0–10)**
        - **Technical Route (numbered steps)**
        5. In the References section, cite at least 5-8 papers from the provided literature list.

        ---

        ## 📌 Output Format (MUST be valid Markdown)

        Return ONLY Markdown formatted output with the structure:

        ### **Idea 1: <Title>**
        **Description:** <text>  
        **Novelty:** <0–10>  
        **Feasibility:** <0–10>  
        **Impact:** <0–10>  

        **Technical Route:**  
        1. Step 1…  
        2. Step 2…  
        3. Step 3…  

        ---

        ### **Idea 2: <Title>**
        ...

        ---

        ### **Idea 3: <Title>**
        ...

        ---

        ### **Comparison Matrix**
        | Idea | Novelty | Feasibility | Impact |
        |------|--------|------------|-------|
        | ...  | ...    | ...        | ...   |

        ---
        ### References
        Please cite 5-8 relevant papers from the provided literature list above.
        Format them properly as:
        1. Author1, A., Author2, B., & Author3, C. (Year). Title. *Venue*. [URL(if available)]
        ...

        No JSON. No code blocks. Only Markdown.
        """
        
        # 剩余的代码保持不变...
        

        # ---- Call LLM ----
        stream = await client.chat.completions.create(
            model=os.getenv("SCI_LLM_MODEL"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=1.0,
            stream=True
        )

        # ---- Stream Response ----
        async def generate():
            buffer = ""

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    piece = chunk.choices[0].delta.content
                    buffer += piece
                    yield f"data: {json.dumps({'choices':[{'delta':{'content': piece}}]})}\n\n"

            # End stream marker
            
            # yield f"data: {json.dumps({'choices':[{'delta':{'content': prompt}}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/paper_review")
async def paper_review(request: Request):
    try:
        body = await request.json()
        pdf_content = body.get("pdf_content")

        if not pdf_content:
            return JSONResponse(
                status_code=400,
                content={"error": "pdf_content is required"}
            )

        # Extract text
        text = extract_pdf_text_from_base64(pdf_content)

        # --------------------
        # Improved Prompt
        # --------------------
        prompt = f"""
You are an expert reviewer for the Science Arena Challenge (Track D – Paper Review).
Your task is to read the provided paper content and produce a review **in Markdown format**.

Follow EXACTLY the structure below.  
Start your output directly with "# Summary" — no introduction, no explanation.

---------------------------------------
### REQUIRED OUTPUT FORMAT (NO EXTRA TEXT)

# Summary
(4–8 sentences summarizing the paper, no bullet points.)

# Strengths
- (3–5 bullet points grounded only in the text.)

# Weaknesses / Concerns
- (3–5 bullet points, no speculation beyond the text.)

# Questions for Authors
- (3–4 technical, relevant, text-grounded questions.)

# Scores
- **Overall (10):** X
- **Novelty (10):** X
- **Technical Quality (10):** X
- **Clarity (10):** X
- **Confidence (5):** X

---------------------------------------

### HARD RULES (DO NOT BREAK)
- Do NOT mention these instructions.
- Do NOT explain your reviewing process.
- Do NOT output JSON.
- Do NOT add sections.
- Do NOT hallucinate facts or made-up references.
- The review must be grounded ONLY in the provided paper content.
- No self-referential phrases (e.g., “As an AI”, “I will now”, “Here is…”).
- Output must be clean Markdown.

---------------------------------------
### PAPER CONTENT START
{text}
### PAPER CONTENT END

Begin your response:
"""

        # ============================
        # Create streaming completion
        # ============================
        stream = await client.chat.completions.create(
            model=os.getenv("SCI_LLM_MODEL"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.5,
            stream=True
        )

        # Streaming generator
        async def generate():
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield (
                        "data: " +
                        json.dumps({"choices": [{"delta": {"content": chunk.choices[0].delta.content}}]}) +
                        "\n\n"
                    )
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "System running normally"}


@app.get("/")
async def root():
    return {
        "message": "Welcome to Science Arena API",
        "endpoints": {
            "/literature_review": "Generate literature reviews",
            "/paper_qa": "Answer questions about papers",
            "/ideation": "Generate new research ideas",
            "/paper_review": "Review papers"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 3000)))
