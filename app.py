import os
import json
import base64
from typing import List
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from openai import AsyncOpenAI
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
import numpy as np

load_dotenv()

app = FastAPI(title="Science Arena Challenge API")

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

        # ---- Reference ideas ----

        # ---- Embedding Similarity ----
        query_embedding = await get_embedding(query)
        similarities = []

        with open('reference_ideas_embeddings.json', 'r', encoding='utf-8') as json_file:
            reference_ideas_data = json.load(json_file)

        # for idea in reference_ideas:
        #     idea_embedding = await get_embedding(idea)
        #     similarity = cosine_similarity(query_embedding, idea_embedding)
        #     similarities.append((idea, similarity))
        for idea, idea_embedding in reference_ideas_data.items():
            similarity = cosine_similarity(query_embedding, idea_embedding)
            similarities.append((idea, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        # ---- Build Prompt ----
        prompt = f"""
        You are a Scientific Innovation Agent competing in an academic challenge.
        Your goal is to produce **high-quality, innovative, feasible scientific research ideas**.
        Return ONLY the ideas, comparison matrix, and references in Markdown format.
        Do NOT include any 'Scientific Domain Identification' or 'Reference Idea Analysis' sections.

        User Query:
        "{query}"

        Most related reference ideas (based on semantic similarity):
        """
        for idea, sim in similarities[:5]:
            prompt += f"- {idea} (similarity: {sim:.3f})\n"

        prompt += """
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
        1. ...
        2. ...
        3. ...
        No JSON. No code blocks. Only Markdown.


        """
        

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
