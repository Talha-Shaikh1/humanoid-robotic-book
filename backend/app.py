import os
import logging
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import cohere
from qdrant_client import QdrantClient

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# ================== ENV ==================
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

COLLECTION_NAME = "humanoid_book"

if not all([COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY]):
    raise RuntimeError("❌ Missing environment variables")

# ================== LOGGING ==================
logging.basicConfig(level=logging.INFO)

# ================== CLIENTS ==================
co = cohere.Client(COHERE_API_KEY)

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=10
)

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=20
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    temperature=0,
    tracing_disabled=True
)

# ================== APP ==================
app = FastAPI(title="Humanoid Robotics Book RAG")

# ================== CORS ==================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://huggingface.co",
        "https://*.huggingface.co"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== GLOBAL ERROR HANDLER ==================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# ================== AGENTS ==================
book_agent = Agent(
    name="HumanoidBookAssistant",
    instructions="""
You are an AI assistant answering questions from a technical book.

Rules:
- Use ONLY the provided context
- If the answer is not found, say exactly:
  "The answer is not available in the book."
"""
)

selection_agent = Agent(
    name="SmartSelectionAssistant",
    instructions="""
You are an AI assistant helping the user understand selected text.

Rules:
- Selected text is PRIMARY
- Related context may be used
- Answer in 3–5 sentences
- Use your own words
- Do NOT quote directly
"""
)

# ================== MODELS ==================
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)

class AskResponse(BaseModel):
    answer: str
    sources: List[str]

class AskSelectionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=300)
    selected_text: str = Field(..., min_length=10, max_length=1500)

class AskSelectionResponse(BaseModel):
    answer: str

# ================== HELPERS ==================
def embed_query(text: str):
    try:
        res = co.embed(
            model="embed-english-v3.0",
            texts=[text],
            input_type="search_query"
        )
        return res.embeddings[0]
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        raise HTTPException(status_code=503, detail="Embedding service unavailable")

def embed_selected_text(text: str):
    try:
        res = co.embed(
            model="embed-english-v3.0",
            texts=[text],
            input_type="search_document"
        )
        return res.embeddings[0]
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        raise HTTPException(status_code=503, detail="Embedding service unavailable")

def search_qdrant(query_vector, limit=5, min_score=0.2):
    try:
        result = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_payload=True
        )
        return [
            p for p in result.points
            if getattr(p, "score", 1.0) >= min_score
        ]
    except Exception as e:
        logging.error(f"Qdrant error: {e}")
        raise HTTPException(status_code=503, detail="Vector search unavailable")

def answer_from_book(question: str, contexts: List[str]) -> str:
    if not contexts:
        return "The answer is not available in the book."

    prompt = f"""
Context:
\"\"\"
{"\n\n".join(contexts)}
\"\"\"

Question:
{question}
"""
    try:
        result = Runner.run_sync(book_agent, prompt, run_config=config)
        return result.final_output
    except Exception as e:
        logging.error(f"LLM error: {e}")
        raise HTTPException(status_code=503, detail="LLM unavailable")

def answer_from_selection(question: str, selected_text: str) -> str:
    query_vector = embed_selected_text(selected_text)
    hits = search_qdrant(query_vector, limit=4)

    book_contexts = [h.payload.get("text", "") for h in hits]

    combined_prompt = f"""
Selected Text:
\"\"\"
{selected_text}
\"\"\"

Related Context:
\"\"\"
{"\n\n".join(book_contexts)}
\"\"\"

Question:
{question}
"""
    try:
        result = Runner.run_sync(selection_agent, combined_prompt, run_config=config)
        return result.final_output
    except Exception as e:
        logging.error(f"LLM error: {e}")
        raise HTTPException(status_code=503, detail="LLM unavailable")

# ================== ROUTES ==================
@app.get("/")
def root():
    return {"status": "RAG backend running 🚀"}

@app.post("/ask", response_model=AskResponse)
def ask_book(payload: AskRequest):
    query_vector = embed_query(payload.question)
    hits = search_qdrant(query_vector)

    contexts = []
    sources = []

    for hit in hits:
        contexts.append(hit.payload.get("text", ""))
        sources.append(hit.payload.get("source_url") or "unknown")

    answer = answer_from_book(payload.question, contexts)

    return {
        "answer": answer,
        "sources": list(set(sources))
    }

@app.post("/ask/selection", response_model=AskSelectionResponse)
def ask_selection(payload: AskSelectionRequest):
    answer = answer_from_selection(
        payload.question,
        payload.selected_text
    )
    return {"answer": answer}
