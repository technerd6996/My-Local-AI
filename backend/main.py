import os
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

from db.database import init_db
from services.classifier import classify, precompute_examples
from services.rag import retrieve, build_context_block
from routers.ingest import router as ingest_router

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")


# ─────────────────────────────────────────────
# Startup / shutdown
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Initialising database schema...")
    init_db()

    print("[STARTUP] Precomputing classifier embeddings...")
    await precompute_examples()

    print("[STARTUP] Router ready.")
    yield
    print("[SHUTDOWN] Bye.")


app = FastAPI(title="Hybrid AI Router - Phase 2", lifespan=lifespan)
app.include_router(ingest_router)


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model:       Optional[str]   = "hybrid-router"
    messages:    list[Message]
    stream:      Optional[bool]  = True
    temperature: Optional[float] = 0.7
    max_tokens:  Optional[int]   = None


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def inject_rag_context(messages: list[dict], context_block: str) -> list[dict]:
    """
    Prepends RAG context as a system message.
    If a system message already exists, appends context to it.
    """
    if not context_block:
        return messages

    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = context_block + "\n\n" + messages[0]["content"]
    else:
        messages = [{"role": "system", "content": context_block}] + messages

    return messages


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "phase": 2}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id":         "hybrid-router",
            "object":     "model",
            "created":    1700000000,
            "owned_by":   "local-router"
        }]
    }


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    messages = [m.model_dump() for m in request.messages]

    # 1. Classify the query
    tier, model, confidence = await classify(messages)

    # 2. Retrieve relevant RAG chunks
    last_user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    rag_chunks = await retrieve(last_user_msg, top_k=3)
    context_block = build_context_block(rag_chunks)

    # 3. Inject RAG context into messages
    messages = inject_rag_context(messages, context_block)

    # 4. Log routing decision
    print(f"[ROUTER] tier={tier} | model={model} | confidence={confidence:.2f} | rag_chunks={len(rag_chunks)}")
    print(f"[ROUTER] query={last_user_msg[:80]}...")

    # 5. Forward to Ollama
    ollama_url = f"{OLLAMA_BASE_URL}/v1/chat/completions"
    payload = {
        "model":       model,
        "messages":    messages,
        "stream":      request.stream,
        "temperature": request.temperature,
    }
    if request.max_tokens:
        payload["max_tokens"] = request.max_tokens

    if request.stream:
        async def stream_ollama():
            async with httpx.AsyncClient(timeout=180.0) as client:
                async with client.stream("POST", ollama_url, json=payload) as resp:
                    first = True
                    async for chunk in resp.aiter_text():
                        if first and chunk.startswith("data:"):
                            try:
                                data = json.loads(chunk[5:].strip())
                                data["model"] = f"{model} [{tier} | conf:{confidence:.2f} | rag:{len(rag_chunks)}]"
                                yield f"data: {json.dumps(data)}\n\n"
                                first = False
                                continue
                            except Exception:
                                pass
                        yield chunk
                        first = False

        return StreamingResponse(
            stream_ollama(),
            media_type="text/event-stream",
            headers={
                "X-Router-Tier":       tier,
                "X-Router-Model":      model,
                "X-Router-Confidence": str(round(confidence, 2)),
                "X-RAG-Chunks":        str(len(rag_chunks)),
            }
        )

    else:
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(ollama_url, json=payload)
            data = resp.json()
            if "model" in data:
                data["model"] = f"{model} [{tier}]"
            return JSONResponse(data)