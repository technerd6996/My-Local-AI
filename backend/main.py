import os
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

from db.database import init_db, init_query_log, log_query, get_conn
from services.classifier import classify, precompute_examples
from services.rag import retrieve, build_context_block
from services.validator import validate_answer, escalate_stub
from routers.ingest import router as ingest_router

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Initialising database schema...")
    init_db()
    init_query_log()

    print("[STARTUP] Precomputing classifier embeddings...")
    await precompute_examples()

    print("[STARTUP] Router ready.")
    yield
    print("[SHUTDOWN] Bye.")


app = FastAPI(title="Hybrid AI Router - Phase 3", lifespan=lifespan)
app.include_router(ingest_router)


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model:       Optional[str]   = "hybrid-router"
    messages:    list[Message]
    stream:      Optional[bool]  = True
    temperature: Optional[float] = 0.7
    max_tokens:  Optional[int]   = None


def inject_rag_context(messages: list[dict], context_block: str) -> list[dict]:
    if not context_block:
        return messages
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = context_block + "\n\n" + messages[0]["content"]
    else:
        messages = [{"role": "system", "content": context_block}] + messages
    return messages


def build_openai_response(content: str, model_label: str) -> dict:
    """Non-streaming OpenAI-format response, used when the client didn't ask to stream."""
    return {
        "id": "chatcmpl-router",
        "object": "chat.completion",
        "created": 0,
        "model": model_label,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }]
    }


async def fake_stream_single_chunk(content: str, model_label: str):
    """
    Open WebUI expects SSE when stream=True. We can't token-stream a
    validated answer (we needed the full thing before deciding whether
    to show it), so this sends the whole validated answer as ONE chunk
    in valid SSE format. No progressive typing, but compatible with
    what the client expects.
    """
    chunk = {
        "id": "chatcmpl-router", "object": "chat.completion.chunk", "created": 0,
        "model": model_label,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}]
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    done = {
        "id": "chatcmpl-router", "object": "chat.completion.chunk", "created": 0,
        "model": model_label,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health():
    return {"status": "ok", "phase": 3}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "hybrid-router", "object": "model", "created": 1700000000, "owned_by": "local-router"}]}


@app.get("/logs/recent")
def logs_recent(limit: int = 10):
    """Quick way to confirm logging is working without opening psql directly."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT created_at, query, tier, drafting_model, validation_method,
               validation_passed, validation_detail, escalation_flagged, latency_ms
        FROM query_log ORDER BY created_at DESC LIMIT %s;
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    cols = ["created_at", "query", "tier", "drafting_model", "validation_method",
            "validation_passed", "validation_detail", "escalation_flagged", "latency_ms"]
    return [dict(zip(cols, [str(v) for v in row])) for row in rows]


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    start_time = time.time()
    messages = [m.model_dump() for m in request.messages]
    last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    ollama_url = f"{OLLAMA_BASE_URL}/v1/chat/completions"

    # ── Path A: Open WebUI internal housekeeping (tags, titles, follow-ups) ──
    # No validation needed here — stays fast and streamed, same as Phase 2.
    if last_user_msg.strip().startswith("### Task:"):
        print("[ROUTER] internal Open WebUI task, bypassing classifier/RAG/validation")
        payload = {"model": "qwen3:8b", "messages": messages, "stream": request.stream, "temperature": request.temperature}

        if request.stream:
            async def stream_ollama():
                async with httpx.AsyncClient(timeout=180.0) as client:
                    async with client.stream("POST", ollama_url, json=payload) as resp:
                        async for chunk in resp.aiter_text():
                            yield chunk
            return StreamingResponse(stream_ollama(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(ollama_url, json=payload)
                return JSONResponse(resp.json())

    # ── Path B: real user query — classify, retrieve, generate FULLY, validate, log ──
    tier, model, confidence = await classify(messages)
    rag_chunks = await retrieve(last_user_msg, top_k=3)
    context_block = build_context_block(rag_chunks)
    gen_messages = inject_rag_context(list(messages), context_block)

    print(f"[ROUTER] tier={tier} | model={model} | confidence={confidence:.2f} | rag_chunks={len(rag_chunks)}")
    print(f"[ROUTER] query={last_user_msg[:80]}...")

    # Non-streaming generation — we need the complete answer before we
    # can validate it. This is the architectural cost of "no room for fails".
    print(f"[TIMING] starting draft generation with {model}...")
    gen_payload = {"model": model, "messages": gen_messages, "stream": False, "temperature": request.temperature}
    async with httpx.AsyncClient(timeout=180.0) as client:
        gen_resp = await client.post(ollama_url, json=gen_payload)
        gen_resp.raise_for_status()
        local_answer = gen_resp.json()["choices"][0]["message"]["content"]
    print(f"[TIMING] draft generation done after {time.time() - start_time:.1f}s total")

    print(f"[TIMING] starting validation...")
    validation = await validate_answer(last_user_msg, local_answer, model)
    print(f"[TIMING] validation done after {time.time() - start_time:.1f}s total")

    if validation["valid"]:
        final_answer = local_answer
        escalation_flagged = False
        print(f"[VALIDATOR] PASS via {validation['method']}")
    else:
        final_answer = escalate_stub(last_user_msg, local_answer, validation)
        escalation_flagged = True
        print(f"[VALIDATOR] FAIL via {validation['method']}: {validation.get('error')}")

    latency_ms = int((time.time() - start_time) * 1000)

    log_query(
        query=last_user_msg, tier=tier, drafting_model=model, confidence=confidence,
        rag_chunks=len(rag_chunks), validation_method=validation["method"],
        validation_passed=validation["valid"], validation_detail=validation.get("error"),
        escalation_flagged=escalation_flagged, latency_ms=latency_ms,
    )

    model_label = f"{model} [{tier} | conf:{confidence:.2f} | rag:{len(rag_chunks)} | valid:{validation['valid']}]"

    if request.stream:
        return StreamingResponse(fake_stream_single_chunk(final_answer, model_label), media_type="text/event-stream")
    else:
        return JSONResponse(build_openai_response(final_answer, model_label))