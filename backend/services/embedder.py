import httpx
import os
import asyncio

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
EMBED_MODEL = "all-minilm"
EMBED_DIM = 384

MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds, doubles each retry


async def embed(text: str) -> list[float]:
    """
    Embed a single string via Ollama.
    num_gpu=0 forces this model to run on CPU only, so it never
    competes with qwen/deepseek for the 6GB of VRAM. The model is
    tiny enough that CPU inference is still fast.
    Retries with backoff on transient 500s from Ollama mid-swap.
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={
                        "model": EMBED_MODEL,
                        "prompt": text,
                        "options": {"num_gpu": 0}
                    }
                )
                resp.raise_for_status()
                return resp.json()["embedding"]
        except httpx.HTTPStatusError as e:
            last_error = e
            print(f"[EMBEDDER] attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))

    raise last_error


async def embed_batch(texts: list[str]) -> list[list[float]]:
    results = []
    for text in texts:
        emb = await embed(text)
        results.append(emb)
    return results


def emb_to_str(emb: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in emb) + "]"