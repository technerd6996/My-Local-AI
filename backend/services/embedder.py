import httpx
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
EMBED_MODEL = "all-minilm"
EMBED_DIM = 384


async def embed(text: str) -> list[float]:
    """
    Embed a single string via Ollama.
    Returns a 384-dimension float list.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text}
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings sequentially.
    Used at startup to precompute classifier example embeddings.
    """
    results = []
    for text in texts:
        emb = await embed(text)
        results.append(emb)
    return results


def emb_to_str(emb: list[float]) -> str:
    """
    Convert embedding list to pgvector-compatible string.
    Example: [0.12, -0.34, ...] → '[0.120000,-0.340000,...]'
    """
    return "[" + ",".join(f"{x:.6f}" for x in emb) + "]"