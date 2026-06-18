import psycopg2
from psycopg2.extras import RealDictCursor
from services.embedder import embed, emb_to_str

SIMILARITY_THRESHOLD = 0.30


async def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """
    Embed the query, find top_k most similar document chunks,
    filter by similarity threshold, return as list of dicts.
    """
    query_emb = await embed(query)
    query_emb_str = emb_to_str(query_emb)

    from db.database import get_conn
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT
            content,
            filename,
            chunk_index,
            1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        WHERE 1 - (embedding <=> %s::vector) >= %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_emb_str, query_emb_str, SIMILARITY_THRESHOLD, query_emb_str, top_k))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [dict(r) for r in rows]


def build_context_block(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a clean system-level context block
    that gets injected before the model call.
    """
    if not chunks:
        return ""

    lines = [
        "Use the following context from the knowledge base to assist your answer.",
        "If the context is not relevant to the question, ignore it.\n",
    ]
    for i, chunk in enumerate(chunks, 1):
        sim_pct = int(chunk["similarity"] * 100)
        lines.append(f"[Context {i} | source: {chunk['filename']} | relevance: {sim_pct}%]")
        lines.append(chunk["content"])
        lines.append("")

    return "\n".join(lines)