import io
import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from services.embedder import embed, emb_to_str
from db.database import get_conn

router = APIRouter(prefix="/ingest", tags=["ingest"])


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    """
    Splits text into overlapping chunks.
    Tries to break at paragraph → newline → sentence → word boundaries.
    """
    chunks = []
    text = text.strip()
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Try to break cleanly at a natural boundary
            for delimiter in ["\n\n", "\n", ". ", " "]:
                pos = text.rfind(delimiter, start, end)
                if pos > start:
                    end = pos + len(delimiter)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


async def store_chunks(filename: str, chunks: list[str], metadata: dict = {}):
    # Embed all chunks first (async), then batch insert (sync)
    embeddings = []
    for chunk in chunks:
        emb = await embed(chunk)
        embeddings.append(emb)

    conn = get_conn()
    cur = conn.cursor()

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        cur.execute("""
            INSERT INTO documents (filename, chunk_index, content, embedding, metadata)
            VALUES (%s, %s, %s, %s::vector, %s)
        """, (filename, i, chunk, emb_to_str(emb), json.dumps(metadata)))

    conn.commit()
    cur.close()
    conn.close()


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

class TextIngestRequest(BaseModel):
    content:  str
    filename: str
    metadata: dict = {}


@router.post("/text")
async def ingest_text(body: TextIngestRequest):
    """Ingest raw text directly (useful for pasting runbooks, notes, etc.)"""
    chunks = chunk_text(body.content)
    if not chunks:
        raise HTTPException(status_code=400, detail="No content to ingest.")
    await store_chunks(body.filename, chunks, body.metadata)
    return {"status": "ok", "filename": body.filename, "chunks_stored": len(chunks)}


@router.post("/file")
async def ingest_file(file: UploadFile = File(...)):
    """Upload a .txt, .md, or .pdf file to the knowledge base."""
    raw = await file.read()
    filename = file.filename

    if filename.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            text = "\n\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF parse error: {e}")

    elif filename.endswith((".txt", ".md")):
        text = raw.decode("utf-8", errors="ignore")

    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use .pdf, .txt, or .md"
        )

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="File appears to be empty.")

    await store_chunks(filename, chunks)
    return {"status": "ok", "filename": filename, "chunks_stored": len(chunks)}


@router.get("/list")
def list_documents():
    """List all ingested documents and their chunk counts."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT filename, COUNT(*) as chunks, MAX(created_at) as ingested_at
        FROM documents
        GROUP BY filename
        ORDER BY ingested_at DESC;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {"filename": r[0], "chunks": r[1], "ingested_at": str(r[2])}
        for r in rows
    ]