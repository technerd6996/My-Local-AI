import os
import psycopg2
from psycopg2.extras import RealDictCursor


def get_conn():
    return psycopg2.connect(os.getenv("POSTGRES_URL"))


def init_db():
    """
    Run once at app startup.
    Creates the documents table and vector index if they don't exist.
    """
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id          SERIAL PRIMARY KEY,
            filename    TEXT NOT NULL,
            chunk_index INT  NOT NULL,
            content     TEXT NOT NULL,
            embedding   vector(384),
            metadata    JSONB NOT NULL DEFAULT '{}',
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)

    # HNSW index: fast approximate nearest neighbour search
    # m=16 (connections per layer), ef_construction=64 (build quality)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
        ON documents
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("[DB] Schema ready.")