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

    def init_query_log():
        """
        Called once at startup. Creates the table that every routing +
        validation decision gets logged to — this is the data source
        Phase 2's dashboard goal (escalation rate, cost saved, accuracy)
        will eventually read from.
        """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            query TEXT,
            tier TEXT,
            drafting_model TEXT,
            confidence REAL,
            rag_chunks INT,
            validation_method TEXT,
            validation_passed BOOLEAN,
            validation_detail TEXT,
            escalation_flagged BOOLEAN,
            latency_ms INT
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("[DB] query_log table ready.")


def init_query_log():
    """
    Called once at startup. Creates the table that every routing +
    validation decision gets logged to — this is the data source
    Phase 2's dashboard goal (escalation rate, cost saved, accuracy)
    will eventually read from.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            query TEXT,
            tier TEXT,
            drafting_model TEXT,
            confidence REAL,
            rag_chunks INT,
            validation_method TEXT,
            validation_passed BOOLEAN,
            validation_detail TEXT,
            escalation_flagged BOOLEAN,
            latency_ms INT
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("[DB] query_log table ready.")


def log_query(query, tier, drafting_model, confidence, rag_chunks,
              validation_method, validation_passed, validation_detail,
              escalation_flagged, latency_ms):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO query_log
        (query, tier, drafting_model, confidence, rag_chunks,
         validation_method, validation_passed, validation_detail,
         escalation_flagged, latency_ms)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (query, tier, drafting_model, confidence, rag_chunks,
          validation_method, validation_passed, validation_detail,
          escalation_flagged, latency_ms))
    conn.commit()
    cur.close()
    conn.close()


def log_query(query, tier, drafting_model, confidence, rag_chunks,
              validation_method, validation_passed, validation_detail,
              escalation_flagged, latency_ms):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO query_log
        (query, tier, drafting_model, confidence, rag_chunks,
         validation_method, validation_passed, validation_detail,
         escalation_flagged, latency_ms)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (query, tier, drafting_model, confidence, rag_chunks,
          validation_method, validation_passed, validation_detail,
          escalation_flagged, latency_ms))
    conn.commit()
    cur.close()
    conn.close()