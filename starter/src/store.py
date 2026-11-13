"""Store and search vectors using PostgreSQL + pgvector."""
import json
from typing import Any
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        self.database_url = database_url
        self.dimensions = dimensions
        self.conn = psycopg2.connect(self.database_url)
        self.conn.autocommit = True
        register_vector(self.conn)

    def create_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id SERIAL PRIMARY KEY,
                chunk_id VARCHAR(100) NOT NULL,
                strategy VARCHAR(50) NOT NULL,
                document_id VARCHAR(100) NOT NULL,
                source TEXT NOT NULL,
                page INT NOT NULL,
                text TEXT NOT NULL,
                metadata JSONB NOT NULL,
                embedding vector({self.dimensions}) NOT NULL,
                UNIQUE (strategy, chunk_id)
            );
            CREATE INDEX IF NOT EXISTS idx_rag_chunks_strat ON rag_chunks (strategy);
            CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc ON rag_chunks (document_id);
            CREATE INDEX IF NOT EXISTS idx_rag_chunks_page ON rag_chunks (page);
            """)

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"Length mismatch: {len(chunks)} chunks vs {len(vectors)} vectors")

        data = []
        for c, v in zip(chunks, vectors):
            data.append((
                c.chunk_id,
                strategy,
                c.document_id,
                c.source,
                c.page,
                c.text,
                json.dumps(c.metadata),
                v
            ))

        with self.conn.cursor() as cur:
            # Delete any existing chunks for this strategy to ensure clean idempotent runs
            cur.execute("DELETE FROM rag_chunks WHERE strategy = %s;", (strategy,))
            
            insert_query = """
            INSERT INTO rag_chunks (chunk_id, strategy, document_id, source, page, text, metadata, embedding)
            VALUES %s
            ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                source = EXCLUDED.source,
                page = EXCLUDED.page,
                text = EXCLUDED.text,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding;
            """
            execute_values(cur, insert_query, data, template="(%s, %s, %s, %s, %s, %s, %s, %s::vector)")

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        where_clauses = ["strategy = %s"]
        params: list[Any] = [query_vector, strategy]

        if filters:
            for k, val in filters.items():
                if k in ["document_id", "source"]:
                    where_clauses.append(f"{k} = %s")
                    params.append(val)
                elif k == "page":
                    where_clauses.append("page = %s")
                    params.append(int(val))
                else:
                    where_clauses.append("metadata->>%s = %s")
                    params.extend([k, str(val)])

        params.extend([query_vector, top_k])
        where_sql = " AND ".join(where_clauses)

        query = f"""
        SELECT chunk_id, document_id, source, page, text, metadata, 1 - (embedding <=> %s::vector) AS score
        FROM rag_chunks
        WHERE {where_sql}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """

        with self.conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        results = []
        for r in rows:
            meta = r[5]
            if isinstance(meta, str):
                meta = json.loads(meta)
            results.append({
                "chunk_id": r[0],
                "document_id": r[1],
                "source": r[2],
                "page": r[3],
                "text": r[4],
                "metadata": meta,
                "score": float(r[6])
            })
        return results

    def close(self) -> None:
        if self.conn and not self.conn.closed:
            self.conn.close()
