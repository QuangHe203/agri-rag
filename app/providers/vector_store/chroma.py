"""ChromaDB vector store provider with persistent local storage.

Compatible with ChromaDB >= 1.0.0.
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb

from app.core.config import get_settings
from app.core.exceptions import VectorStoreUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkDocument:
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class RetrievalResult:
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    score: float


class ChromaVectorStore:
    """Persistent ChromaDB vector store for agriculture knowledge."""

    def __init__(self):
        settings = get_settings()
        import os
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        try:
            # ChromaDB 1.x removed the `settings` kwarg from PersistentClient.
            # Telemetry can be disabled via env var ANONYMIZED_TELEMETRY=false.
            os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
            self._client = chromadb.PersistentClient(
                path=settings.VECTOR_DB_PATH,
            )
            self._collection = self._client.get_or_create_collection(
                name=settings.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB collection '%s' ready (path=%s)",
                settings.COLLECTION_NAME,
                settings.VECTOR_DB_PATH,
            )
        except Exception as exc:
            logger.error("Failed to initialize ChromaDB: %s", exc)
            raise VectorStoreUnavailableError(
                "Không thể khởi tạo vector database."
            ) from exc

    def add_chunks(self, chunks: List[ChunkDocument]) -> None:
        """Add (or upsert) chunks into the vector store."""
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        embeddings = [c.embedding for c in chunks]

        start = time.time()
        try:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            elapsed = time.time() - start
            logger.info("Upserted %d chunks into vector store in %.2fs", len(chunks), elapsed)
        except Exception as exc:
            logger.error("Failed to upsert chunks: %s", exc)
            raise VectorStoreUnavailableError("Không thể lưu vector vào database.") from exc

    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        min_score: float,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Query the vector store and return top-K results above the min_score threshold.

        ChromaDB with cosine space returns distances in [0, 2] (0 = identical).
        We convert to similarity score: score = 1 - distance/2, giving [0, 1].

        ChromaDB 1.x requires n_results <= collection size; we guard against that.
        """
        # Guard: ChromaDB 1.x raises an error if n_results > collection size
        try:
            total = self._collection.count()
        except Exception as exc:
            logger.error("Failed to count collection: %s", exc)
            raise VectorStoreUnavailableError("Truy vấn vector database thất bại.") from exc

        if total == 0:
            logger.info("Vector store is empty, returning no results")
            return []

        n_results = min(top_k, total)

        start = time.time()
        try:
            kwargs: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where

            results = self._collection.query(**kwargs)
            elapsed = time.time() - start
            logger.info("Vector store query completed in %.2fs", elapsed)
        except Exception as exc:
            logger.error("Vector store query failed: %s", exc)
            raise VectorStoreUnavailableError("Truy vấn vector database thất bại.") from exc

        retrieval_results: List[RetrievalResult] = []
        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        seen_contents: set[str] = set()

        for chunk_id, doc, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
            # Convert cosine distance [0,2] → similarity score [0,1]
            score = max(0.0, 1.0 - dist / 2.0)
            if score < min_score:
                continue
            # Deduplicate near-identical content
            content_hash = hashlib.md5(doc[:200].encode()).hexdigest()
            if content_hash in seen_contents:
                continue
            seen_contents.add(content_hash)
            retrieval_results.append(
                RetrievalResult(chunk_id=chunk_id, content=doc, metadata=meta or {}, score=score)
            )

        retrieval_results.sort(key=lambda r: r.score, reverse=True)
        return retrieval_results

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all vectors belonging to a document. Returns the number of deleted items."""
        try:
            existing = self._collection.get(where={"document_id": document_id})
            ids = existing.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
                logger.info(
                    "Deleted %d vectors for document_id=%s", len(ids), document_id
                )
            return len(ids)
        except Exception as exc:
            logger.error("Failed to delete vectors for document %s: %s", document_id, exc)
            raise VectorStoreUnavailableError(
                "Không thể xóa vector từ database."
            ) from exc

    def is_available(self) -> bool:
        try:
            self._collection.count()
            return True
        except Exception:
            return False


_chroma_instance: Optional[ChromaVectorStore] = None


def get_vector_store() -> ChromaVectorStore:
    """Return a singleton ChromaVectorStore."""
    global _chroma_instance
    if _chroma_instance is None:
        _chroma_instance = ChromaVectorStore()
    return _chroma_instance
