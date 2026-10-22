"""BM25 sparse retrieval index with thread-safe persistence.

Complements the dense vector store for exact term matching, proper nouns,
codes, and rare vocabulary in Vietnamese agricultural texts.

Uses Reciprocal Rank Fusion (RRF) at the service layer to merge BM25
and vector results — this module only handles BM25 indexing and search.

Persistence: pickle file at BM25_INDEX_PATH (default ./data/bm25_index.pkl).
The index is loaded on startup and updated incrementally on each ingestion.
On server restart, the index is reloaded from disk — no rebuild needed.
"""

import pickle
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lightweight Vietnamese tokenizer.

    Extracts lowercased tokens that include:
    - ASCII alphanumeric characters
    - Vietnamese diacritics (Unicode Latin Extended blocks)
    Tokens shorter than 2 characters are dropped (noise reduction).
    """
    tokens = re.findall(
        r"[a-z0-9\u00c0-\u024f\u1e00-\u1eff]+",
        text.lower(),
    )
    return [t for t in tokens if len(t) > 1]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BM25Entry:
    """One indexed chunk — stores everything needed to build a RetrievalResult."""
    chunk_id: str
    document_id: str   # for delete_by_document_id
    content: str
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

class BM25Index:
    """Thread-safe BM25Okapi index with atomic pickle persistence.

    Internal state (always kept in sync):
        _entries : List[BM25Entry]  — parallel to _corpus
        _corpus  : List[List[str]] — tokenized content for BM25
        _bm25    : BM25Okapi | None — rebuilt whenever corpus changes
    """

    def __init__(self, persist_path: str):
        self._path = Path(persist_path)
        self._lock = threading.RLock()
        self._entries: List[BM25Entry] = []
        self._corpus: List[List[str]] = []
        self._bm25 = None
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load index from disk, or start empty if file missing/corrupt."""
        if not self._path.exists():
            logger.info("BM25: no index file at %s — starting empty.", self._path)
            return
        try:
            with open(self._path, "rb") as f:
                data = pickle.load(f)
            self._entries = data.get("entries", [])
            self._corpus = data.get("corpus", [])
            self._rebuild_model()
            logger.info(
                "BM25: loaded %d entries from %s",
                len(self._entries), self._path,
            )
        except Exception as exc:
            logger.warning("BM25: failed to load index (%s) — starting empty.", exc)
            self._entries = []
            self._corpus = []
            self._bm25 = None

    def _save(self) -> None:
        """Atomically persist index to disk using a temp file + rename."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump({"entries": self._entries, "corpus": self._corpus}, f)
            tmp.replace(self._path)
        except Exception as exc:
            logger.error("BM25: failed to persist index: %s", exc)

    def _rebuild_model(self) -> None:
        """Rebuild the BM25Okapi model from the current corpus."""
        if not self._corpus:
            self._bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._corpus)
        except Exception as exc:
            logger.error("BM25: model rebuild failed: %s", exc)
            self._bm25 = None

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_chunks(self, entries: List[BM25Entry]) -> int:
        """Add new entries to the index.  Skips duplicate chunk_ids.

        Returns the number of newly added entries.
        """
        if not entries:
            return 0
        with self._lock:
            existing_ids = {e.chunk_id for e in self._entries}
            added = 0
            for entry in entries:
                if entry.chunk_id in existing_ids:
                    continue
                tokens = _tokenize(entry.content)
                self._entries.append(entry)
                self._corpus.append(tokens)
                added += 1
            if added:
                self._rebuild_model()
                self._save()
                logger.info(
                    "BM25: added %d chunks (total=%d)", added, len(self._entries)
                )
        return added

    def delete_by_ids(self, chunk_ids: List[str]) -> int:
        """Remove entries whose chunk_id is in *chunk_ids*.

        Returns the number of removed entries.
        """
        if not chunk_ids:
            return 0
        id_set = set(chunk_ids)
        with self._lock:
            keep = [e.chunk_id not in id_set for e in self._entries]
            removed = sum(1 for k in keep if not k)
            if not removed:
                return 0
            self._entries = [e for e, k in zip(self._entries, keep) if k]
            self._corpus = [c for c, k in zip(self._corpus, keep) if k]
            self._rebuild_model()
            self._save()
            logger.info(
                "BM25: removed %d chunks (total=%d)", removed, len(self._entries)
            )
        return removed

    def delete_by_document_id(self, document_id: str) -> int:
        """Remove all entries belonging to *document_id*."""
        with self._lock:
            ids = [e.chunk_id for e in self._entries if e.document_id == document_id]
        return self.delete_by_ids(ids)

    # ------------------------------------------------------------------
    # Read operation
    # ------------------------------------------------------------------

    def search(
        self, query: str, top_k: int
    ) -> List[Tuple[str, float, BM25Entry]]:
        """Return up to *top_k* (chunk_id, bm25_score, entry) tuples.

        Score is a BM25Okapi term-frequency score; higher is more relevant.
        Results with score == 0.0 (no term overlap) are excluded.
        """
        # Take a snapshot of references under the lock so scoring runs outside it
        with self._lock:
            if self._bm25 is None or not self._entries:
                return []
            tokens = _tokenize(query)
            if not tokens:
                return []
            bm25_snap = self._bm25
            entries_snap = list(self._entries)

        start = time.time()
        scores = bm25_snap.get_scores(tokens)
        logger.debug("BM25 scoring %.4fs for %d docs", time.time() - start, len(scores))

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for i, score in ranked[:top_k]:
            if score <= 0.0:
                break
            if i < len(entries_snap):
                results.append((entries_snap[i].chunk_id, float(score), entries_snap[i]))
        return results

    def count(self) -> int:
        with self._lock:
            return len(self._entries)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_bm25_instance: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """Return the singleton BM25Index (lazy-loaded from disk on first call)."""
    global _bm25_instance
    if _bm25_instance is None:
        from app.core.config import get_settings
        settings = get_settings()
        _bm25_instance = BM25Index(persist_path=settings.BM25_INDEX_PATH)
    return _bm25_instance
