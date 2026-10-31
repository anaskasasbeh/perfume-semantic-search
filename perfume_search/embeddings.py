"""Text → vector with the Gemini embedding API.

* Perfumes are embedded once at build time with task type RETRIEVAL_DOCUMENT.
* A visitor's query is embedded at request time with task type RETRIEVAL_QUERY.
* Vectors shorter than 3072 dimensions must be L2-normalised by the caller,
  which is also what makes a plain dot product equal to cosine similarity.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import numpy as np

from . import config

log = logging.getLogger(__name__)

BATCH_SIZE = 100  # small batches keep the result order reliable
# The free tier counts every text in a batch as one request, so the
# per-minute budget is measured in texts, not in API calls.
TEXTS_PER_MINUTE_BUDGET = 100
TOKENS_PER_MINUTE_BUDGET = 25_000  # stays under the free tier's 30k TPM
MAX_RETRIES = 6


class EmbeddingError(RuntimeError):
    """The embedding service could not return a vector."""


class DailyQuotaExceeded(EmbeddingError):
    """The free tier's daily request limit was reached."""


class Embedder(Protocol):
    """Anything that turns text into vectors (Gemini in production, fakes in tests)."""

    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale every row to length 1 (safe for zero rows)."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # ~4 characters per token for English


def _is_daily_quota(error: Exception) -> bool:
    return "perday" in str(error).replace(" ", "").lower()


def _status_code(error: Exception) -> int | None:
    return getattr(error, "code", None)


class GeminiEmbedder:
    """Thin wrapper around ``google-genai`` with batching, throttling and a disk cache."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = config.EMBEDDING_MODEL,
        dim: int = config.EMBEDDING_DIM,
        timeout_ms: int = config.QUERY_TIMEOUT_MS,
        cache_dir: Path = config.CACHE_DIR,
    ) -> None:
        api_key = api_key or config.GEMINI_API_KEY
        if not api_key:
            raise EmbeddingError("GEMINI_API_KEY is not set (see .env.example).")
        from google import genai  # imported lazily so tests and fallback mode stay light
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms))
        self.model = model
        self.dim = dim
        self._cache_path = cache_dir / f"embeddings_{model}_{dim}.jsonl"
        self._window: deque[tuple[float, int, int]] = deque()  # (timestamp, tokens, texts) of recent calls

    # ---- public API -------------------------------------------------------
    def embed_query(self, text: str) -> np.ndarray:
        """One fast call for a visitor's query. No retries: the caller falls back instead."""
        try:
            vectors = self._call([text], task_type="RETRIEVAL_QUERY")
        except Exception as error:  # network, quota, bad key, timeout …
            raise EmbeddingError(f"Query embedding failed: {error}") from error
        return vectors[0]

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Embed several queries (used at build time for the example chips)."""
        return self._embed_with_cache(texts, task_type="RETRIEVAL_QUERY")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed the whole catalog; resumable thanks to the on-disk cache."""
        return self._embed_with_cache(texts, task_type="RETRIEVAL_DOCUMENT")

    # ---- internals --------------------------------------------------------
    def _call(self, texts: list[str], task_type: str) -> np.ndarray:
        config_ = self._types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        response = self._client.models.embed_content(model=self.model, contents=texts, config=config_)
        vectors = np.array([embedding.values for embedding in response.embeddings], dtype=np.float32)
        if vectors.shape != (len(texts), self.dim):
            raise EmbeddingError(f"Unexpected embedding shape {vectors.shape}, expected {(len(texts), self.dim)}")
        return normalize(vectors)

    def _key(self, text: str, task_type: str) -> str:
        return hashlib.sha1(f"{self.model}|{self.dim}|{task_type}|{text}".encode()).hexdigest()

    def _read_cache(self) -> dict[str, list[float]]:
        if not self._cache_path.exists():
            return {}
        cache: dict[str, list[float]] = {}
        with self._cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    cache[item["key"]] = item["vector"]
        return cache

    def _append_cache(self, keys: Iterable[str], vectors: np.ndarray) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("a", encoding="utf-8") as handle:
            for key, vector in zip(keys, vectors, strict=True):
                handle.write(json.dumps({"key": key, "vector": [round(float(x), 7) for x in vector]}) + "\n")

    def _throttle(self, tokens: int, texts: int) -> None:
        """Sleep just enough to stay inside the per-minute budgets."""
        while True:
            now = time.monotonic()
            while self._window and now - self._window[0][0] > 60:
                self._window.popleft()
            used_tokens = sum(item[1] for item in self._window)
            used_texts = sum(item[2] for item in self._window)
            if used_texts + texts <= TEXTS_PER_MINUTE_BUDGET and used_tokens + tokens <= TOKENS_PER_MINUTE_BUDGET:
                self._window.append((now, tokens, texts))
                return
            wait = 60 - (now - self._window[0][0]) + 1
            log.info("Free-tier minute budget used, waiting %.0f s …", wait)
            time.sleep(max(wait, 1))

    def _embed_with_cache(self, texts: list[str], task_type: str) -> np.ndarray:
        cache = self._read_cache()
        keys = [self._key(text, task_type) for text in texts]
        todo, seen = [], set()
        for i, key in enumerate(keys):  # identical texts are embedded only once
            if key not in cache and key not in seen:
                seen.add(key)
                todo.append(i)
        log.info("%d texts: %d cached, %d to embed (%s)", len(texts), len(texts) - len(todo), len(todo), task_type)

        for start in range(0, len(todo), BATCH_SIZE):
            batch_idx = todo[start : start + BATCH_SIZE]
            batch = [texts[i] for i in batch_idx]
            self._throttle(sum(_estimate_tokens(t) for t in batch), len(batch))
            vectors = self._call_with_retry(batch, task_type)
            batch_keys = [keys[i] for i in batch_idx]
            self._append_cache(batch_keys, vectors)
            cache.update({k: v.tolist() for k, v in zip(batch_keys, vectors, strict=True)})
            log.info("Embedded %d / %d", min(start + BATCH_SIZE, len(todo)), len(todo))

        return normalize(np.array([cache[key] for key in keys], dtype=np.float32))

    def _call_with_retry(self, batch: list[str], task_type: str) -> np.ndarray:
        delay = 15.0
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._call(batch, task_type)
            except Exception as error:
                if _is_daily_quota(error):
                    raise DailyQuotaExceeded(
                        "Daily free quota reached (it resets at midnight Pacific Time). "
                        "Progress is saved; run the same command again after the reset."
                    ) from error
                retryable = _status_code(error) in (408, 429, 500, 502, 503, 504) or isinstance(error, TimeoutError)
                if not retryable or attempt == MAX_RETRIES:
                    raise EmbeddingError(f"Embedding batch failed: {error}") from error
                log.warning("Attempt %d failed (%s). Retrying in %.0f s …", attempt, _status_code(error), delay)
                time.sleep(delay)
                delay = min(delay * 2, 120)
        raise EmbeddingError("unreachable")
