"""GeminiEmbedder logic (batching, cache, retries) with the network call replaced."""

import numpy as np
import pytest

import perfume_search.embeddings as emb
from perfume_search.embeddings import DailyQuotaExceeded, EmbeddingError, GeminiEmbedder


class FakeAPIError(Exception):
    def __init__(self, code, message=""):
        super().__init__(f"{code} {message}")
        self.code = code


@pytest.fixture()
def embedder(tmp_path, monkeypatch):
    clock = {"now": 0.0}  # fake clock: sleeping just moves time forward

    def fake_sleep(seconds):
        clock["now"] += seconds

    monkeypatch.setattr(emb.time, "sleep", fake_sleep)
    monkeypatch.setattr(emb.time, "monotonic", lambda: clock["now"])
    e = GeminiEmbedder(api_key="test-key", dim=4, cache_dir=tmp_path)
    e.calls = []

    def fake_call(texts, task_type):
        e.calls.append((list(texts), task_type))
        return emb.normalize(np.array([[len(t), 1, 0, 0] for t in texts], dtype=np.float32))

    monkeypatch.setattr(e, "_call", fake_call)
    return e


def test_documents_are_batched_and_cached(embedder):
    texts = [f"perfume {i}" for i in range(250)]
    first = embedder.embed_documents(texts)
    assert first.shape == (250, 4)
    assert [len(batch) for batch, _ in embedder.calls] == [100, 100, 50]
    assert np.allclose(np.linalg.norm(first, axis=1), 1)
    embedder.calls.clear()
    second = embedder.embed_documents(texts)  # resumed run: everything comes from the cache
    assert embedder.calls == [] and np.allclose(first, second)


def test_identical_texts_are_embedded_once(embedder):
    vectors = embedder.embed_documents(["same text", "other", "same text"])
    assert [len(batch) for batch, _ in embedder.calls] == [2]
    assert np.allclose(vectors[0], vectors[2])


def test_rate_limit_is_retried(embedder, monkeypatch):
    attempts = {"n": 0}
    real_call = embedder._call

    def flaky(texts, task_type):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise FakeAPIError(429, "RESOURCE_EXHAUSTED per minute")
        return real_call(texts, task_type)

    monkeypatch.setattr(embedder, "_call", flaky)
    assert embedder.embed_documents(["a", "b"]).shape == (2, 4)
    assert attempts["n"] == 3


def test_daily_quota_stops_with_clear_error(embedder, monkeypatch):
    def quota(texts, task_type):
        raise FakeAPIError(429, "Quota exceeded for metric EmbedContentRequestsPerDay")

    monkeypatch.setattr(embedder, "_call", quota)
    with pytest.raises(DailyQuotaExceeded):
        embedder.embed_documents(["a"])


def test_query_failure_raises_embedding_error(embedder, monkeypatch):
    def down(texts, task_type):
        raise TimeoutError("timed out")

    monkeypatch.setattr(embedder, "_call", down)
    with pytest.raises(EmbeddingError):
        embedder.embed_query("vanilla")


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.setattr(emb.config, "GEMINI_API_KEY", "")
    with pytest.raises(EmbeddingError):
        GeminiEmbedder(api_key="")
