"""Shared fixtures: a tiny catalog and a fake embedder, so tests never call the network."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from perfume_search.embeddings import EmbeddingError, normalize

ROWS = [
    ("Vanilla Dream Eau de Parfum", "Brand A", "A cozy story.", "Vanilla, tonka bean, amber"),
    ("Vanilla Dream Eau de Toilette", "Brand A", "Same scent, lighter.", "Vanilla, tonka bean, amber"),
    ("Winter Amber Parfum", "Brand B", "Warm.", "Amber, benzoin, vanilla"),
    ("Citrus Splash Eau de Toilette", "Brand C", "Sunny.", "Lemon, bergamot, mandarin"),
    ("Garden Rose Eau de Parfum", "Brand B", "Romantic.", "Rose, peony, musk"),
    ("Herbal Rosemary Cologne", "Brand C", "Green herbs.", "Rosemary, sage, cedar"),
    ("Oud Night Extrait", "Brand D", "Dark and smoky.", "Oud, saffron, leather"),
    ("No Notes Here Eau de Parfum", "Brand D", "Amélie created a soft musky veil of white flowers.", ""),
]
CSV_COLUMNS = ["Name", "Brand", "Description", "Notes", "Image URL"]


class FakeEmbedder:
    """Deterministic 'embeddings': bag of hashed words. Similar words → similar vectors."""

    model = "fake-embedder"

    def __init__(self, dim: int = 64, fail: bool = False) -> None:
        self.dim = dim
        self.fail = fail
        self.calls = 0

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().replace(",", " ").replace(".", " ").split():
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return normalize(np.stack([self._vector(t) for t in texts]))

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> np.ndarray:
        self.calls += 1
        if self.fail:
            raise EmbeddingError("simulated outage")
        return self.embed_documents([text])[0]


@pytest.fixture()
def raw_csv(tmp_path: Path) -> Path:
    path = tmp_path / "perfumes.csv"
    df = pd.DataFrame([(*row, f"https://example.com/{i}.jpg") for i, row in enumerate(ROWS)], columns=CSV_COLUMNS)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


@pytest.fixture()
def artifacts(tmp_path: Path, raw_csv: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build real artifacts from the tiny catalog with the fake embedder."""
    import perfume_search.build as build_module

    monkeypatch.setattr(build_module, "choose_k", lambda vectors: (2, {2: 0.1}))
    out = tmp_path / "artifacts"
    build_module.build(raw_csv=raw_csv, out_dir=out, embedder=FakeEmbedder())
    return out
