"""The search engine used by the API and the CLI.

It loads the pre-built artifacts once and answers four kinds of requests:
semantic search, "more like this", home-page groups and filter lists.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .embeddings import Embedder, EmbeddingError, GeminiEmbedder, normalize
from .keyword import KeywordIndex
from .notes import FAMILIES, NOTE_CATALOG

log = logging.getLogger(__name__)

_ARABIC = re.compile(r"[؀-ۿ]")
CARD_NOTES = 6


def is_arabic(text: str) -> bool:
    return bool(_ARABIC.search(text))


def normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass
class SearchResult:
    mode: str  # "semantic" | "keyword" | "browse"
    items: list[dict]
    notice: str | None = None  # e.g. "fallback", "arabic_unavailable"
    took_ms: float = 0.0
    total: int = 0
    extra: dict = field(default_factory=dict)


class _LRU:
    """Tiny least-recently-used cache for query vectors (saves API quota)."""

    def __init__(self, size: int = 512) -> None:
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()
        self._size = size

    def get(self, key: str) -> np.ndarray | None:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key: str, value: np.ndarray) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self._size:
            self._data.popitem(last=False)


class SearchEngine:
    def __init__(self, artifacts_dir: Path = config.ARTIFACTS_DIR, embedder: Embedder | None = None) -> None:
        self.dir = artifacts_dir
        if not (artifacts_dir / "perfumes.json").exists():
            raise FileNotFoundError(
                f"No artifacts in {artifacts_dir}. Run `python -m perfume_search build` first "
                "(it needs GEMINI_API_KEY and the Kaggle CSV)."
            )
        self.perfumes: list[dict] = json.loads((artifacts_dir / "perfumes.json").read_text(encoding="utf-8"))
        self.vectors = normalize(np.load(artifacts_dir / "embeddings.npy"))
        self.groups: list[dict] = json.loads((artifacts_dir / "groups.json").read_text(encoding="utf-8"))
        self.meta: dict = json.loads((artifacts_dir / "meta.json").read_text(encoding="utf-8"))
        if len(self.perfumes) != len(self.vectors):
            raise ValueError("perfumes.json and embeddings.npy have different lengths; rebuild the artifacts.")

        self._brands = np.array([p["brand"] for p in self.perfumes])
        self._note_sets = [set(p["note_ids"]) for p in self.perfumes]
        self._base_keys = [(p["base_name"].lower(), p["brand"].lower()) for p in self.perfumes]
        self._query_cache = _LRU()
        self._load_examples()
        self._keyword: KeywordIndex | None = None
        self._embedder = embedder if embedder is not None else self._default_embedder()
        log.info(
            "Engine ready: %d perfumes, %d groups, semantic=%s",
            len(self.perfumes),
            len(self.groups),
            self._embedder is not None,
        )

    # ---- setup ------------------------------------------------------------
    @staticmethod
    def _default_embedder() -> Embedder | None:
        if not config.GEMINI_API_KEY:
            log.warning("GEMINI_API_KEY not set: search runs in keyword (TF-IDF) mode only.")
            return None
        try:
            return GeminiEmbedder()
        except EmbeddingError as error:
            log.warning("Embedder unavailable (%s): keyword mode only.", error)
            return None

    def _load_examples(self) -> None:
        path = self.dir / "examples.json"
        vectors_path = self.dir / "example_vectors.npy"
        self.examples: list[dict] = []
        if path.exists() and vectors_path.exists():
            self.examples = json.loads(path.read_text(encoding="utf-8"))
            vectors = normalize(np.load(vectors_path))
            for example, vector in zip(self.examples, vectors, strict=True):
                self._query_cache.put(normalize_query(example["text"]), vector)

    @property
    def keyword(self) -> KeywordIndex:
        if self._keyword is None:
            self._keyword = KeywordIndex([p["text"] for p in self.perfumes])
        return self._keyword

    @property
    def semantic_available(self) -> bool:
        return self._embedder is not None

    @property
    def embedder(self) -> Embedder | None:
        return self._embedder

    @property
    def note_sets(self) -> list[set[str]]:
        return self._note_sets

    # ---- helpers ----------------------------------------------------------
    def card(self, index: int, score: float | None = None) -> dict:
        p = self.perfumes[index]
        card = {
            "id": p["id"],
            "name": p["name"],
            "brand": p["brand"],
            "image_url": p["image_url"],
            "notes": p["notes"][:CARD_NOTES],
            "note_ids": p["note_ids"],
            "group": p["group"],
        }
        if score is not None:
            card["score"] = round(float(score), 4)
        return card

    def mask(
        self, brand: str | None = None, include: list[str] | None = None, exclude: list[str] | None = None
    ) -> np.ndarray:
        """Boolean filter: brand must match, ALL included notes present, NO excluded note present."""
        keep = np.ones(len(self.perfumes), dtype=bool)
        if brand:
            keep &= self._brands == brand
        include_set, exclude_set = set(include or []), set(exclude or [])
        if include_set or exclude_set:
            keep &= np.array(
                [include_set <= notes and not (exclude_set & notes) for notes in self._note_sets], dtype=bool
            )
        return keep

    @staticmethod
    def _top(scores: np.ndarray, keep: np.ndarray, limit: int) -> list[int]:
        scores = np.where(keep, scores, -np.inf)
        count = min(limit, int(keep.sum()))
        if count <= 0:
            return []
        top = np.argpartition(-scores, count - 1)[:count]
        return top[np.argsort(-scores[top])].tolist()

    def query_vector(self, query: str) -> np.ndarray:
        """Embedding of a query: example/LRU cache first, then the API."""
        key = normalize_query(query)
        cached = self._query_cache.get(key)
        if cached is not None:
            return cached
        if self._embedder is None:
            raise EmbeddingError("No embedder configured.")
        vector = self._embedder.embed_query(query)
        self._query_cache.put(key, vector)
        return vector

    # ---- public API -------------------------------------------------------
    def search(
        self,
        query: str = "",
        brand: str | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        limit: int = 12,
        offset: int = 0,
    ) -> SearchResult:
        start = time.perf_counter()
        keep = self.mask(brand, include, exclude)
        total = int(keep.sum())
        query = query.strip()

        if not query:  # filters only → alphabetical browse
            ids = [i for i in np.where(keep)[0].tolist()]
            ids.sort(key=lambda i: self.perfumes[i]["name"].lower())
            page = ids[offset : offset + limit]
            return SearchResult(
                "browse",
                [self.card(i) for i in page],
                total=total,
                took_ms=(time.perf_counter() - start) * 1000,
            )

        notice = None
        try:
            scores = self.vectors @ self.query_vector(query)
            mode = "semantic"
        except EmbeddingError as error:
            if self._embedder is not None:  # a real outage, not just "no key configured"
                log.warning("Semantic search unavailable, using keyword fallback: %s", error)
            mode, notice = "keyword", "fallback"
            if is_arabic(query):
                return SearchResult(
                    mode,
                    [],
                    notice="arabic_unavailable",
                    total=0,
                    took_ms=(time.perf_counter() - start) * 1000,
                )
            scores = self.keyword.scores(query)
            keep &= scores > 0  # keyword mode: only perfumes that share a word

        ids = self._top(scores, keep, offset + limit)[offset:]
        return SearchResult(
            mode,
            [self.card(i, scores[i]) for i in ids],
            notice=notice,
            total=int(keep.sum()) if mode == "keyword" else total,
            took_ms=(time.perf_counter() - start) * 1000,
        )

    def similar(
        self,
        perfume_id: int,
        limit: int = 10,
        brand: str | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[dict]:
        """Nearest neighbours of one perfume. Other sizes/concentrations of the same scent are skipped."""
        if not 0 <= perfume_id < len(self.perfumes):
            raise KeyError(perfume_id)
        keep = self.mask(brand, include, exclude)
        same = self._base_keys[perfume_id]
        keep &= np.array([key != same for key in self._base_keys], dtype=bool)
        scores = self.vectors @ self.vectors[perfume_id]
        return [self.card(i, scores[i]) for i in self._top(scores, keep, limit)]

    def get(self, perfume_id: int) -> dict:
        if not 0 <= perfume_id < len(self.perfumes):
            raise KeyError(perfume_id)
        p = self.perfumes[perfume_id]
        return {
            key: p[key] for key in ("id", "name", "brand", "description", "notes", "note_ids", "image_url", "group")
        }

    def group_list(self, per_group: int = 20) -> list[dict]:
        return [
            {
                "id": g["id"],
                "name_en": g["name_en"],
                "name_ar": g["name_ar"],
                "note_ids": g["note_ids"],
                "size": g["size"],
                "items": [self.card(i) for i in g["members"][:per_group]],
            }
            for g in self.groups
        ]

    def group_items(self, group_id: int, offset: int = 0, limit: int = 24) -> dict:
        if not 0 <= group_id < len(self.groups):
            raise KeyError(group_id)
        g = self.groups[group_id]
        members = g["members"][offset : offset + limit]
        return {
            "id": g["id"],
            "name_en": g["name_en"],
            "name_ar": g["name_ar"],
            "size": g["size"],
            "items": [self.card(i) for i in members],
        }

    def filters(self) -> dict:
        brands: dict[str, int] = {}
        for p in self.perfumes:
            brands[p["brand"]] = brands.get(p["brand"], 0) + 1
        note_counts: dict[str, int] = {}
        for notes in self._note_sets:
            for note_id in notes:
                note_counts[note_id] = note_counts.get(note_id, 0) + 1
        return {
            "brands": [{"name": b, "count": c} for b, c in sorted(brands.items(), key=lambda x: x[0].lower())],
            "notes": [
                {"id": n.id, "en": n.en, "ar": n.ar, "family": n.family, "count": note_counts.get(n.id, 0)}
                for n in NOTE_CATALOG
                if note_counts.get(n.id, 0) > 0
            ],
            "families": [{"id": fid, "en": en, "ar": ar} for fid, (en, ar) in FAMILIES.items()],
        }
