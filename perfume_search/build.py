"""Build step (runs on your PC): CSV → cleaned catalog → embeddings → groups → artifacts/.

The API only reads the files written here, so the live site never needs the
raw dataset and never re-embeds the catalog.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import config
from .clustering import build_groups, choose_k
from .data import Perfume, load_catalog
from .embeddings import Embedder, GeminiEmbedder

log = logging.getLogger(__name__)

# Clickable examples on the home page. Their vectors are computed here, so a
# click never spends API quota.
EXAMPLE_QUERIES: list[dict] = [
    {"text": "warm vanilla for winter nights", "lang": "en"},
    {"text": "fresh citrus for a hot summer day", "lang": "en"},
    {"text": "smells like clean laundry", "lang": "en"},
    {"text": "dark smoky oud", "lang": "en"},
    {"text": "romantic rose for a date", "lang": "en"},
    {"text": "sweet like a dessert", "lang": "en"},
    {"text": "عطر دافي لليالي الشتا", "lang": "ar"},
    {"text": "ريحة منعشة للصيف", "lang": "ar"},
    {"text": "ريحة غسيل نظيف", "lang": "ar"},
    {"text": "عود فخم وقوي", "lang": "ar"},
    {"text": "ورد رومانسي", "lang": "ar"},
    {"text": "ريحة حلويات", "lang": "ar"},
]


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def build(
    raw_csv: Path = config.RAW_CSV,
    out_dir: Path = config.ARTIFACTS_DIR,
    embedder: Embedder | None = None,
    limit: int | None = None,
) -> dict:
    """Run the full pipeline and return the metadata that was saved."""
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog: list[Perfume] = load_catalog(raw_csv)
    if limit:
        catalog = catalog[:limit]

    embedder = embedder or GeminiEmbedder(timeout_ms=60_000)
    log.info("Embedding %d perfumes with %s …", len(catalog), getattr(embedder, "model", type(embedder).__name__))
    vectors = embedder.embed_documents([p.embedding_text for p in catalog])

    log.info("Choosing the number of groups …")
    k, silhouettes = choose_k(vectors)
    groups = build_groups(vectors, [p.note_ids for p in catalog], k)
    for group in groups:
        for member in group.members:
            catalog[member].group = group.id

    example_vectors = embedder.embed_queries([e["text"] for e in EXAMPLE_QUERIES])

    perfumes = []
    for p in catalog:
        item = p.to_dict()
        item["text"] = item.pop("embedding_text")
        item.pop("extra", None)
        perfumes.append(item)

    np.save(out_dir / "embeddings.npy", vectors.astype(np.float32))
    np.save(out_dir / "example_vectors.npy", np.asarray(example_vectors, dtype=np.float32))
    _write_json(out_dir / "perfumes.json", perfumes)
    _write_json(out_dir / "groups.json", [g.to_dict() for g in groups])
    _write_json(out_dir / "examples.json", EXAMPLE_QUERIES)

    meta = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": getattr(embedder, "model", type(embedder).__name__),
        "dim": int(vectors.shape[1]),
        "perfumes": len(catalog),
        "perfumes_without_notes": sum(1 for p in catalog if not p.notes),
        "brands": len({p.brand for p in catalog}),
        "embedding_text": "base name + notes (description snippet only when notes are missing)",
        "k": k,
        "silhouette_by_k": silhouettes,
        "group_sizes": [len(g.members) for g in groups],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Artifacts written to %s (k=%d, silhouette=%.4f)", out_dir, k, silhouettes[k])
    return meta
