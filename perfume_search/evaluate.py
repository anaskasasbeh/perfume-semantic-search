"""Offline evaluation: semantic search (Gemini) vs keyword search (TF-IDF).

Each test query comes with the notes that make a perfume a correct answer,
written before looking at any results (data/eval/queries.json). For example,
"smells like clean laundry" counts a result as relevant if it has musk or
aldehydes.

Metrics (top 10 results):
* Precision@10 – share of the 10 results that are relevant.
* Hit@10       – 1 if at least one of the 10 results is relevant.
* Random       – the share of the whole catalog that is relevant, i.e. what
                 picking 10 perfumes at random would score on average.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path

import numpy as np

from . import config
from .engine import SearchEngine

log = logging.getLogger(__name__)
K = 10


def _relevant_mask(engine: SearchEngine, relevant_notes: list[str]) -> np.ndarray:
    wanted = set(relevant_notes)
    return np.array([bool(wanted & notes) for notes in engine.note_sets], dtype=bool)


def _top_k(scores: np.ndarray, k: int = K, positive_only: bool = False) -> np.ndarray:
    order = np.argsort(-scores)[:k]
    return order[scores[order] > 0] if positive_only else order


def evaluate(
    queries_path: Path = config.EVAL_QUERIES,
    out_dir: Path = config.REPORTS_DIR,
    engine: SearchEngine | None = None,
) -> dict:
    engine = engine or SearchEngine()
    if engine.embedder is None:
        raise SystemExit("GEMINI_API_KEY is required for the semantic part of the evaluation.")
    queries = json.loads(queries_path.read_text(encoding="utf-8"))

    rows, latencies = [], []
    for item in queries:
        relevant = _relevant_mask(engine, item["relevant_notes"])
        started = time.perf_counter()
        query_vector = engine.embedder.embed_query(item["query"])  # direct call, no cache
        latencies.append((time.perf_counter() - started) * 1000)
        semantic_top = _top_k(engine.vectors @ query_vector)
        keyword_top = _top_k(engine.keyword.scores(item["query"]), positive_only=True)
        rows.append(
            {
                "id": item["id"],
                "lang": item["lang"],
                "query": item["query"],
                "random": float(relevant.mean()),
                "semantic_p10": float(relevant[semantic_top].sum() / K),
                "semantic_hit10": float(relevant[semantic_top].any()),
                "keyword_p10": float(relevant[keyword_top].sum() / K),
                "keyword_hit10": float(relevant[keyword_top].any()) if len(keyword_top) else 0.0,
                "keyword_no_results": float(len(keyword_top) == 0),
            }
        )
        time.sleep(0.7)  # stay well under the free tier's requests-per-minute limit

    def summary(lang: str | None) -> dict:
        subset = [r for r in rows if lang is None or r["lang"] == lang]
        mean = lambda key: round(statistics.mean(r[key] for r in subset), 3)  # noqa: E731
        return {
            "queries": len(subset),
            **{
                key: mean(key)
                for key in (
                    "random",
                    "semantic_p10",
                    "semantic_hit10",
                    "keyword_p10",
                    "keyword_hit10",
                    "keyword_no_results",
                )
            },
        }

    report = {
        "model": engine.meta.get("model"),
        "dim": engine.meta.get("dim"),
        "k": K,
        "overall": summary(None),
        "english": summary("en"),
        "arabic": summary("ar"),
        "query_embedding_latency_ms": {
            "median": round(statistics.median(latencies), 1),
            "max": round(max(latencies), 1),
        },
        "per_query": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "evaluation.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def to_markdown(report: dict) -> str:
    def pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    lines = [
        f"Model: `{report['model']}` ({report['dim']} dims). Top {report['k']} results per query.",
        "",
        "| Queries | Semantic P@10 | Keyword P@10 | Random P@10 "
        "| Semantic Hit@10 | Keyword Hit@10 | Keyword no results |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, key in (("English", "english"), ("Arabic", "arabic"), ("All", "overall")):
        s = report[key]
        lines.append(
            f"| {label} ({s['queries']}) | {pct(s['semantic_p10'])} | {pct(s['keyword_p10'])} | {pct(s['random'])} | "
            f"{pct(s['semantic_hit10'])} | {pct(s['keyword_hit10'])} | {pct(s['keyword_no_results'])} |"
        )
    lat = report["query_embedding_latency_ms"]
    lines += [
        "",
        f"Query embedding latency (Gemini API round trip): median {lat['median']} ms, max {lat['max']} ms.",
        "",
    ]
    return "\n".join(lines)
