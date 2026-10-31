import json

import perfume_search.evaluate as ev
from perfume_search.engine import SearchEngine

from .conftest import FakeEmbedder


def test_evaluation_writes_reports(artifacts, tmp_path, monkeypatch):
    monkeypatch.setattr(ev.time, "sleep", lambda seconds: None)
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps(
            [
                {"id": "citrus", "lang": "en", "query": "lemon bergamot", "relevant_notes": ["lemon", "bergamot"]},
                {"id": "citrus", "lang": "ar", "query": "ليمون", "relevant_notes": ["lemon"]},
            ]
        ),
        encoding="utf-8",
    )
    engine = SearchEngine(artifacts, embedder=FakeEmbedder())
    report = ev.evaluate(queries, tmp_path / "reports", engine=engine)
    assert report["english"]["semantic_hit10"] == 1.0
    assert report["arabic"]["keyword_no_results"] == 1.0  # TF-IDF cannot match Arabic to English text
    assert (tmp_path / "reports" / "evaluation.md").exists()
