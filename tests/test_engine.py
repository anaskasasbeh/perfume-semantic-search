from perfume_search.engine import SearchEngine

from .conftest import FakeEmbedder


def test_semantic_search_ranks_matching_perfume_first(artifacts):
    engine = SearchEngine(artifacts, embedder=FakeEmbedder())
    result = engine.search("lemon bergamot")
    assert result.mode == "semantic"
    assert result.items[0]["name"] == "Citrus Splash Eau de Toilette"
    scores = [item["score"] for item in result.items]
    assert scores == sorted(scores, reverse=True)


def test_filters_include_exclude_and_brand(artifacts):
    engine = SearchEngine(artifacts, embedder=FakeEmbedder())
    names = {i["name"] for i in engine.search("", include=["vanilla"], exclude=["benzoin"]).items}
    assert names == {"Vanilla Dream Eau de Parfum", "Vanilla Dream Eau de Toilette"}
    assert all(i["brand"] == "Brand C" for i in engine.search("fresh", brand="Brand C").items)


def test_fallback_to_keyword_when_embedder_fails(artifacts):
    engine = SearchEngine(artifacts, embedder=FakeEmbedder(fail=True))
    result = engine.search("oud saffron")
    assert result.mode == "keyword" and result.notice == "fallback"
    assert result.items[0]["name"] == "Oud Night Extrait"


def test_arabic_query_in_fallback_returns_notice(artifacts):
    engine = SearchEngine(artifacts, embedder=FakeEmbedder(fail=True))
    result = engine.search("عطر فانيلا")
    assert result.items == [] and result.notice == "arabic_unavailable"


def test_example_queries_do_not_call_the_api(artifacts):
    embedder = FakeEmbedder()
    engine = SearchEngine(artifacts, embedder=embedder)
    engine.search(engine.examples[0]["text"])
    assert embedder.calls == 0


def test_similar_skips_itself_and_other_sizes_of_same_scent(artifacts):
    engine = SearchEngine(artifacts, embedder=FakeEmbedder())
    names = [i["name"] for i in engine.similar(0, limit=5)]
    assert "Vanilla Dream Eau de Parfum" not in names
    assert "Vanilla Dream Eau de Toilette" not in names
    assert names[0] == "Winter Amber Parfum"


def test_groups_cover_every_perfume(artifacts):
    engine = SearchEngine(artifacts, embedder=FakeEmbedder())
    assert sum(g["size"] for g in engine.groups) == len(engine.perfumes)


def test_group_names_are_unique():
    from perfume_search.clustering import name_groups

    names = name_groups([["rose", "musk", "amber", "oud"], ["musk", "amber", "rose", "iris"], []])
    assert names[0] == ["rose", "musk", "amber"]
    assert sorted(names[1]) != sorted(names[0])  # same notes as the bigger group → one note swapped
    assert names[2] == []  # no notes at all → generic name in the UI
