import pytest
from fastapi.testclient import TestClient

from perfume_search.api import app, get_engine
from perfume_search.engine import SearchEngine

from .conftest import FakeEmbedder


@pytest.fixture()
def client(artifacts):
    engine = SearchEngine(artifacts, embedder=FakeEmbedder())
    app.dependency_overrides[get_engine] = lambda: engine
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_search_endpoint(client):
    response = client.get("/api/search", params={"q": "rose peony"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "semantic"
    assert body["results"][0]["name"] == "Garden Rose Eau de Parfum"


def test_unknown_note_is_rejected(client):
    assert client.get("/api/search", params={"include": "unicorn"}).status_code == 422


def test_query_length_is_limited(client):
    assert client.get("/api/search", params={"q": "x" * 301}).status_code == 422


def test_perfume_and_similar_endpoints(client):
    assert client.get("/api/perfumes/0").json()["brand"] == "Brand A"
    assert client.get("/api/perfumes/999").status_code == 404
    assert len(client.get("/api/perfumes/0/similar", params={"limit": 3}).json()) == 3


def test_groups_filters_examples_health(client):
    assert len(client.get("/api/groups").json()) == 2
    assert any(n["id"] == "vanilla" for n in client.get("/api/filters").json()["notes"])
    assert client.get("/api/examples").json()
    assert client.get("/api/health").json()["status"] == "ok"


def test_website_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "<html" in response.text.lower()


def test_image_proxy_only_serves_catalog_images(client, monkeypatch):
    import perfume_search.api as api_module

    monkeypatch.setattr(api_module, "fetch_image", lambda url: (b"\x89PNG", "image/png"))
    ok = client.get("/api/image/0")
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/png"
    assert "immutable" in ok.headers["cache-control"]
    assert client.get("/api/image/999").status_code == 404

    def broken(url):
        raise OSError("blocked")

    monkeypatch.setattr(api_module, "fetch_image", broken)
    assert client.get("/api/image/0").status_code == 404
