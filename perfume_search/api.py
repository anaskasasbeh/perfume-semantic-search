"""FastAPI backend: JSON endpoints under /api and the static website at /."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

from . import __version__, config
from .engine import SearchEngine
from .images import fetch_image
from .notes import NOTES_BY_ID

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

# CDN caching: identical requests are answered by Vercel's edge, not the function.
STATIC_CACHE = "public, max-age=3600, s-maxage=86400"
IMAGE_CACHE = "public, max-age=604800, s-maxage=31536000, immutable"
SEARCH_CACHE = "public, max-age=600, s-maxage=86400"


# ---- response models (they also document the API at /docs) -----------------
class Card(BaseModel):
    id: int
    name: str
    brand: str
    image_url: str
    notes: list[str]
    note_ids: list[str]
    group: int
    score: float | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    notice: str | None
    total: int
    took_ms: float
    results: list[Card]


class Perfume(BaseModel):
    id: int
    name: str
    brand: str
    description: str
    notes: list[str]
    note_ids: list[str]
    image_url: str
    group: int


class GroupRow(BaseModel):
    id: int
    name_en: str
    name_ar: str
    size: int
    note_ids: list[str] = []
    items: list[Card]


@lru_cache(maxsize=1)
def get_engine() -> SearchEngine:
    """Load the artifacts once per server instance."""
    return SearchEngine(config.ARTIFACTS_DIR)


EngineDep = Annotated[SearchEngine, Depends(get_engine)]


def parse_notes(raw: str | None) -> list[str]:
    """'vanilla,oud' → ['vanilla', 'oud'], rejecting unknown note ids."""
    if not raw:
        return []
    ids = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [note_id for note_id in ids if note_id not in NOTES_BY_ID]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown note id(s): {', '.join(unknown)}")
    return ids


app = FastAPI(
    title="Perfume Semantic Search API",
    version=__version__,
    description="Semantic perfume search with Gemini embeddings, a TF-IDF fallback and K-means scent groups.",
)


@app.get("/api/health")
def health(engine: EngineDep) -> dict:
    return {
        "status": "ok",
        "perfumes": len(engine.perfumes),
        "groups": len(engine.groups),
        "semantic_available": engine.semantic_available,
        "model": engine.meta.get("model"),
    }


@app.get("/api/search", response_model=SearchResponse)
def search(
    engine: EngineDep,
    response: Response,
    q: Annotated[str, Query(max_length=300, description="Free-text description, English or Arabic")] = "",
    brand: Annotated[str | None, Query(max_length=120)] = None,
    include: Annotated[str | None, Query(description="Comma-separated note ids that must be present")] = None,
    exclude: Annotated[str | None, Query(description="Comma-separated note ids that must be absent")] = None,
    limit: Annotated[int, Query(ge=1, le=48)] = 12,
    offset: Annotated[int, Query(ge=0, le=2000)] = 0,
) -> SearchResponse:
    result = engine.search(
        q,
        brand=brand or None,
        include=parse_notes(include),
        exclude=parse_notes(exclude),
        limit=limit,
        offset=offset,
    )
    # Never let the CDN keep a fallback answer: the next request may be semantic again.
    response.headers["Cache-Control"] = SEARCH_CACHE if result.mode != "keyword" else "no-store"
    return SearchResponse(
        query=q,
        mode=result.mode,
        notice=result.notice,
        total=result.total,
        took_ms=round(result.took_ms, 1),
        results=result.items,
    )


@app.get("/api/perfumes/{perfume_id}", response_model=Perfume)
def perfume(perfume_id: int, engine: EngineDep, response: Response) -> dict:
    try:
        item = engine.get(perfume_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Perfume not found") from None
    response.headers["Cache-Control"] = STATIC_CACHE
    return item


@app.get("/api/perfumes/{perfume_id}/similar", response_model=list[Card])
def similar(
    perfume_id: int,
    engine: EngineDep,
    response: Response,
    limit: Annotated[int, Query(ge=1, le=24)] = 10,
) -> list[dict]:
    try:
        items = engine.similar(perfume_id, limit=limit)
    except KeyError:
        raise HTTPException(status_code=404, detail="Perfume not found") from None
    response.headers["Cache-Control"] = STATIC_CACHE
    return items


@app.get("/api/image/{perfume_id}", include_in_schema=False)
def image(perfume_id: int, engine: EngineDep) -> Response:
    """Serve a product image through our server. Only catalog URLs can be fetched (no open proxy)."""
    try:
        url = engine.get(perfume_id)["image_url"]
        data, content_type = fetch_image(url)
    except KeyError:
        raise HTTPException(status_code=404, detail="Perfume not found") from None
    except Exception as error:  # noqa: BLE001 - any upstream failure means "no image"
        log.info("Image %s unavailable: %s", perfume_id, error)
        raise HTTPException(status_code=404, detail="Image unavailable") from None
    return Response(content=data, media_type=content_type, headers={"Cache-Control": IMAGE_CACHE})


@app.get("/api/groups", response_model=list[GroupRow])
def groups(engine: EngineDep, response: Response, per_group: Annotated[int, Query(ge=1, le=40)] = 16) -> list[dict]:
    response.headers["Cache-Control"] = STATIC_CACHE
    return engine.group_list(per_group)


@app.get("/api/groups/{group_id}", response_model=GroupRow)
def group(
    group_id: int,
    engine: EngineDep,
    response: Response,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=48)] = 24,
) -> dict:
    try:
        data = engine.group_items(group_id, offset, limit)
    except KeyError:
        raise HTTPException(status_code=404, detail="Group not found") from None
    response.headers["Cache-Control"] = STATIC_CACHE
    return data


@app.get("/api/filters")
def filters(engine: EngineDep, response: Response) -> dict:
    response.headers["Cache-Control"] = STATIC_CACHE
    return engine.filters()


@app.get("/api/examples")
def examples(engine: EngineDep, response: Response) -> list[dict]:
    response.headers["Cache-Control"] = STATIC_CACHE
    return engine.examples


# The website. API routes above always win over static files.
app.frontend("/", directory=config.WEB_DIR)
