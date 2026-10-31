"""Check which product image links still work (they are hosted by the retailer)."""

from __future__ import annotations

import json
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config

log = logging.getLogger(__name__)
_HEADERS = {"User-Agent": "Mozilla/5.0 (perfume-semantic-search image check)"}
MAX_IMAGE_BYTES = 3_000_000


def fetch_image(url: str, timeout: float = 8.0) -> tuple[bytes, str]:
    """Download one product image server-side (used by the /api/image proxy).

    Browsers can refuse to show the retailer's images on another site; a
    server-side request is not affected by that rule.
    """
    if not url.startswith("https://"):
        raise ValueError("Only https image URLs are allowed")
    request = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise ValueError(f"Not an image: {content_type!r}")
        data = response.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image too large")
    return data, content_type


def _is_ok(url: str, timeout: float = 10.0) -> bool:
    if not url.startswith("http"):
        return False
    for method in ("HEAD", "GET"):
        try:
            request = urllib.request.Request(url, method=method, headers=_HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status < 400 and response.headers.get("Content-Type", "").startswith("image"):
                    return True
        except Exception:  # noqa: BLE001 - any failure just means "not reachable"
            continue
    return False


def check_images(
    artifacts_dir: Path = config.ARTIFACTS_DIR, out_dir: Path = config.REPORTS_DIR, workers: int = 16
) -> dict:
    path = artifacts_dir / "perfumes.json"
    if not path.exists():
        raise FileNotFoundError(f"No artifacts in {artifacts_dir}. Run `python -m perfume_search build` first.")
    perfumes = json.loads(path.read_text(encoding="utf-8"))
    urls = [p["image_url"] for p in perfumes]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_is_ok, urls))
    broken = [p["id"] for p, ok in zip(perfumes, results, strict=True) if not ok]
    report = {
        "checked": len(urls),
        "working": len(urls) - len(broken),
        "broken": len(broken),
        "broken_ids": broken,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
