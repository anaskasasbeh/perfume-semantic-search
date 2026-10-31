"""Entrypoint for Vercel and for local runs: `uvicorn app:app --reload`."""

from perfume_search.api import app

__all__ = ["app"]
