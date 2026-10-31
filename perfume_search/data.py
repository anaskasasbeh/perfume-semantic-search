"""Load the raw perfume CSV and turn it into a clean catalog.

Steps: read the file → repair broken characters → clean names and notes →
match catalog notes → build the short text that gets embedded.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import ftfy
import pandas as pd

from . import config
from .notes import match_notes

log = logging.getLogger(__name__)

# Concentration words at the end of a product name ("Tihota Eau de Parfum").
_CONCENTRATION = re.compile(
    r"\s*[-–]?\s*(?:"
    r"extrait de parfum|parfum extrait|parfum concentrat(?:e|ion)|eau de parfum(?: intense)?|"
    r"eau de toilette|eau de cologne|cologne|pure oud(?: oil)?|perfume oil|parfum|extrait|edp|edt|"
    r"travel spray|hair mist|body mist"
    r")\s*(?:\.{3})?\s*$",
    re.IGNORECASE,
)
_C1_CONTROLS = re.compile(r"[\x80-\x9f]")
_SPACES = re.compile(r"\s+")
_NOTE_SPLIT = re.compile(r"[,;/]")
_DESCRIPTION_FALLBACK_CHARS = 300


@dataclass
class Perfume:
    """One cleaned catalog entry."""

    id: int
    name: str
    base_name: str
    brand: str
    description: str
    notes: list[str]
    note_ids: list[str]
    image_url: str
    embedding_text: str
    group: int = -1
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def clean_text(value: object) -> str:
    """Repair mojibake (e.g. 'AmÃ©lie' → 'Amélie') and normalise whitespace."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = ftfy.fix_text(str(value))
    text = _C1_CONTROLS.sub("", text)  # leftovers of broken quote marks
    text = text.replace('"œ', '"').replace('"˜', "‘")  # broken quotes left by the source encoding
    return _SPACES.sub(" ", text).strip()


def clean_name(name: str) -> str:
    """Drop the concentration suffix: 'Tihota Eau de Parfum' → 'Tihota'."""
    base = _CONCENTRATION.sub("", name).strip(" -–.")
    return base or name


def split_notes(notes: str) -> list[str]:
    """'Vanilla bean, musks' → ['Vanilla bean', 'Musks'] (deduplicated, order kept)."""
    seen: set[str] = set()
    result: list[str] = []
    for part in _NOTE_SPLIT.split(notes or ""):
        note = part.strip(" .")
        if not note:
            continue
        key = note.lower()
        if key not in seen:
            seen.add(key)
            result.append(note[0].upper() + note[1:])
    return result


def build_embedding_text(base_name: str, notes: list[str], description: str) -> str:
    """Text sent to the embedding model: name + notes.

    The marketing descriptions read like short stories, so they are left out.
    The only exception is a perfume with no notes at all: then the first
    sentences of its description are used so the vector is not empty.
    """
    if notes:
        return f"{base_name}. Notes: {', '.join(notes)}."
    snippet = description[:_DESCRIPTION_FALLBACK_CHARS].rsplit(" ", 1)[0]
    return f"{base_name}. {snippet}"


def read_raw_csv(path: Path = config.RAW_CSV, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Read the source CSV and rename its columns to our internal names."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from Kaggle (see README → Dataset) and place it there."
        )
    column_map = column_map or config.COLUMN_MAP
    # latin-1 never fails to decode; ftfy then repairs the mixed encodings.
    df = pd.read_csv(path, encoding="latin-1", dtype=str, keep_default_na=False)
    missing = [src for src in column_map.values() if src not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}. Update COLUMN_MAP in config.py.")
    df = df.rename(columns={src: dst for dst, src in column_map.items()})
    return df[list(column_map)]


def build_catalog(df: pd.DataFrame) -> list[Perfume]:
    """Clean every row and return the catalog, skipping rows without a name."""
    catalog: list[Perfume] = []
    for row in df.itertuples(index=False):
        name = clean_text(row.name)
        if not name:
            continue
        notes = split_notes(clean_text(row.notes))
        description = clean_text(row.description)
        base_name = clean_name(name)
        catalog.append(
            Perfume(
                id=len(catalog),
                name=name,
                base_name=base_name,
                brand=clean_text(row.brand) or "Unknown",
                description=description,
                notes=notes,
                note_ids=match_notes(", ".join(notes)),
                image_url=clean_text(row.image_url),
                embedding_text=build_embedding_text(base_name, notes, description),
            )
        )
    without_notes = sum(1 for p in catalog if not p.notes)
    log.info("Catalog: %d perfumes (%d without notes use a description snippet)", len(catalog), without_notes)
    return catalog


def load_catalog(path: Path = config.RAW_CSV) -> list[Perfume]:
    """Convenience wrapper: CSV path → cleaned catalog."""
    return build_catalog(read_raw_csv(path))
