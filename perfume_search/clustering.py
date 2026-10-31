"""Group perfumes into scent families for the home page rows.

K-means runs on the embedding vectors. The number of groups is chosen by the
silhouette score inside a range that suits the UI, and each group is named
after the notes that are most over-represented in it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import config
from .notes import NOTES_BY_ID

log = logging.getLogger(__name__)


@dataclass
class Group:
    id: int
    note_ids: list[str]  # the notes that name the group
    name_en: str
    name_ar: str
    members: list[int]  # perfume ids, most representative first

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "note_ids": self.note_ids,
            "name_en": self.name_en,
            "name_ar": self.name_ar,
            "size": len(self.members),
            "members": self.members,
        }


def choose_k(
    vectors: np.ndarray,
    k_range: tuple[int, int] = config.K_RANGE,
    min_fraction: float = config.MIN_GROUP_FRACTION,
    seed: int = config.RANDOM_SEED,
) -> tuple[int, dict[int, float]]:
    """Try every k in the range and keep the best silhouette whose groups are not tiny."""
    scores: dict[int, float] = {}
    best_k, best_score = k_range[0], -1.0
    min_size = int(min_fraction * len(vectors))
    for k in range(k_range[0], k_range[1] + 1):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(vectors)
        score = float(silhouette_score(vectors, labels, metric="cosine"))
        scores[k] = round(score, 4)
        smallest = int(np.bincount(labels).min())
        log.info("k=%d silhouette=%.4f smallest group=%d", k, score, smallest)
        if smallest >= min_size and score > best_score:
            best_k, best_score = k, score
    return best_k, scores


GENERIC_NAME = ("Rare & unusual notes", "مكوّنات نادرة ومميزة")


def rank_notes(member_note_ids: list[list[str]], overall_share: dict[str, float]) -> list[str]:
    """Order a group's notes: distinctive ones first (high share x lift), then simply common ones."""
    counts: dict[str, int] = {}
    for note_ids in member_note_ids:
        for note_id in note_ids:
            counts[note_id] = counts.get(note_id, 0) + 1
    size = max(len(member_note_ids), 1)
    share = {note_id: count / size for note_id, count in counts.items()}
    lift = {note_id: share[note_id] / max(overall_share.get(note_id, 1e-9), 1e-9) for note_id in share}
    distinctive = sorted(
        (n for n in share if share[n] >= 0.10 and lift[n] > 1.1),
        key=lambda n: share[n] * np.log(lift[n]),
        reverse=True,
    )
    common = sorted((n for n in share if share[n] >= 0.10 and n not in distinctive), key=lambda n: -share[n])
    return distinctive + common


def name_groups(ranked: list[list[str]], top: int = 3) -> list[list[str]]:
    """Take the top notes of each group, making sure no two groups get the same name."""
    names: list[list[str]] = []
    used: set[tuple[str, ...]] = set()
    for candidates in ranked:
        chosen = candidates[:top]
        extra = candidates[top:]
        while tuple(sorted(chosen)) in used and extra:  # same name as a bigger group: swap the last note
            chosen = chosen[:-1] + [extra.pop(0)]
        used.add(tuple(sorted(chosen)))
        names.append(chosen)
    return names


def build_groups(vectors: np.ndarray, note_ids: list[list[str]], k: int, seed: int = config.RANDOM_SEED) -> list[Group]:
    """Cluster, order members by closeness to the centre, and name each group."""
    model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(vectors)
    centres = model.cluster_centers_ / np.linalg.norm(model.cluster_centers_, axis=1, keepdims=True)
    total = len(note_ids)
    overall: dict[str, float] = {}
    for ids in note_ids:
        for note_id in ids:
            overall[note_id] = overall.get(note_id, 0) + 1 / total

    members_by_group = []
    for gid in range(k):
        members = np.where(model.labels_ == gid)[0]
        closeness = vectors[members] @ centres[gid]
        members_by_group.append(members[np.argsort(-closeness)].tolist())
    members_by_group.sort(key=len, reverse=True)  # biggest rows first; they also get first pick of names

    ranked = [rank_notes([note_ids[i] for i in members], overall) for members in members_by_group]
    groups: list[Group] = []
    for gid, (members, top_notes) in enumerate(zip(members_by_group, name_groups(ranked), strict=True)):
        groups.append(
            Group(
                id=gid,
                note_ids=top_notes,
                name_en=" · ".join(NOTES_BY_ID[n].en for n in top_notes) or GENERIC_NAME[0],
                name_ar=" · ".join(NOTES_BY_ID[n].ar for n in top_notes) or GENERIC_NAME[1],
                members=members,
            )
        )
    return groups
