"""TF-IDF keyword search: the fallback when the embedding API is unavailable,
and the baseline the semantic search is compared against.

TF-IDF matches *words*, not meanings: "clean laundry" only finds perfumes whose
text literally contains those words. It also cannot match Arabic queries to
English perfume text.
"""

from __future__ import annotations

import numpy as np


class KeywordIndex:
    """Fits a TF-IDF model on the same text that is sent to the embedding model."""

    def __init__(self, texts: list[str]) -> None:
        # Imported here so the API only pays the import cost if the fallback is used.
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(texts)  # rows are L2-normalised

    def scores(self, query: str) -> np.ndarray:
        """Cosine similarity between the query and every perfume (0 = no shared words)."""
        query_vector = self._vectorizer.transform([query])
        return (self._matrix @ query_vector.T).toarray().ravel().astype(np.float32)
