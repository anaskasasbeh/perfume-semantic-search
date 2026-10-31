Model: `gemini-embedding-001` (768 dims). Top 10 results per query.

| Queries | Semantic P@10 | Keyword P@10 | Random P@10 | Semantic Hit@10 | Keyword Hit@10 | Keyword no results |
|---|---|---|---|---|---|---|
| English (15) | 81.3% | 44.0% | 28.0% | 100.0% | 100.0% | 0.0% |
| Arabic (15) | 77.3% | 0.0% | 28.0% | 100.0% | 0.0% | 100.0% |
| All (30) | 79.3% | 22.0% | 28.0% | 100.0% | 50.0% | 50.0% |

Query embedding latency (Gemini API round trip): median 341.3 ms, max 608.0 ms.
