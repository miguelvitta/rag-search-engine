import os

from .keyword_search import InvertedIndex
from .search_utils import DEFAULT_SEARCH_LIMIT
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_result = self._bm25_search(query, limit * 500)
        chunked_result = self.semantic_search.search_chunks(query, limit * 500)
        bm25_result_scores = []
        for result in bm25_result:
            bm25_result_scores.append(result["score"])
        chunked_result_scores = []
        for result in chunked_result:
            chunked_result_scores.append(result["score"])
        normalized_bm25_scores = normalize_scores(bm25_result_scores)
        normalized_chunked_scores = normalize_scores(chunked_result_scores)
        for result, normalized_score in zip(bm25_result, normalized_bm25_scores):
            result["bm25_score"] = normalized_score
        for result, normalized_score in zip(chunked_result, normalized_chunked_scores):
            result["semantic_score"] = normalized_score
        combined = {}
        for result in bm25_result:
            doc_id = result["id"]
            combined[doc_id] = result
            combined[doc_id]["semantic_score"] = 0.0
        for result in chunked_result:
            doc_id = result["id"]
            if doc_id not in combined:
                combined[doc_id] = result
                combined[doc_id]["bm25_score"] = 0.0
            combined[doc_id]["semantic_score"] = result["semantic_score"]
        for item in combined.values():
            item["hybrid_score"] = alpha * item["bm25_score"] + (1 - alpha) * item["semantic_score"]

        sorted_items = list(sorted(
            combined.values(),
            key=lambda item: item["hybrid_score"],
            reverse=True,
        ))
        return sorted_items



    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    normalized_scores = []
    for s in scores:
        normalized_scores.append((s - min_score) / (max_score - min_score))

    return normalized_scores
