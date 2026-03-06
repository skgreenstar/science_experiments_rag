"""Retriever-agnostic RRF Fusion."""
from __future__ import annotations

from app.services.search.base_retriever import RetrievalResult


class RRFFusion:
    """여러 retriever 결과를 rank 기반으로 결합한다."""

    def combine(
        self,
        result_lists: list[list[RetrievalResult]],
        *,
        k: int = 60,
        weights: list[float] | None = None,
    ) -> list[RetrievalResult]:
        if not result_lists:
            return []

        if weights is None:
            weights = [1.0] * len(result_lists)

        scores: dict[str, float] = {}
        best_item: dict[str, RetrievalResult] = {}

        for idx, results in enumerate(result_lists):
            w = weights[idx] if idx < len(weights) else 1.0
            if w <= 0:
                continue
            for rank, item in enumerate(results):
                key = str(item.chunk_id)
                scores[key] = scores.get(key, 0.0) + (w / (k + rank + 1))
                if key not in best_item:
                    best_item[key] = item

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused: list[RetrievalResult] = []
        for key, score in ordered:
            base = best_item[key]
            fused.append(
                RetrievalResult(
                    doc_id=base.doc_id,
                    chunk_id=base.chunk_id,
                    score=score,
                    source=base.source,
                    content=base.content,
                    metadata=base.metadata,
                    evidence_path=base.evidence_path,
                )
            )
        return fused
