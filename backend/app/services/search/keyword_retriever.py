"""ElasticsearchNoriEngine 어댑터.

기존 키워드 엔진은 유지하고, BaseRetriever 인터페이스에 맞게 래핑한다.
"""
from __future__ import annotations

from typing import Any

from app.services.search.base_retriever import BaseRetriever, RetrievalResult
from app.services.search.keyword_es import ElasticsearchNoriEngine


class KeywordRetriever(BaseRetriever):
    """키워드 검색 공통 인터페이스 래퍼."""

    def __init__(self, keyword_engine: ElasticsearchNoriEngine) -> None:
        self.keyword_engine = keyword_engine

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        doc_id = None
        if filters and "doc_id" in filters:
            doc_id = filters["doc_id"]

        results = await self.keyword_engine.search(
            query=query,
            top_k=top_k,
            doc_id=doc_id,
        )

        return [
            RetrievalResult(
                doc_id=item.document_id,
                chunk_id=item.chunk_id,
                score=float(item.score),
                source="keyword",
                content=item.content,
                metadata=item.metadata or {},
            )
            for item in results
        ]

