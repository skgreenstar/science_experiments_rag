"""VectorSearchEngine 어댑터.

기존 PGVector 엔진은 유지하고, BaseRetriever 인터페이스에 맞게 래핑한다.
"""
from __future__ import annotations

import uuid
from typing import Any

from app.services.embedding.base import EmbeddingProvider
from app.services.search.base_retriever import BaseRetriever, RetrievalResult
from app.services.search.vector import VectorSearchEngine


class VectorRetriever(BaseRetriever):
    """벡터 검색 공통 인터페이스 래퍼."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_engine: VectorSearchEngine,
    ) -> None:
        self.embedder = embedder
        self.vector_engine = vector_engine

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        doc_id = None
        if filters and "doc_id" in filters:
            raw_doc_id = filters["doc_id"]
            if isinstance(raw_doc_id, uuid.UUID):
                doc_id = raw_doc_id
            elif isinstance(raw_doc_id, str):
                try:
                    doc_id = uuid.UUID(raw_doc_id)
                except ValueError:
                    doc_id = None

        query_embedding = await self.embedder.embed_query(query)
        results = await self.vector_engine.search(
            query_embedding=query_embedding,
            top_k=top_k,
            doc_id=doc_id,
        )

        return [
            RetrievalResult(
                doc_id=item.document_id,
                chunk_id=item.chunk_id,
                score=float(item.score),
                source="vector",
                content=item.content,
                metadata=item.metadata or {},
            )
            for item in results
        ]
