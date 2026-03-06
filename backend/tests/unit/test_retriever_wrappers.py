from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

from app.models.schemas import SearchResult
from app.services.search.keyword_retriever import KeywordRetriever
from app.services.search.vector_retriever import VectorRetriever


class _DummyEmbedder:
    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


async def test_vector_retriever_wraps_existing_engine():
    chunk_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    mock_vector_engine = AsyncMock()
    mock_vector_engine.search.return_value = [
        SearchResult(
            chunk_id=chunk_id,
            document_id=doc_id,
            content="vector content",
            score=0.91,
            metadata={"source": "pgvector"},
        )
    ]

    retriever = VectorRetriever(
        embedder=_DummyEmbedder(),
        vector_engine=mock_vector_engine,
    )

    results = await retriever.retrieve("테스트 질의", top_k=5, filters={"doc_id": str(doc_id)})

    assert len(results) == 1
    assert results[0].source == "vector"
    assert results[0].chunk_id == chunk_id
    assert results[0].doc_id == doc_id
    assert results[0].content == "vector content"
    assert results[0].metadata == {"source": "pgvector"}
    mock_vector_engine.search.assert_awaited_once()


async def test_keyword_retriever_wraps_existing_engine():
    chunk_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    mock_keyword_engine = AsyncMock()
    mock_keyword_engine.search.return_value = [
        SearchResult(
            chunk_id=chunk_id,
            document_id=doc_id,
            content="keyword content",
            score=12.34,
            metadata={"source": "elasticsearch"},
        )
    ]

    retriever = KeywordRetriever(keyword_engine=mock_keyword_engine)
    results = await retriever.retrieve("국정 감사", top_k=3, filters={"doc_id": str(doc_id)})

    assert len(results) == 1
    assert results[0].source == "keyword"
    assert results[0].chunk_id == chunk_id
    assert results[0].doc_id == doc_id
    assert results[0].content == "keyword content"
    assert results[0].metadata == {"source": "elasticsearch"}
    mock_keyword_engine.search.assert_awaited_once()

