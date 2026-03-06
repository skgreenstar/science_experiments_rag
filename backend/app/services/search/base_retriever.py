"""검색기 공통 인터페이스.

Phase 1에서는 기존 엔진(Vector/Keyword)을 변경하지 않고
어댑터 레이어로 감싸기 위한 최소 추상화만 제공한다.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalResult:
    """저장소 유형과 무관한 공통 검색 결과 스키마."""

    doc_id: uuid.UUID
    chunk_id: uuid.UUID
    score: float
    source: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_path: list[dict[str, Any]] = field(default_factory=list)


class BaseRetriever(ABC):
    """모든 검색기(벡터/키워드/그래프)의 공통 계약."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """쿼리에 대한 검색 결과를 반환한다."""
        raise NotImplementedError

