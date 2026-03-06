"""검색 모드별 retriever 선택 정책."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.services.search.base_retriever import BaseRetriever


class SearchMode(str, Enum):
    HYBRID = "hybrid"
    AUTO = "auto"
    VECTOR = "vector"
    VECTOR_ONLY = "vector_only"
    KEYWORD = "keyword"
    KEYWORD_ONLY = "keyword_only"
    GRAPH = "graph"
    GRAPH_ONLY = "graph_only"
    CASCADING = "cascading"


@dataclass(slots=True)
class RetrievalPlan:
    name: str
    retriever: BaseRetriever
    weight: float
    query_kind: str  # "vector" | "keyword"


class SearchPolicy:
    """모드별 retriever/가중치 선택."""

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        keyword_retriever: BaseRetriever,
        graph_retriever: BaseRetriever | None = None,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.graph_retriever = graph_retriever

    def get_plan(
        self,
        mode: str,
        *,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        graph_enabled: bool = False,
        graph_weight: float = 0.2,
        question_category: str = "",
        query: str = "",
    ) -> list[RetrievalPlan]:
        normalized = self._normalize(mode)

        if normalized in {SearchMode.VECTOR, SearchMode.VECTOR_ONLY}:
            return [RetrievalPlan("vector_search", self.vector_retriever, 1.0, "vector")]

        if normalized in {SearchMode.KEYWORD, SearchMode.KEYWORD_ONLY}:
            return [RetrievalPlan("keyword_search", self.keyword_retriever, 1.0, "keyword")]

        if normalized in {SearchMode.GRAPH, SearchMode.GRAPH_ONLY}:
            if self.graph_retriever is not None and graph_enabled:
                return [RetrievalPlan("graph_search", self.graph_retriever, 1.0, "keyword")]
            # graph 모드를 명시했는데 graph가 비활성이면 자동 폴백하지 않는다.
            return []

        if normalized == SearchMode.AUTO:
            v_w, k_w, g_w = self._auto_weights(question_category=question_category, query=query)
            plans = [
                RetrievalPlan("vector_search", self.vector_retriever, v_w, "vector"),
                RetrievalPlan("keyword_search", self.keyword_retriever, k_w, "keyword"),
            ]
            if self.graph_retriever is not None and graph_enabled and g_w > 0:
                plans.append(RetrievalPlan("graph_search", self.graph_retriever, g_w, "keyword"))
            return plans

        if normalized == SearchMode.HYBRID:
            plans = [
                RetrievalPlan("vector_search", self.vector_retriever, vector_weight, "vector"),
                RetrievalPlan("keyword_search", self.keyword_retriever, keyword_weight, "keyword"),
            ]
            if self.graph_retriever is not None and graph_enabled and graph_weight > 0:
                plans.append(RetrievalPlan("graph_search", self.graph_retriever, graph_weight, "keyword"))
            return plans

        # cascading은 상위 오케스트레이터에서 별도 처리
        return [
            RetrievalPlan("vector_search", self.vector_retriever, vector_weight, "vector"),
            RetrievalPlan("keyword_search", self.keyword_retriever, keyword_weight, "keyword"),
        ]

    @staticmethod
    def _normalize(mode: str) -> SearchMode:
        try:
            return SearchMode(mode)
        except ValueError:
            return SearchMode.HYBRID

    @staticmethod
    def _auto_weights(*, question_category: str, query: str) -> tuple[float, float, float]:
        category = (question_category or "").strip().lower()
        q = (query or "").lower()
        relational_markers = ["관계", "연결", "영향", "비교", "원인", "연관", "네트워크"]
        if category == "relational" or any(marker in q for marker in relational_markers):
            return 0.2, 0.3, 0.5

        if category == "extraction":
            return 0.25, 0.75, 0.0
        if category == "regulatory":
            return 0.4, 0.6, 0.0
        # explanatory / fallback
        return 0.7, 0.3, 0.0
