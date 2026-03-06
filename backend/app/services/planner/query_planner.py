"""질의 유형 기반 검색 전략 라우팅."""
from __future__ import annotations

from app.services.fusion.search_policy import SearchMode


class QueryPlanner:
    """규칙 기반 질의 분류로 검색 전략을 선택한다."""

    RELATIONAL_PATTERNS = ["관계", "연결", "영향", "차이", "비교", "원인", "연관"]
    FACTUAL_PATTERNS = ["정의", "무엇", "언제", "어디서", "몇", "기준", "요건"]

    def plan(self, query: str) -> dict:
        query_type = self._classify(query)
        plans = {
            "relational": {
                "mode": SearchMode.HYBRID,
                "vector_weight": 0.5,
                "keyword_weight": 0.0,
                "graph_enabled": True,
                "graph_weight": 0.5,
            },
            "factual": {
                "mode": SearchMode.HYBRID,
                "vector_weight": 0.5,
                "keyword_weight": 0.5,
                "graph_enabled": False,
                "graph_weight": 0.0,
            },
            "exploratory": {
                "mode": SearchMode.VECTOR_ONLY,
                "vector_weight": 1.0,
                "keyword_weight": 0.0,
                "graph_enabled": False,
                "graph_weight": 0.0,
            },
        }
        plan = dict(plans.get(query_type, plans["factual"]))
        plan["query_type"] = query_type
        plan["fallback_mode"] = SearchMode.HYBRID
        return plan

    def _classify(self, query: str) -> str:
        q = (query or "").strip()
        if any(p in q for p in self.RELATIONAL_PATTERNS):
            return "relational"
        if any(p in q for p in self.FACTUAL_PATTERNS):
            return "factual"
        return "exploratory"

