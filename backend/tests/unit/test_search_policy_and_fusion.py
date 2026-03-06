from __future__ import annotations

import uuid

from app.services.fusion.rrf import RRFFusion
from app.services.fusion.search_policy import SearchMode, SearchPolicy
from app.services.search.base_retriever import RetrievalResult


class _DummyRetriever:
    async def retrieve(self, query: str, top_k: int = 10, filters=None):
        return []


def _rr(
    *,
    doc_id: uuid.UUID | None = None,
    chunk_id: uuid.UUID | None = None,
    score: float = 0.0,
    source: str = "vector",
) -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id or uuid.uuid4(),
        chunk_id=chunk_id or uuid.uuid4(),
        score=score,
        source=source,
        content="content",
        metadata={},
    )


def test_search_policy_modes():
    policy = SearchPolicy(vector_retriever=_DummyRetriever(), keyword_retriever=_DummyRetriever())

    plan = policy.get_plan(SearchMode.VECTOR_ONLY.value)
    assert len(plan) == 1
    assert plan[0].query_kind == "vector"

    plan = policy.get_plan(SearchMode.KEYWORD.value)
    assert len(plan) == 1
    assert plan[0].query_kind == "keyword"

    plan = policy.get_plan(SearchMode.HYBRID.value, vector_weight=0.7, keyword_weight=0.3)
    assert len(plan) == 2
    assert plan[0].weight == 0.7
    assert plan[1].weight == 0.3


def test_search_policy_auto_uses_question_category():
    policy = SearchPolicy(vector_retriever=_DummyRetriever(), keyword_retriever=_DummyRetriever())
    plan = policy.get_plan(SearchMode.AUTO.value, question_category="extraction", query="이름은 무엇?")
    assert len(plan) == 2
    assert plan[0].weight == 0.25
    assert plan[1].weight == 0.75


def test_search_policy_auto_relational_enables_graph():
    graph = _DummyRetriever()
    policy = SearchPolicy(
        vector_retriever=_DummyRetriever(),
        keyword_retriever=_DummyRetriever(),
        graph_retriever=graph,
    )
    plan = policy.get_plan(
        SearchMode.AUTO.value,
        graph_enabled=True,
        question_category="explanatory",
        query="A와 B의 관계를 비교해줘",
    )
    assert len(plan) == 3
    assert [p.name for p in plan] == ["vector_search", "keyword_search", "graph_search"]
    assert [p.weight for p in plan] == [0.2, 0.3, 0.5]


def test_search_policy_graph_only():
    graph = _DummyRetriever()
    policy = SearchPolicy(
        vector_retriever=_DummyRetriever(),
        keyword_retriever=_DummyRetriever(),
        graph_retriever=graph,
    )
    plan = policy.get_plan(SearchMode.GRAPH_ONLY.value, graph_enabled=True)
    assert len(plan) == 1
    assert plan[0].name == "graph_search"
    assert plan[0].weight == 1.0


def test_search_policy_graph_only_unavailable_returns_empty_plan():
    policy = SearchPolicy(
        vector_retriever=_DummyRetriever(),
        keyword_retriever=_DummyRetriever(),
        graph_retriever=None,
    )
    plan = policy.get_plan(SearchMode.GRAPH_ONLY.value, graph_enabled=False)
    assert plan == []


def test_rrf_fusion_combines_and_orders():
    shared_chunk = uuid.uuid4()
    shared_doc = uuid.uuid4()

    vec_results = [
        _rr(doc_id=shared_doc, chunk_id=shared_chunk, source="vector"),
        _rr(source="vector"),
    ]
    kw_results = [
        _rr(doc_id=shared_doc, chunk_id=shared_chunk, source="keyword"),
    ]

    fusion = RRFFusion()
    fused = fusion.combine([vec_results, kw_results], k=60, weights=[0.7, 0.3])

    assert len(fused) == 2
    # 동일 chunk가 양쪽 결과에 있으면 fused score가 단일 출처보다 높아야 한다.
    assert fused[0].chunk_id == shared_chunk
    assert fused[0].score > fused[1].score


def test_rrf_fusion_ignores_zero_weight_list():
    shared_chunk = uuid.uuid4()
    shared_doc = uuid.uuid4()
    vec_results = [_rr(doc_id=shared_doc, chunk_id=shared_chunk, source="vector")]
    kw_results = [_rr(doc_id=shared_doc, chunk_id=shared_chunk, source="keyword")]

    fusion = RRFFusion()
    fused = fusion.combine([vec_results, kw_results], k=60, weights=[1.0, 0.0])

    assert len(fused) == 1
    assert fused[0].chunk_id == shared_chunk
