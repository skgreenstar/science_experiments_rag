from app.services.fusion.search_policy import SearchMode
from app.services.planner.query_planner import QueryPlanner


def test_query_planner_relational():
    planner = QueryPlanner()
    plan = planner.plan("산화와 환원의 관계는?")
    assert plan["query_type"] == "relational"
    assert plan["mode"] == SearchMode.HYBRID
    assert plan["graph_enabled"] is True
    assert plan["graph_weight"] == 0.5
    assert plan["fallback_mode"] == SearchMode.HYBRID


def test_query_planner_factual():
    planner = QueryPlanner()
    plan = planner.plan("광합성의 정의는 무엇인가?")
    assert plan["query_type"] == "factual"
    assert plan["mode"] == SearchMode.HYBRID
    assert plan["graph_enabled"] is False


def test_query_planner_exploratory():
    planner = QueryPlanner()
    plan = planner.plan("빛에 대해 알려줘")
    assert plan["query_type"] == "exploratory"
    assert plan["mode"] == SearchMode.VECTOR_ONLY
    assert plan["graph_enabled"] is False

