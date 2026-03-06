"""모니터링 API 라우터.

Langfuse 트레이스 프록시 및 시스템 통계를 제공한다.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import Document, DocumentStatus, get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["monitoring"])


# --- Schemas ---


class MonitoringStatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    today_queries: int
    avg_response_time_ms: float


class TraceListResponse(BaseModel):
    items: list[dict[str, Any]]
    total: int


class CostResponse(BaseModel):
    total_cost: float
    period: str
    breakdown: list[dict[str, Any]]


# --- Stats ---


@router.get("/monitoring/stats", response_model=MonitoringStatsResponse)
async def get_stats(db: AsyncSession = Depends(get_db)):
    """집계 통계를 반환한다."""
    # 문서 수
    doc_count_result = await db.execute(
        select(func.count(Document.id)).where(Document.status == DocumentStatus.INDEXED)
    )
    total_documents = doc_count_result.scalar() or 0

    # 총 chunk 수 (문서 모델의 chunk_count 합산)
    chunk_sum_result = await db.execute(
        select(func.coalesce(func.sum(Document.chunk_count), 0)).where(
            Document.status == DocumentStatus.INDEXED
        )
    )
    total_chunks = chunk_sum_result.scalar() or 0

    # 오늘 쿼리 수, 평균 응답 시간은 Langfuse에서 가져오지만
    # Langfuse 미연동 시 기본값 반환
    today_queries, avg_response = await _get_langfuse_query_stats()

    return MonitoringStatsResponse(
        total_documents=total_documents,
        total_chunks=total_chunks,
        today_queries=today_queries,
        avg_response_time_ms=avg_response,
    )


# --- Traces (Langfuse 프록시) ---


@router.get("/monitoring/traces", response_model=TraceListResponse)
async def list_traces():
    """Langfuse 트레이스 목록을 UI 친화 스키마로 변환해 반환한다."""
    traces = await _langfuse_api_get("/api/public/traces", params={"limit": 50})
    if traces is None:
        return TraceListResponse(items=[], total=0)

    raw_items = traces.get("data", [])
    items = [_normalize_trace_item(item) for item in raw_items]
    items = await _enrich_missing_queries(items)
    return TraceListResponse(items=items, total=len(items))


@router.get("/monitoring/traces/{trace_id}")
async def get_trace(trace_id: str):
    """Langfuse 트레이스 상세를 UI 친화 스키마로 변환해 반환한다."""
    trace = await _langfuse_api_get(f"/api/public/traces/{trace_id}")
    if trace is None:
        return {}

    raw = trace.get("data", trace)
    normalized = _normalize_trace_item(raw)
    observations = raw.get("observations", [])
    normalized["spans"] = [_normalize_span_item(obs) for obs in observations]
    return normalized


# --- Costs ---


@router.get("/monitoring/costs", response_model=CostResponse)
async def get_costs():
    """비용 추적 정보를 반환한다."""
    # Langfuse 미연동 시 기본값
    return CostResponse(total_cost=0.0, period="today", breakdown=[])


# --- Internal helpers ---


async def _langfuse_api_get(path: str, params: dict | None = None) -> dict | None:
    """Langfuse REST API를 호출한다. 키 미설정 시 None 반환."""
    env = get_settings()
    if not env.langfuse_public_key or not env.langfuse_secret_key:
        return None

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{env.langfuse_host}{path}",
                params=params,
                auth=(env.langfuse_public_key, env.langfuse_secret_key),
                timeout=10.0,
            )
            if resp.status_code == 200:
                return resp.json()
            return None
    except Exception as e:
        logger.warning("Langfuse API call failed: %s", e)
        return None


async def _get_langfuse_query_stats() -> tuple[int, float]:
    """Langfuse에서 오늘의 쿼리 수, 평균 응답 시간을 조회한다."""
    env = get_settings()
    if not env.langfuse_public_key or not env.langfuse_secret_key:
        return 0, 0.0

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{env.langfuse_host}/api/public/traces",
                # Langfuse Public API의 limit 상한(100)을 넘기면 400 에러가 발생한다.
                params={"limit": 100},
                auth=(env.langfuse_public_key, env.langfuse_secret_key),
                timeout=10.0,
            )
            if resp.status_code != 200:
                return 0, 0.0

            data = resp.json().get("data", [])
            durations = [_extract_duration_ms(item) for item in data]
            durations = [d for d in durations if d > 0]
            avg_duration = sum(durations) / len(durations) if durations else 0.0
            return len(data), round(avg_duration, 2)
    except Exception:
        return 0, 0.0


async def _enrich_missing_queries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing_ids = [
        item.get("id")
        for item in items
        if not str(item.get("query", "")).strip() or not str(item.get("output", "")).strip()
    ]
    missing_ids = [trace_id for trace_id in missing_ids if isinstance(trace_id, str) and trace_id]
    if not missing_ids:
        return items

    async def _fetch_query(trace_id: str) -> tuple[str, str, str]:
        trace = await _langfuse_api_get(f"/api/public/traces/{trace_id}")
        if not trace:
            return trace_id, "", ""
        raw = trace.get("data", trace)
        return trace_id, _extract_query_from_trace(raw), _extract_output_from_trace(raw)

    # 목록 성능 보호를 위해 동시 상세 조회를 제한한다.
    semaphore = asyncio.Semaphore(8)

    async def _fetch_with_limit(trace_id: str) -> tuple[str, str, str]:
        async with semaphore:
            return await _fetch_query(trace_id)

    fetched = await asyncio.gather(*[_fetch_with_limit(trace_id) for trace_id in missing_ids])
    query_map = {trace_id: query for trace_id, query, _ in fetched if query.strip()}
    output_map = {trace_id: output for trace_id, _, output in fetched if output.strip()}
    for item in items:
        trace_id = item.get("id")
        if not isinstance(trace_id, str):
            continue
        if trace_id in query_map:
            item["query"] = query_map[trace_id]
        if trace_id in output_map:
            item["output"] = output_map[trace_id]
    return items


def _normalize_trace_item(item: dict[str, Any]) -> dict[str, Any]:
    query = _extract_query_from_trace(item)
    return {
        "id": item.get("id", ""),
        "query": query,
        "output": _extract_output_from_trace(item),
        "total_duration_ms": _extract_duration_ms(item),
        "status": _extract_status(item),
        "spans": [],
        "created_at": _extract_created_at(item),
    }


def _normalize_span_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": item.get("name") or item.get("type") or "span",
        "duration_ms": _duration_from_time_range(item.get("startTime"), item.get("endTime")),
        "status": "error" if item.get("level") == "ERROR" else "success",
        "metadata": item.get("metadata") or {},
    }


def _extract_query(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("query", "question", "original_query", "input", "prompt", "text", "variants", "queries"):
            v = value.get(key)
            extracted = _extract_query(v)
            if extracted.strip():
                return extracted
        return ""
    if isinstance(value, list):
        for v in value:
            extracted = _extract_query(v)
            if extracted.strip():
                return extracted
        return ""
    if value is None:
        return ""
    return str(value)


def _extract_query_from_trace(item: dict[str, Any]) -> str:
    """Trace 루트/observation에서 사용자 쿼리를 최대한 복원한다."""
    # 1) Trace 루트 input 우선
    query = _extract_query(item.get("input"))
    if query.strip():
        return query

    # 2) Trace 루트 metadata에서 보조 탐색
    query = _extract_query(item.get("metadata"))
    if query.strip():
        return query

    # 3) Observation input/output/metadata 순회 탐색
    observations = item.get("observations")
    if isinstance(observations, list):
        for obs in observations:
            if not isinstance(obs, dict):
                continue
            for field in ("input", "output", "metadata"):
                query = _extract_query(obs.get(field))
                if query.strip():
                    return query

    return ""


def _extract_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("answer", "final_answer", "response", "text", "content", "output", "generated_text"):
            v = value.get(key)
            extracted = _extract_output(v)
            if extracted.strip():
                return extracted
        return ""
    if isinstance(value, list):
        for v in value:
            extracted = _extract_output(v)
            if extracted.strip():
                return extracted
        return ""
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _extract_output_from_trace(item: dict[str, Any]) -> str:
    # 1) Trace 루트 output 우선
    output = _extract_output(item.get("output"))
    if output.strip():
        return output

    observations = item.get("observations")
    if not isinstance(observations, list):
        return ""

    # 2) 답변 생성 관측치 우선
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        name = str(obs.get("name", "")).lower()
        obs_type = str(obs.get("type", "")).lower()
        if "answer" in name or obs_type == "generation":
            output = _extract_output(obs.get("output"))
            if output.strip():
                return output

    # 3) 기타 관측치 output에서 fallback
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        output = _extract_output(obs.get("output"))
        if output.strip():
            return output

    return ""


def _extract_duration_ms(item: dict[str, Any]) -> float:
    # Langfuse public traces의 latency/duration 값은 초 단위(float)로 내려올 수 있다.
    # UI는 ms 단위를 기대하므로 필드별로 단위를 명시적으로 맞춘다.
    v = item.get("latencyMs")
    if isinstance(v, (int, float)):
        return float(v)

    v = item.get("latency")
    if isinstance(v, (int, float)):
        return float(v) * 1000.0

    v = item.get("duration_ms")
    if isinstance(v, (int, float)):
        return float(v)

    v = item.get("duration")
    if isinstance(v, (int, float)):
        # duration은 초 단위로 내려오는 케이스를 우선 처리
        return float(v) * 1000.0

    v = item.get("total_duration_ms")
    if isinstance(v, (int, float)):
        return float(v)

    return _duration_from_time_range(item.get("timestamp"), item.get("updatedAt"))


def _extract_status(item: dict[str, Any]) -> str:
    level = str(item.get("level", "")).upper()
    if level == "ERROR" or item.get("statusMessage"):
        return "error"
    return "success"


def _extract_created_at(item: dict[str, Any]) -> str:
    for key in ("timestamp", "createdAt", "created_at"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return datetime.now(timezone.utc).isoformat()


def _duration_from_time_range(start: Any, end: Any) -> float:
    start_dt = _parse_iso_datetime(start)
    end_dt = _parse_iso_datetime(end)
    if start_dt and end_dt:
        return max((end_dt - start_dt).total_seconds() * 1000.0, 0.0)
    return 0.0


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
