"""Neo4j 기반 Graph Retriever."""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

from app.services.graph.neo4j_client import Neo4jClient
from app.services.search.base_retriever import BaseRetriever, RetrievalResult

_QUERY_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9._:-]{1,}|[가-힣]{2,}")


class GraphRetriever(BaseRetriever):
    def __init__(self, neo4j_client: Neo4jClient) -> None:
        self.neo4j = neo4j_client

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        raw_tokens = _QUERY_TOKEN_PATTERN.findall(query or "")
        tokens: list[str] = []
        seen: set[str] = set()
        for token in raw_tokens:
            t = token.strip().lower()
            if not t or t.isdigit():
                continue
            # 영문 토큰은 3자 이상만 허용하여 과매칭을 줄인다.
            if re.match(r"^[a-z0-9._:-]+$", t) and len(t) < 3:
                continue
            if t in seen:
                continue
            seen.add(t)
            tokens.append(t)
            if len(tokens) >= 10:
                break
        if not tokens:
            return []
        doc_id = filters.get("doc_id") if filters else None
        rows = await self.neo4j.search_chunks(tokens=tokens, top_k=top_k, doc_id=doc_id)
        results: list[RetrievalResult] = []
        for row in rows:
            try:
                cid = uuid.UUID(str(row.get("chunk_id")))
                did = uuid.UUID(str(row.get("document_id")))
            except (ValueError, TypeError):
                continue
            metadata_raw = row.get("metadata_json")
            metadata: dict[str, Any] = {}
            if isinstance(metadata_raw, str) and metadata_raw:
                try:
                    parsed = json.loads(metadata_raw)
                    if isinstance(parsed, dict):
                        metadata = parsed
                except json.JSONDecodeError:
                    metadata = {}
            results.append(
                RetrievalResult(
                    doc_id=did,
                    chunk_id=cid,
                    score=float(row.get("score", 0.0)),
                    source="graph",
                    content=str(row.get("content", "")),
                    metadata={
                        **metadata,
                        "matched_entities": row.get("matched_entities", []),
                        "matched_relations": row.get("matched_relations", []),
                    },
                )
            )
        return results
