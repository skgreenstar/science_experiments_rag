"""Graph extractor.

LLM(JSON) 기반 엔티티/관계 추출을 우선 사용하고, 실패 시 토큰 기반으로 폴백한다.
"""
from __future__ import annotations

import json
import re
from typing import Any

from app.services.generation.base import LLMProvider

_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9._:-]{1,}|[가-힣]{2,}")
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)
_STOPWORDS = {
    "그리고",
    "하지만",
    "에서",
    "으로",
    "대한",
    "관련",
    "문서",
    "내용",
    "질문",
    "검색",
    "테스트",
}


class GraphExtractor:
    PROMPT = """
다음 텍스트에서 엔티티와 관계를 추출하세요.
반드시 JSON 객체만 반환하세요. 설명 문장/마크다운 금지.

스키마:
{
  "entities": [{"id":"e1","type":"개념|인물|조직|장소|수치|기타","name":"..."}],
  "relations": [{"from":"e1","to":"e2","type":"..."}]
}

규칙:
1) relation의 from/to는 entities의 id를 사용
2) type은 짧고 명확한 한국어 동사/관계명
3) 텍스트에 없는 사실은 만들지 말 것

텍스트:
{text}
""".strip()

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.llm = llm

    async def extract(self, text: str, max_entities: int = 30) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        if self.llm is None:
            return self._fallback(text, max_entities=max_entities)

        try:
            raw = await self.llm.generate(self.PROMPT.format(text=(text or "")[:4000]))
            payload = self._parse_json(raw)
            entities, relations = self._normalize(payload, max_entities=max_entities)
            if entities:
                return entities, relations
        except Exception:
            # LLM 추출 실패 시 토큰 기반 폴백
            pass

        return self._fallback(text, max_entities=max_entities)

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        text = (raw or "").strip()
        block = _JSON_BLOCK.search(text)
        if block:
            text = block.group(1).strip()
        return json.loads(text)

    def _normalize(
        self,
        payload: dict[str, Any],
        *,
        max_entities: int,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        raw_entities = payload.get("entities") if isinstance(payload, dict) else []
        raw_relations = payload.get("relations") if isinstance(payload, dict) else []

        entities: list[dict[str, str]] = []
        id_to_name: dict[str, str] = {}
        seen_names: set[str] = set()

        if isinstance(raw_entities, list):
            for idx, item in enumerate(raw_entities):
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if len(name) < 2 or name in seen_names:
                    continue
                eid = str(item.get("id") or f"e{idx+1}").strip()
                etype = str(item.get("type") or "기타").strip()[:32]
                seen_names.add(name)
                id_to_name[eid] = name
                entities.append({"id": eid, "type": etype, "name": name})
                if len(entities) >= max_entities:
                    break

        relations: list[dict[str, str]] = []
        seen_rel: set[tuple[str, str, str]] = set()
        if isinstance(raw_relations, list):
            for item in raw_relations:
                if not isinstance(item, dict):
                    continue
                src = str(item.get("from", "")).strip()
                dst = str(item.get("to", "")).strip()
                rtype = str(item.get("type", "")).strip()[:64]
                src_name = id_to_name.get(src)
                dst_name = id_to_name.get(dst)
                if not src_name or not dst_name or not rtype:
                    continue
                key = (src_name, dst_name, rtype)
                if key in seen_rel:
                    continue
                seen_rel.add(key)
                relations.append({"from": src_name, "to": dst_name, "type": rtype})

        return entities, relations

    def _fallback(self, text: str, *, max_entities: int = 30) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        entities = self.extract_entities(text, max_entities=max_entities)
        normalized = [
            {"id": f"e{i+1}", "type": "기타", "name": name}
            for i, name in enumerate(entities)
        ]
        relations: list[dict[str, str]] = []
        for i in range(len(entities) - 1):
            relations.append({"from": entities[i], "to": entities[i + 1], "type": "연관"})
            if len(relations) >= 20:
                break
        return normalized, relations

    def extract_entities(self, text: str, max_entities: int = 30) -> list[str]:
        candidates = _TOKEN_PATTERN.findall(text or "")
        entities: list[str] = []
        seen: set[str] = set()
        for token in candidates:
            t = token.strip()
            if len(t) < 2:
                continue
            if t in _STOPWORDS:
                continue
            if t in seen:
                continue
            seen.add(t)
            entities.append(t)
            if len(entities) >= max_entities:
                break
        return entities
