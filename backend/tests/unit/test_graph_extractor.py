from __future__ import annotations

import pytest

from app.services.graph.extractor import GraphExtractor


class _FakeLLM:
    def __init__(self, response: str):
        self.response = response

    async def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return self.response


@pytest.mark.asyncio
async def test_graph_extractor_parses_llm_json():
    llm = _FakeLLM(
        """
        {
          "entities":[
            {"id":"e1","type":"조직","name":"쿠팡"},
            {"id":"e2","type":"조직","name":"대형마트"}
          ],
          "relations":[
            {"from":"e1","to":"e2","type":"경쟁"}
          ]
        }
        """
    )
    extractor = GraphExtractor(llm=llm)
    entities, relations = await extractor.extract("쿠팡과 대형마트는 경쟁한다.")

    assert len(entities) == 2
    assert any(e["name"] == "쿠팡" for e in entities)
    assert relations == [{"from": "쿠팡", "to": "대형마트", "type": "경쟁"}]


@pytest.mark.asyncio
async def test_graph_extractor_fallback_when_invalid_json():
    llm = _FakeLLM("not-json")
    extractor = GraphExtractor(llm=llm)
    entities, relations = await extractor.extract("쿠팡은 이커머스 회사다.")

    assert len(entities) >= 1
    assert relations == []
