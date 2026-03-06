"""Neo4j 비동기 클라이언트."""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable

from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled and bool(uri and user and password)
        self._driver = (
            AsyncGraphDatabase.driver(uri, auth=(user, password))
            if self.enabled
            else None
        )

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()

    async def upsert_chunk(
        self,
        *,
        doc_id: str,
        chunk_id: str,
        chunk_index: int,
        content: str,
        metadata: dict | None,
        entities: Iterable[dict] | Iterable[str],
        relations: Iterable[dict] | None = None,
    ) -> None:
        if not self.enabled or self._driver is None:
            return
        normalized_entities: list[dict[str, str]] = []
        seen_entity_names: set[str] = set()
        for idx, item in enumerate(entities):
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                etype = str(item.get("type") or "기타").strip()[:32]
            else:
                name = str(item).strip()
                if not name:
                    continue
                etype = "기타"
            if name in seen_entity_names:
                continue
            seen_entity_names.add(name)
            normalized_entities.append({"id": f"e{idx+1}", "name": name, "type": etype})

        normalized_relations: list[dict[str, str]] = []
        seen_rel: set[tuple[str, str, str]] = set()
        for item in relations or []:
            if not isinstance(item, dict):
                continue
            src = str(item.get("from", "")).strip()
            dst = str(item.get("to", "")).strip()
            rtype = str(item.get("type", "")).strip()[:64]
            if not src or not dst or not rtype:
                continue
            key = (src, dst, rtype)
            if key in seen_rel:
                continue
            seen_rel.add(key)
            normalized_relations.append({"from": src, "to": dst, "type": rtype})

        async with self._driver.session() as session:
            await session.execute_write(
                self._upsert_chunk_tx,
                doc_id,
                chunk_id,
                chunk_index,
                content,
                metadata or {},
                normalized_entities,
                normalized_relations,
            )

    @staticmethod
    async def _upsert_chunk_tx(
        tx,
        doc_id,
        chunk_id,
        chunk_index,
        content,
        metadata,
        entities,
        relations,
    ):
        await tx.run(
            """
            MERGE (d:Document {id: $doc_id})
            MERGE (c:Chunk {id: $chunk_id})
            SET c.document_id = $doc_id,
                c.chunk_index = $chunk_index,
                c.content = $content,
                c.metadata_json = $metadata_json
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            doc_id=doc_id,
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            content=content,
            metadata_json=json.dumps(metadata, ensure_ascii=False),
        )
        await tx.run(
            """
            MATCH ()-[r:RELATED {source_chunk_id: $chunk_id}]->()
            DELETE r
            """,
            chunk_id=chunk_id,
        )
        await tx.run(
            """
            MATCH (c:Chunk {id: $chunk_id})-[m:MENTIONS]->(:Entity)
            DELETE m
            """,
            chunk_id=chunk_id,
        )
        for entity in entities:
            await tx.run(
                """
                MERGE (e:Entity {name: $name})
                SET e.type = $etype
                MERGE (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                name=entity["name"],
                etype=entity["type"],
                chunk_id=chunk_id,
            )
        for rel in relations:
            await tx.run(
                """
                MATCH (s:Entity {name: $src_name})
                MATCH (t:Entity {name: $dst_name})
                MERGE (s)-[r:RELATED {type: $rel_type, source_chunk_id: $chunk_id}]->(t)
                """,
                src_name=rel["from"],
                dst_name=rel["to"],
                rel_type=rel["type"],
                chunk_id=chunk_id,
            )

    async def delete_document(self, doc_id: str) -> None:
        if not self.enabled or self._driver is None:
            return
        async with self._driver.session() as session:
            await session.execute_write(
                lambda tx: tx.run(
                    """
                    MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                    DETACH DELETE c
                    """,
                    doc_id=doc_id,
                )
            )
            await session.execute_write(
                lambda tx: tx.run(
                    """
                    MATCH (e:Entity)
                    WHERE NOT (e)<-[:MENTIONS]-(:Chunk)
                    DETACH DELETE e
                    """
                )
            )
            await session.execute_write(
                lambda tx: tx.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    DETACH DELETE d
                    """,
                    doc_id=doc_id,
                )
            )

    async def search_chunks(
        self,
        *,
        tokens: list[str],
        top_k: int,
        doc_id: str | None = None,
    ) -> list[dict]:
        if not self.enabled or self._driver is None or not tokens:
            return []

        where_doc = "AND c.document_id = $doc_id" if doc_id else ""
        cypher = f"""
        MATCH (c:Chunk)<-[:HAS_CHUNK]-(:Document)
        WHERE 1=1 {where_doc}
        CALL {{
            WITH c
            MATCH (e:Entity)<-[:MENTIONS]-(c)
            WHERE any(token IN $tokens WHERE
                toLower(e.name) = toLower(token)
                OR toLower(e.name) STARTS WITH toLower(token)
                OR (size(token) >= 3 AND toLower(e.name) CONTAINS toLower(token))
            )
            RETURN collect(DISTINCT e.name) AS matched_entities, count(DISTINCT e) AS entity_score
        }}
        CALL {{
            WITH c
            OPTIONAL MATCH (c)-[:MENTIONS]->(s:Entity)-[r:RELATED]->(t:Entity)
            WHERE any(token IN $tokens WHERE
                toLower(s.name) = toLower(token)
                OR toLower(s.name) STARTS WITH toLower(token)
                OR toLower(t.name) = toLower(token)
                OR toLower(t.name) STARTS WITH toLower(token)
                OR (size(token) >= 3 AND toLower(s.name) CONTAINS toLower(token))
                OR (size(token) >= 3 AND toLower(t.name) CONTAINS toLower(token))
                OR toLower(r.type) CONTAINS toLower(token)
            )
            RETURN collect(DISTINCT r.type) AS matched_relations, count(DISTINCT r) AS relation_score
        }}
        WITH c, matched_entities, matched_relations, (entity_score + relation_score * 2.0) AS score
        WHERE score > 0
        RETURN c.id AS chunk_id,
               c.document_id AS document_id,
               c.content AS content,
               c.metadata_json AS metadata_json,
               matched_entities,
               matched_relations,
               score
        ORDER BY score DESC, c.chunk_index ASC
        LIMIT $top_k
        """

        params = {"tokens": tokens, "top_k": top_k}
        if doc_id:
            params["doc_id"] = doc_id

        async with self._driver.session() as session:
            result = await session.run(cypher, params)
            rows = await result.data()
            return rows
