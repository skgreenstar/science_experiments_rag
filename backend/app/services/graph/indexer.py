"""그래프 인덱서: Chunk -> Neo4j."""
from __future__ import annotations

from app.services.chunking.base import Chunk
from app.services.graph.extractor import GraphExtractor
from app.services.graph.neo4j_client import Neo4jClient


class GraphIndexer:
    def __init__(self, neo4j_client: Neo4jClient, extractor: GraphExtractor) -> None:
        self.neo4j = neo4j_client
        self.extractor = extractor

    async def index_document(self, doc_id: str, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            chunk_id = str((chunk.metadata or {}).get("_chunk_id") or "")
            if not chunk_id:
                continue
            entities, relations = await self.extractor.extract(chunk.content)
            await self.neo4j.upsert_chunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                metadata=chunk.metadata or {},
                entities=entities,
                relations=relations,
            )

    async def delete_document(self, doc_id: str) -> None:
        await self.neo4j.delete_document(doc_id)
