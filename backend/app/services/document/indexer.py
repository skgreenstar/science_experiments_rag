"""듀얼 인덱서: PGVector + Elasticsearch 동시 저장."""
import uuid

from app.services.chunking.base import Chunk
from app.services.embedding.base import EmbeddingProvider


class DocumentIndexer:
    """임베딩 생성 후 PGVector와 Elasticsearch에 동시 인덱싱."""

    def __init__(self, embedding_provider: EmbeddingProvider, pg_store, es_store, graph_indexer=None):
        self.embedding_provider = embedding_provider
        self.pg_store = pg_store
        self.es_store = es_store
        self.graph_indexer = graph_indexer

    async def index(self, doc_id: str, chunks: list[Chunk]):
        if not chunks:
            return

        # 검색 저장소 간 chunk_id 일관성을 보장한다.
        used_chunk_ids: set[str] = set()
        for chunk in chunks:
            metadata = dict(chunk.metadata) if isinstance(chunk.metadata, dict) else {}
            chunk_id = str(metadata.get("_chunk_id") or "").strip()
            if not chunk_id or chunk_id in used_chunk_ids:
                chunk_id = str(uuid.uuid4())
            metadata["_chunk_id"] = chunk_id
            chunk.metadata = metadata
            used_chunk_ids.add(chunk_id)

        # 1. 임베딩 생성 (배치)
        texts = [c.content for c in chunks]
        embeddings = await self.embedding_provider.embed_documents(texts)

        # 2. PGVector 저장
        await self.pg_store.write(chunks, embeddings, meta={"doc_id": doc_id})

        # 3. Elasticsearch 인덱싱
        await self.es_store.write(chunks, meta={"doc_id": doc_id})

        # 4. Graph 인덱싱 (선택)
        if self.graph_indexer is not None:
            await self.graph_indexer.index_document(doc_id, chunks)

    async def delete(self, doc_id: str):
        await self.pg_store.delete(filters={"doc_id": doc_id})
        await self.es_store.delete(filters={"doc_id": doc_id})
        if self.graph_indexer is not None:
            await self.graph_indexer.delete_document(doc_id)
