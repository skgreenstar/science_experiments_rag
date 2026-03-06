"""문서 인덱싱 비동기 태스크."""
import asyncio
import logging

from app.worker import celery_app

logger = logging.getLogger(__name__)


async def get_document_file_path(doc_id: str) -> str:
    """documents 테이블에서 doc_id로 파일 저장 경로를 조회."""
    from app.config import get_settings
    from app.models.database import Document, init_db, _async_session_factory
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        doc = await session.get(Document, doc_id)
        if not doc:
            raise ValueError(f"Document {doc_id} not found")
        file_path = doc.file_path

    await engine.dispose()
    return file_path


async def create_processor():
    """DocumentProcessor 인스턴스를 생성."""
    from app.config import get_settings
    from app.models.database import init_db, _async_session_factory
    from app.services.document.converter import DocumentConverter
    from app.services.document.indexer import DocumentIndexer
    from app.services.document.processor import DocumentProcessor
    from app.services.providers import build_embedding_provider, build_llm_provider
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # RAG 설정 로드 (embedding_model, contextual chunking 등)
    from app.services.settings import SettingsService
    settings_service = SettingsService()
    rag_settings_session = session_factory()
    settings_service._db = rag_settings_session
    rag_settings = await settings_service.get_settings()
    await rag_settings_session.close()

    # 임베딩: 설정 기반(provider/model)
    embedding = build_embedding_provider(settings, rag_settings)
    converter = DocumentConverter()

    from app.services.document.stores.pgvector_store import PgVectorStore
    from app.services.document.stores.elasticsearch_store import ElasticsearchStore
    from app.services.graph.extractor import GraphExtractor
    from app.services.graph.indexer import GraphIndexer
    from app.services.graph.neo4j_client import Neo4jClient

    pg_store = PgVectorStore(session_factory=session_factory)
    es_store = ElasticsearchStore(es_url=settings.elasticsearch_url)
    graph_indexer = None
    if settings.neo4j_enabled and rag_settings.graph_enabled:
        graph_llm = None
        try:
            graph_llm = build_llm_provider(
                settings,
                rag_settings,
                model=rag_settings.llm_model,
                temperature=rag_settings.llm_temperature,
            )
        except Exception as exc:
            logger.warning("Graph extractor LLM init failed, fallback extractor will be used: %s", exc)
        graph_client = Neo4jClient(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            enabled=True,
        )
        graph_indexer = GraphIndexer(graph_client, GraphExtractor(llm=graph_llm))
    indexer = DocumentIndexer(embedding, pg_store, es_store, graph_indexer=graph_indexer)

    # Contextual Chunking LLM 프로바이더 (조건부 생성)
    chunk_llm = None
    if rag_settings.contextual_chunking_enabled:
        chunk_llm = build_llm_provider(
            settings,
            rag_settings,
            model=rag_settings.contextual_chunking_model,
            temperature=rag_settings.llm_temperature,
        )

    session = session_factory()
    processor = DocumentProcessor(
        converter=converter,
        indexer=indexer,
        db_session=session,
        chunking_strategy=rag_settings.chunking_strategy,
        embedder=embedding,
        llm_provider=chunk_llm,
        contextual_chunking_enabled=rag_settings.contextual_chunking_enabled,
        contextual_chunking_max_doc_chars=rag_settings.contextual_chunking_max_doc_chars,
    )
    return processor


@celery_app.task(bind=True, max_retries=3)
def index_document_task(self, doc_id: str):
    """문서 인덱싱 비동기 태스크."""
    async def _run() -> None:
        file_path = await get_document_file_path(doc_id)
        processor = await create_processor()
        try:
            await processor.process(doc_id, file_path)
        finally:
            graph_indexer = getattr(processor.indexer, "graph_indexer", None)
            if graph_indexer is not None and getattr(graph_indexer, "neo4j", None) is not None:
                await graph_indexer.neo4j.close()
            await processor.db_session.close()

    try:
        asyncio.run(_run())
    except Exception as exc:
        self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, max_retries=3)
def index_document_to_graph(self, doc_id: str, chunks: list[dict]):
    """벡터/키워드와 별개로 그래프 인덱싱만 수행한다."""

    async def _run() -> None:
        from app.config import get_settings
        from app.services.graph.extractor import GraphExtractor
        from app.services.graph.neo4j_client import Neo4jClient
        from app.services.providers import build_llm_provider
        from app.services.settings import SettingsService
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        env = get_settings()
        if not env.neo4j_enabled:
            return

        engine = create_async_engine(env.database_url, echo=False)
        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as session:
            rag_settings = await SettingsService(db=session).get_settings()

        if not rag_settings.graph_enabled:
            await engine.dispose()
            return

        graph_llm = None
        try:
            graph_llm = build_llm_provider(
                env,
                rag_settings,
                model=rag_settings.llm_model,
                temperature=rag_settings.llm_temperature,
            )
        except Exception as exc:
            logger.warning("Graph-only task LLM init failed, fallback extractor will be used: %s", exc)

        neo4j = Neo4jClient(
            uri=env.neo4j_uri,
            user=env.neo4j_user,
            password=env.neo4j_password,
            enabled=True,
        )
        extractor = GraphExtractor(llm=graph_llm)
        try:
            for chunk in chunks:
                content = str(chunk.get("content") or "")
                chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
                if not content or not chunk_id:
                    continue
                entities, relations = await extractor.extract(content)
                await neo4j.upsert_chunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    chunk_index=int(chunk.get("chunk_index") or 0),
                    content=content,
                    metadata=chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {},
                    entities=entities,
                    relations=relations,
                )
        finally:
            await neo4j.close()
            await engine.dispose()

    try:
        asyncio.run(_run())
    except Exception as exc:
        self.retry(exc=exc, countdown=60)
