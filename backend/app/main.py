from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import documents, evaluation, health, monitoring, search, settings, system, watcher
from app.config import get_settings
from app.exceptions import RAGException
from app.models.database import init_db
from app.monitoring.langfuse import LangfuseMonitor


@asynccontextmanager
async def lifespan(app: FastAPI):
    env = get_settings()
    init_db(env.database_url)

    # Langfuse 모니터 초기화
    app.state.langfuse_monitor = LangfuseMonitor(
        public_key=env.langfuse_public_key,
        secret_key=env.langfuse_secret_key,
        host=env.langfuse_host,
    )

    # 검색 오케스트레이터 초기화
    from app.api import search as search_api
    from app.models.database import _async_session_factory
    from app.services.generation.evidence_extractor import EvidenceExtractor
    from app.services.guardrails.numeric_verifier import NumericVerifier
    from app.services.hyde.generator import HyDEGenerator
    from app.services.providers import build_embedding_provider, build_llm_provider
    from app.services.reranking.korean import KoreanCrossEncoder
    from app.services.search.hybrid import HybridSearchOrchestrator
    from app.services.search.keyword_es import ElasticsearchNoriEngine
    from app.services.search.multi_query import MultiQueryGenerator
    from app.services.search.query_expander import QueryExpander
    from app.services.search.question_classifier import QuestionClassifier
    from app.services.search.vector import VectorSearchEngine
    from app.services.settings import SettingsService

    # RAG 설정 로드 (provider/model 등)
    async with _async_session_factory() as _s:
        _ss = SettingsService(db=_s)
        _rag = await _ss.get_settings()
    embedder = build_embedding_provider(env, _rag)
    llm = build_llm_provider(env, _rag)

    vector_engine = VectorSearchEngine(session_factory=_async_session_factory)
    keyword_engine = ElasticsearchNoriEngine(es_url=env.elasticsearch_url)
    reranker = KoreanCrossEncoder()
    hyde_generator = HyDEGenerator(llm=llm)

    # Phase 10: Query Expander
    query_expander = QueryExpander(llm=llm)

    # Phase 11: 멀티쿼리, 질문 분류, 근거 추출, 숫자 검증
    multi_query_generator = MultiQueryGenerator(llm=llm)
    question_classifier = QuestionClassifier()
    evidence_extractor = EvidenceExtractor(llm=llm)
    numeric_verifier = NumericVerifier()

    orchestrator = HybridSearchOrchestrator(
        embedder=embedder,
        vector_engine=vector_engine,
        keyword_engine=keyword_engine,
        reranker=reranker,
        hyde_generator=hyde_generator,
        llm=llm,
        langfuse_monitor=app.state.langfuse_monitor,
        query_expander=query_expander,
        multi_query_generator=multi_query_generator,
        question_classifier=question_classifier,
        evidence_extractor=evidence_extractor,
        numeric_verifier=numeric_verifier,
    )
    search_api.set_orchestrator(orchestrator)

    settings_session = _async_session_factory()
    settings_service = SettingsService(db=settings_session)
    search_api.set_search_settings_service(settings_service)

    yield

    # 종료 시 정리
    await settings_session.close()
    app.state.langfuse_monitor.flush()


app = FastAPI(title="RAG", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.error_code, "message": str(exc)},
    )


app.include_router(health.router, prefix="/api")
app.include_router(settings.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(watcher.router, prefix="/api")
app.include_router(system.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(evaluation.router, prefix="/api")
app.include_router(monitoring.router, prefix="/api")
