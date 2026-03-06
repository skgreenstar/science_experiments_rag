import httpx
from fastapi import APIRouter

from app.config import get_settings

router = APIRouter()


async def check_db() -> bool:
    """PostgreSQL SELECT 1 실행."""
    try:
        from app.models.database import get_engine

        engine = get_engine()
        if engine is None:
            return False
        from sqlalchemy import text

        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def check_elasticsearch() -> bool:
    """Elasticsearch GET / 호출."""
    try:
        settings = get_settings()
        async with httpx.AsyncClient() as client:
            resp = await client.get(settings.elasticsearch_url, timeout=5.0)
            return resp.status_code == 200
    except Exception:
        return False


async def check_llm_provider() -> bool:
    """현재 LLM provider 연결 상태 확인."""
    try:
        settings = get_settings()
        provider = (settings.rag_llm_provider or "openai").lower()
        llm_model = settings.rag_llm_model

        if provider == "openai":
            if not settings.openai_api_key:
                return False
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                    timeout=5.0,
                )
                return resp.status_code == 200

        if provider == "ollama":
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{settings.ollama_url}/api/tags", timeout=5.0)
                if resp.status_code != 200:
                    return False
                if not llm_model:
                    return True
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                return any(name == llm_model or name.startswith(f"{llm_model}:") for name in models)

        if provider == "anthropic":
            return bool(settings.anthropic_api_key)

        return False
    except Exception:
        return False


async def check_redis() -> bool:
    """Redis PING."""
    try:
        import redis.asyncio as aioredis

        settings = get_settings()
        r = aioredis.from_url(settings.redis_url)
        await r.ping()
        await r.aclose()
        return True
    except Exception:
        return False


async def check_langfuse() -> bool:
    """Langfuse Public API 연결 확인 (키 필요)."""
    try:
        settings = get_settings()
        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            return False
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.langfuse_host}/api/public/projects",
                auth=(settings.langfuse_public_key, settings.langfuse_secret_key),
                timeout=5.0,
            )
            return resp.status_code == 200
    except Exception:
        return False


@router.get("/health")
async def health_check():
    db_ok = await check_db()
    es_ok = await check_elasticsearch()
    llm_ok = await check_llm_provider()
    redis_ok = await check_redis()
    langfuse_ok = await check_langfuse()

    return {
        "status": "ok",
        "components": {
            "database": "connected" if db_ok else "disconnected",
            "elasticsearch": "connected" if es_ok else "disconnected",
            "llm": "connected" if llm_ok else "disconnected",
            "redis": "connected" if redis_ok else "disconnected",
            "langfuse": "connected" if langfuse_ok else "disconnected",
        },
    }
