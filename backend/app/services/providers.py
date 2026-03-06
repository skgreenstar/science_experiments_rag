"""임베딩/LLM 프로바이더 팩토리."""
from __future__ import annotations

from app.config import RAGSettings, Settings
from app.services.embedding.base import EmbeddingProvider
from app.services.embedding.ollama import OllamaEmbedding
from app.services.embedding.openai import OpenAIEmbedding
from app.services.generation.base import LLMProvider
from app.services.generation.claude import ClaudeLLM
from app.services.generation.ollama import OllamaLLM
from app.services.generation.openai import OpenAILLM


class ResizedEmbeddingProvider:
    """임베딩 벡터 차원을 target_dims로 맞춘다.

    - 짧으면 0으로 패딩
    - 길면 잘라냄
    """

    def __init__(self, base: EmbeddingProvider, target_dims: int):
        self.base = base
        self.target_dims = target_dims

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = await self.base.embed_documents(texts)
        return [self._resize(v) for v in vectors]

    async def embed_query(self, text: str) -> list[float]:
        vector = await self.base.embed_query(text)
        return self._resize(vector)

    def _resize(self, vector: list[float]) -> list[float]:
        size = len(vector)
        if size == self.target_dims:
            return vector
        if size > self.target_dims:
            return vector[: self.target_dims]
        return vector + [0.0] * (self.target_dims - size)


def build_embedding_provider(env: Settings, rag: RAGSettings) -> EmbeddingProvider:
    provider = rag.embedding_provider.lower()
    if provider == "openai":
        if not env.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for embedding_provider=openai")
        return OpenAIEmbedding(
            api_key=env.openai_api_key,
            model=rag.embedding_model,
            dimensions=env.rag_embedding_dimensions,
        )
    if provider == "ollama":
        base = OllamaEmbedding(url=env.ollama_url, model=rag.embedding_model)
        if env.rag_embedding_dimensions > 0:
            return ResizedEmbeddingProvider(base, env.rag_embedding_dimensions)
        return base
    raise ValueError(f"Unsupported embedding provider: {rag.embedding_provider}")


def build_llm_provider(
    env: Settings,
    rag: RAGSettings,
    *,
    model: str | None = None,
    temperature: float | None = None,
) -> LLMProvider:
    provider = rag.llm_provider.lower()
    model_name = model or rag.llm_model
    temp = rag.llm_temperature if temperature is None else temperature

    if provider == "openai":
        if not env.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for llm_provider=openai")
        return OpenAILLM(api_key=env.openai_api_key, model=model_name, temperature=temp)
    if provider == "ollama":
        return OllamaLLM(url=env.ollama_url, model=model_name, temperature=temp)
    if provider == "anthropic":
        if not env.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for llm_provider=anthropic")
        return ClaudeLLM(api_key=env.anthropic_api_key, model=model_name)
    raise ValueError(f"Unsupported llm provider: {rag.llm_provider}")
