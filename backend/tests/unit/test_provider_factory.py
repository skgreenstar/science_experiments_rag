from app.config import RAGSettings, Settings
from app.services.generation.ollama import OllamaLLM
from app.services.providers import (
    ResizedEmbeddingProvider,
    build_embedding_provider,
    build_llm_provider,
)


class _DummyEmbedding:
    async def embed_documents(self, texts):
        return [[1.0, 2.0]]

    async def embed_query(self, text):
        return [1.0, 2.0]


def test_build_embedding_provider_ollama():
    env = Settings(ollama_url="http://localhost:11434")
    rag = RAGSettings(embedding_provider="ollama", embedding_model="bge-m3")
    provider = build_embedding_provider(env, rag)
    assert isinstance(provider, ResizedEmbeddingProvider)


def test_build_llm_provider_ollama():
    env = Settings(ollama_url="http://localhost:11434")
    rag = RAGSettings(llm_provider="ollama", llm_model="exaone3.5:7.8b")
    provider = build_llm_provider(env, rag)
    assert isinstance(provider, OllamaLLM)
    assert provider.model == "exaone3.5:7.8b"


def test_build_openai_provider_requires_api_key():
    env = Settings(openai_api_key=None)
    rag = RAGSettings(embedding_provider="openai", embedding_model="text-embedding-3-small")

    try:
        build_embedding_provider(env, rag)
        assert False, "Expected RuntimeError"
    except RuntimeError as e:
        assert "OPENAI_API_KEY" in str(e)


async def test_resized_embedding_provider_padding():
    provider = ResizedEmbeddingProvider(_DummyEmbedding(), target_dims=4)
    vector = await provider.embed_query("q")
    assert vector == [1.0, 2.0, 0.0, 0.0]
