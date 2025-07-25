from pydantic import BaseModel
from typing import List, Optional
from typing import List, Optional
from src.models.providers import LLMProvider, EmbeddingProvider


DEFAULT_TOP_K = 5


class QueryRequest(BaseModel):
    query: str
    filter_id: str
    top_k: Optional[int] = DEFAULT_TOP_K
    history: Optional[List[str]] = []
    llm_provider: Optional[LLMProvider] = LLMProvider.OPENAI
    embedding_provider: Optional[EmbeddingProvider] = EmbeddingProvider.OPENAI_EMBEDDINGS


class VectorSearchQueryRequest(BaseModel):
    query: str
    filter_id: str
    top_k: Optional[int] = DEFAULT_TOP_K
    embedding_provider: Optional[EmbeddingProvider] = EmbeddingProvider.OPENAI_EMBEDDINGS


class VectorizationRequest(BaseModel):
    embedding_provider: Optional[EmbeddingProvider] = EmbeddingProvider.OPENAI_EMBEDDINGS
    filter_id: str