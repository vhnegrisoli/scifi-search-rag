from pydantic import BaseModel
from typing import List, Optional
from src.config.config import AppConfig
from typing import List, Optional
from src.models.providers import LLMProvider, EmbeddingProvider


config = AppConfig().get_config()


class QueryRequest(BaseModel):
    query: str
    filter_id: Optional[str] = config.filter_id
    top_k: Optional[int] = config.top_k
    history: Optional[List[str]] = []
    llm_provider: Optional[LLMProvider] = LLMProvider.OPENAI
    ebedding_provider: Optional[EmbeddingProvider] = EmbeddingProvider.OPENAI_EMBEDDINGS
