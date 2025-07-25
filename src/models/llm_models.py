from pydantic import BaseModel
from typing import List, Optional


class LLMUsageResponse(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    content: str
    total_docs: int = 0
    usage: Optional[LLMUsageResponse] = None
    docs: List[str] = []


class VectorizationResponse(BaseModel):
    status: str
    filter_id: str
    message: str


class VectorSearchResponse(BaseModel):
    docs: List[str] = []