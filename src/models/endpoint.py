from pydantic import BaseModel
from typing import List, Optional
from src.config.config import AppConfig
from typing import List, Optional


config = AppConfig().get_config()


class QueryRequest(BaseModel):
    query: str
    filter_id: Optional[str] = config.filter_id
    top_k: Optional[int] = config.top_k
    history: Optional[List[str]] = []
