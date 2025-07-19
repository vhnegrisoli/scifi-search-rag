from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from src.vector_store.vector_store import VectorStore
from src.llm.llm_integration import LLMCall


router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    filter_id: Optional[str] = None
    history: Optional[List[str]] = []


@router.post("/query")
def query_rag(req: QueryRequest):
    vs = VectorStore()
    docs = vs.search_vector_store(
        filter_id=req.filter_id,
        query=req.query,
        k=5
    )
    llm = LLMCall()
    response = llm.call_llm(
        message=req.query,
        docs=docs,
        history=req.history
    )
    return {"response": response.content}