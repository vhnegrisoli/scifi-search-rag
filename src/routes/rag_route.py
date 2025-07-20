from fastapi import APIRouter, status, HTTPException
from src.models.endpoint import QueryRequest
from src.models.llm_models import LLMResponse
from src.services.rag_service import RagService


router = APIRouter()


@router.post("/query", response_model=LLMResponse)
def query_rag(request: QueryRequest) -> dict:
    service = RagService()
    response = service.query_rag(request=request)
    
    if "error" in response:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=response["error"]
        )

    return response