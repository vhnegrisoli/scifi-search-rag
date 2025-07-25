from fastapi import APIRouter, status, HTTPException
from src.models.endpoint import VectorizationRequest
from src.models.endpoint import VectorSearchQueryRequest
from src.models.llm_models import VectorizationResponse
from src.models.llm_models import VectorSearchResponse
from src.services.vectorization_service import VectorizationService


router = APIRouter()


@router.post("/vectorize", response_model=VectorizationResponse)
def vectorize(request: VectorizationRequest) -> dict:

    service = VectorizationService(embedding_provider=request.embedding_provider)
    response = service.vectorize_document(request=request)

    if "error" in response:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=response["error"]
        )

    return response

@router.get('/vector/search', response_model=VectorSearchResponse)
def vector_search(request: VectorSearchQueryRequest) -> dict:
    service = VectorizationService(embedding_provider=request.embedding_provider)
    try:
        return service.search_vector_store(request=request)
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ex)
        )