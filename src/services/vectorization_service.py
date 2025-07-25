from typing import List
from src.models.providers import EmbeddingProvider
from src.models.endpoint import VectorSearchQueryRequest, VectorizationRequest
from src.vector_store.vector_store import VectorStore
from langchain_core.documents import Document


class VectorizationService:
    
    def __init__(self, embedding_provider: EmbeddingProvider):
        self._vector_store = VectorStore(provider=embedding_provider)

    def vectorize_document(self, request: VectorizationRequest) -> dict:
        try:
            self._vector_store.create_embeddings(filter_id=request.filter_id)
            return {
                "status": "SUCCESS",
                "message": f"File vectorized successfully for filter_id {request.filter_id} and {request.embedding_provider} provider",
                "filter_id": request.filter_id             
            }
        except Exception as ex:
            return {
                "error": str(ex)
            }
        
    def search_vector_store_endpoint(self, request: VectorSearchQueryRequest) -> dict:
        docs = self.search_vector_store(
            filter_id=request.filter_id,
            top_k=request.top_k,
            query=request.query
        )
        return {
            "docs": docs
        }
        
    def search_vector_store(self,
                            query: str,
                            filter_id: str,
                            top_k: int) -> List[str]:
        docs = self._vector_store.search_vector_store(
            filter_id=filter_id,
            k=top_k,
            query=query
        )
        return [doc.page_content for doc in docs]