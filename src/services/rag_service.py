from typing import List
from src.models.endpoint import QueryRequest
from src.llm.llm_integration import LLMCall
from src.models.llm_models import LLMResponse
from src.vector_store.vector_store import VectorStore
from langchain_core.documents import Document


class FilterIdNotFoundError(Exception):
    pass


class RagService:
    def __init__(self):
        self._vector_store = None
        self._llm = None

    def _define_llm_vector_store(self, request: QueryRequest) -> LLMCall:
        self._llm = LLMCall(provider=request.llm_provider)
        self._vector_store = VectorStore(provider=request.embedding_provider)

    def query_rag(self, request: QueryRequest) -> dict:
        try:
            self._validate_filter_id(filter_id=request.filter_id)
            self._define_llm_vector_store(request=request)
            docs = self._get_vector_search(request=request)
            chain = self._get_llm_call(request=request, docs=docs)
            return chain.model_dump()
        except Exception as ex:
            return {
                "error": str(ex)
            }

    def _validate_filter_id(self, filter_id: str) -> None:
        if not filter_id:
            raise FilterIdNotFoundError("Filter ID must be provided!")

    def _get_vector_search(self, request: QueryRequest) -> List[Document]:
        return self._vector_store.search_vector_store(
            filter_id=request.filter_id,
            query=request.query,
            k=request.top_k
    )

    def _get_llm_call(self,
                      request: QueryRequest,
                      docs: List[Document]) -> LLMResponse:
        response = self._llm.call_llm(
            message=request.query,
            docs=docs,
            history=request.history,
            provider=request.llm_provider
        )
        return response
