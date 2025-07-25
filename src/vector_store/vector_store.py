from src.config.config import AppConfig
from src.models.providers import EmbeddingProvider
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
from typing import List
from uuid import uuid4
import os


DEFAULT_CHUNK_SIZE = 3000
DEFAULT_CHUNK_OVERLAP = 200
FILTER_ID_METADATA = 'filter_id'
BATCH_UPSERT = 50

load_dotenv()

ROOT = os.getenv("PROJECT_ROOT", '.')
FILE_PATH = os.path.join(ROOT, "files", "scifi.txt")


class VectorStore:
    def __init__(self, provider: EmbeddingProvider):
        self._provider = provider
        self._config = AppConfig().get_config()
        self._index = ''
        self._define_embedding_model()
        self._vector_store = PineconeVectorStore(
            pinecone_api_key=self._config.pinecone_key,
            index_name=self._index,
            embedding=self._embedding
        )

    def _define_embedding_model(self) -> Embeddings:
        if EmbeddingProvider.OPENAI_EMBEDDINGS == self._provider:
            self._embedding = OpenAIEmbeddings(
                api_key=self._config.openai_key,
                model=self._config.openai_embedding_model
            )
            self._index = self._config.pinecone_openai_index
        if EmbeddingProvider.HUGGINGFACE_EMBEDDINGS == self._provider:
            self._embedding = HuggingFaceEmbeddings(
                model_name=self._config.huggingface_embedding_model,
            )
            self._index = self._config.pinecone_huggingface_index

    def _chunk_docs(self, docs, chunk_size):
        for i in range(0, len(docs), chunk_size):
            yield docs[i:i + chunk_size]

    def create_embeddings(self, filter_id: str) -> None:
        print(f"Creating embeddings for file: {filter_id}")
        docs = self._split_document(filter_id=filter_id)
        self._insert_into_vector_store(docs=docs)

    def _split_document(self, filter_id: str) -> List[Document]:
        loader = TextLoader(FILE_PATH, encoding='utf-8')
        documents = loader.load()

        text_splitter = CharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )

        docs = text_splitter.split_documents(documents)

        for doc in docs:
            doc.metadata[FILTER_ID_METADATA] = filter_id

        return docs

    def _insert_into_vector_store(self, docs: List[Document]) -> None:
        vectorstore = self._vector_store.from_existing_index(
            index_name=self._index,
            embedding=self._embedding
        )
        for chunk in self._chunk_docs(docs, BATCH_UPSERT):
            vectorstore.add_documents(
                chunk,
                namespace=self._index
            )

    def search_vector_store(self, query: str, filter_id: str, k: int = 15) -> List[Document]:
        vector_store = self._vector_store.from_existing_index(
            index_name=self._index,
            embedding=self._embedding
        )
        metadata_filter = {FILTER_ID_METADATA: filter_id}
        return vector_store.similarity_search(
            query=query,
            k=k,
            namespace=self._index,
            filter=metadata_filter
        )
