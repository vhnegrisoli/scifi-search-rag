from src.config.config import AppConfig

from langchain_community.embeddings import OpenAIEmbeddings
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
    def __init__(self):
        self.config = AppConfig().get_config()
        self.embedding = OpenAIEmbeddings(
            api_key=self.config.openai_key,
            model=self.config.openai_embedding_model
        )
        self.vector_store = PineconeVectorStore(
            pinecone_api_key=self.config.pinecone_key,
            index_name=self.config.pinecone_index,
            embedding=self.embedding
        )
    
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
        vectorstore = self.vector_store.from_existing_index(
            index_name=self.config.pinecone_index,
            embedding=self.embedding
        )
        for chunk in self._chunk_docs(docs, BATCH_UPSERT):
            vectorstore.add_documents(
                chunk,
                namespace=self.config.pinecone_index
            )

    def search_vector_store(self, query: str, filter_id: str, k: int = 15) -> List[Document]:
        vector_store = self.vector_store.from_existing_index(
            index_name=self.config.pinecone_index,
            embedding=self.embedding
        )
        metadata_filter = { FILTER_ID_METADATA: filter_id }
        return vector_store.similarity_search(
            query=query,
            k=k,
            namespace=self.config.pinecone_index,
            filter=metadata_filter
        )
        
    
    @staticmethod
    def start():
        query = 'O que o documento diz sobre Asimov e Os Próprios Deuses?'

        vs = VectorStore()

        #filter_id = str(uuid4())
        filter_id = '36eafca6-5848-4e58-be2c-fdc8888588ff'

        #vs.create_embeddings(filter_id=filter_id)
        res = vs.search_vector_store(query=query, filter_id=filter_id)
        print(len(res))
        print(res[0])