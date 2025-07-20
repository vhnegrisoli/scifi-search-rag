from dataclasses import dataclass
from dotenv import load_dotenv
import os


load_dotenv()


EMPTY = ''
DEFAULT_TOP_K = 5


@dataclass
class AppConfigData:
    openai_key: str = None
    openai_model: str = None
    openai_embedding_model: str = None
    pinecone_key: str = None
    pinecone_index: str = None
    filter_id: str = None
    top_k: int = DEFAULT_TOP_K


class AppConfig:
    def __init__(self):
        self.openai_key = os.getenv('OPENAI_KEY', EMPTY)
        self.openai_model = os.getenv('OPENAI_MODEL', EMPTY)
        self.openai_embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', EMPTY)
        self.pinecone_key = os.getenv('PINECONE_API_KEY', EMPTY)
        self.pinecone_index = os.getenv('PINECONE_INDEX', EMPTY)
        self.filter_id = os.getenv('DEFAULT_VECTOR_FILTER_ID', EMPTY)

    def get_config(self) -> AppConfigData:
        return AppConfigData(
            openai_key=self.openai_key,
            openai_model=self.openai_model,
            pinecone_key=self.pinecone_key,
            pinecone_index=self.pinecone_index,
            filter_id=self.filter_id
        )