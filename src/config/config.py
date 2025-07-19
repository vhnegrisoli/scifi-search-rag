from dataclasses import dataclass
from dotenv import load_dotenv
import os


load_dotenv()


EMPTY = ''


@dataclass
class AppConfigData:
    openai_key: str = None
    openai_model: str = None
    openai_embedding_model: str = None
    pinecone_key: str = None
    pinecone_index: str = None


class AppConfig:
    def __init__(self):
        self.openai_key = os.getenv('OPENAI_KEY', EMPTY)
        self.openai_model = os.getenv('OPENAI_MODEL', EMPTY)
        self.openai_embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', EMPTY)
        self.pinecone_key = os.getenv('PINECONE_API_KEY', EMPTY)
        self.pinecone_index = os.getenv('PINECONE_INDEX', EMPTY)
    
    def get_config(self) -> AppConfigData:
        return AppConfigData(
            openai_key=self.openai_key,
            openai_model=self.openai_model,
            pinecone_key=self.pinecone_key,
            pinecone_index=self.pinecone_index
        )