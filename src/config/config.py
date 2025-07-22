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
    pinecone_openai_index: str = None
    pinecone_huggingface_index: str = None
    llama_model: str = None
    huggingface_embedding_model: str = None


class AppConfig:
    def __init__(self):
        self._openai_key = os.getenv('OPENAI_KEY', EMPTY)
        self._openai_model = os.getenv('OPENAI_MODEL', EMPTY)
        self._openai_embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', EMPTY)
        self._pinecone_key = os.getenv('PINECONE_API_KEY', EMPTY)
        self.pinecone_openai_index = os.getenv('PINECONE_OPENAI_INDEX', EMPTY)
        self.pinecone_huggingface_index = os.getenv('PINECONE_HUGGINGFACE_INDEX', EMPTY)
        self._llama_model = os.getenv('LLAMA_MODEL', EMPTY)
        self._huggingface_embedding_model = os.getenv('HUGGINGFACE_EMBEDDING_MODEL', EMPTY)

    def get_config(self) -> AppConfigData:
        return AppConfigData(
            openai_key=self._openai_key,
            openai_model=self._openai_model,
            openai_embedding_model=self._openai_embedding_model,
            pinecone_key=self._pinecone_key,
            pinecone_openai_index=self.pinecone_openai_index,
            pinecone_huggingface_index=self.pinecone_huggingface_index,
            llama_model=self._llama_model,
            huggingface_embedding_model=self._huggingface_embedding_model
        )
