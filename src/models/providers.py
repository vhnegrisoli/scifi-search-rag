from enum import Enum


class LLMProvider(Enum):
    OPENAI = 'OPENAI'
    LLAMA3 = 'LLAMA3'


class EmbeddingProvider(Enum):
    OPENAI_EMBEDDINGS = 'OPENAI_EMBEDDINGS'
    HUGGINGFACE_EMBEDDINGS = 'HUGGINGFACE_EMBEDDINGS'
