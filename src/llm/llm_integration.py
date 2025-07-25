from langchain_core.messages import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from src.config.config import AppConfig
from src.models.llm_models import LLMResponse, LLMUsageResponse
from src.config.prompt import RAG_PROMPT
from src.models.providers import LLMProvider
from typing import List


class LLMCall:
    def __init__(self, provider: LLMProvider):
        self._provider = provider
        self._config = AppConfig().get_config()
        self._prompt = PromptTemplate.from_template(RAG_PROMPT)
        self._llm = None

    def _get_llm(self) -> BaseChatModel:
        if LLMProvider.OPENAI == self._provider:
            self._llm = ChatOpenAI(
                model=self._config.openai_model,
                api_key=self._config.openai_key
            )
        if LLMProvider.LLAMA3 == self._provider:
            self._llm = ChatOllama(model="llama3.1:8b")

    def _format_docs(self, docs: List[Document]) -> str:
        return "Document chunk:\n\n" + "\n\nDocument chunk:\n\n".join(doc for doc in docs)

    def _format_history(self, history: List[str]) -> str:
        return "\n".join(history)

    def call_llm(self,
                 message: str,
                 history: List[str],
                 docs: List[str],
                 provider: LLMProvider) -> LLMResponse:

        self._get_llm()

        formatted_prompt = self._prompt.format(
            context=self._format_docs(docs),
            memory=self._format_history(history),
            message=message
        )
        response = self._llm.invoke(formatted_prompt)

        return LLMResponse(
            content=response.content,
            usage=self._get_usage(response=response, provider=provider),
            total_docs=len(docs),
            docs=docs
        )

    def _get_usage(self, response: BaseMessage, provider: LLMProvider) -> LLMUsageResponse:
        input = 0
        output = 0
        if LLMProvider.OPENAI == provider:
            usage = response.response_metadata.get("token_usage", None)
            if usage:
                input = self._get_from_usage(usage=usage, key='prompt_tokens')
                output = self._get_from_usage(usage=usage, key='completion_tokens')

        if LLMProvider.LLAMA3 == provider:
            usage = response.usage_metadata
            if usage:
                input = self._get_from_usage(usage=usage, key='input_tokens')
                output = self._get_from_usage(usage=usage, key='output_tokens')

        return LLMUsageResponse(
            input_tokens=input,
            output_tokens=output,
            total_tokens=input+output
        )

    def _get_from_usage(self, usage: dict, key: str) -> int:
        return usage.get(key, 0)
