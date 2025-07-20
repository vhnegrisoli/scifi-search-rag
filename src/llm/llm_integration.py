from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from src.config.config import AppConfig
from src.models.llm_models import LLMResponse, LLMUsageResponse
from src.config.prompt import RAG_PROMPT
from typing import List


class LLMCall:
    def __init__(self):
        self.config = AppConfig().get_config()
        self.llm = ChatOpenAI(
            model=self.config.openai_model,
            api_key=self.config.openai_key
        )
        self.prompt = PromptTemplate.from_template(RAG_PROMPT)

    def _format_docs(self, docs: List[Document]) -> str:
        return "Document chunk:\n\n" + "\n\nDocument chunk:\n\n".join(doc.page_content for doc in docs)

    def _format_history(self, history: List[str]) -> str:
        return "\n".join(history)

    def call_llm(self,
                 message: str,
                 history: List[str],
                 docs: List[Document]) -> LLMResponse:
        formatted_prompt = self.prompt.format(
            context=self._format_docs(docs),
            memory=self._format_history(history),
            message=message
        )
        response = self.llm.invoke(formatted_prompt)
        
        return LLMResponse(
            content=response.content,
            usage=self._get_usage(response=response),
            total_docs=len(docs),
            docs=[doc.page_content for doc in docs]
        )

    def _get_usage(self, response: BaseMessage) -> LLMUsageResponse:
        usage = response.response_metadata.get("token_usage", None)
        if usage:
            return LLMUsageResponse(
                input_tokens=int(usage.get('prompt_tokens')),
                output_tokens=int(usage.get('completion_tokens')),
                total_tokens=int(usage.get('total_tokens'))
            )
        else:
            return LLMUsageResponse()