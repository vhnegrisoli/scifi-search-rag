from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from src.config.config import AppConfig
from typing import List, Optional
from langchain_core.documents import Document
from dataclasses import dataclass, asdict, field
import json


@dataclass
class LLMUsageResponse:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    content: str
    prompt: str
    total_docs: int = 0
    usage: Optional[LLMUsageResponse] = None
    docs: List[str] = field(default_factory=list)

    def to_json(self):
        return json.dumps(asdict(self))

class LLMCall:
    def __init__(self):
        self.config = AppConfig().get_config()
        self.llm = ChatOpenAI(
            model=self.config.openai_model,
            api_key=self.config.openai_key
        )
        self.prompt = PromptTemplate.from_template(
            """
            You are an AI assistant specialized in Sci-Fi contexts.

            Context from documents:
            {context}

            Memory of previous interactions:
            {memory}

            User's current message:
            {message}

            Answer clearly and concisely.

            You must not invent anything, only answer based on the context, message and memory.
            """
        )

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
            prompt=formatted_prompt,
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