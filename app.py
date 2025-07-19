from src.vector_store.vector_store import VectorStore
from src.llm.llm_integration import LLMCall
import json
from dataclasses import asdict


if __name__ == "__main__":
    vs = VectorStore()
    query = 'Quais livros tratam sobre viagem no tempo?'
    docs = vs.search_vector_store(
        filter_id='36eafca6-5848-4e58-be2c-fdc8888588ff',
        query=query,
        k=5
    )
    memory = [
        "user: Olá, como vai?",
        "assistant: Tudo certo, e por aí?",
        "user: Qual livro de Isaac Asimov trata sobre universos paralelos?",
        "assistant: O livro 'Os Próprios Deuses', de 1972"
    ]
    llm = LLMCall()
    response = llm.call_llm(
        message=query,
        docs=docs,
        history=memory
    )
    print(response.content)
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(asdict(response), f, indent=4, ensure_ascii=False)