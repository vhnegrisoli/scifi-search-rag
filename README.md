# 🤖📚 Sci-Fi History RAG API

Um microserviço em Python que utiliza **LangChain**, **Pinecone** e **FastAPI**, e com os modelos **OpenAI**, **Llama 3** e **HuggingFace** para responder perguntas sobre a história da ficção científica com base em um livro indexado.

> 🔍 Powered by Retrieval-Augmented Generation (RAG) para oferecer respostas contextualizadas e precisas.

## ✨ Funcionalidades

- 🔎 **RAG (Retrieval-Augmented Generation)** com LangChain
- 📖 Baseado em um livro sobre a história da ficção científica
- 🧠 Vetorização com Pinecone
- ⚡ API REST com FastAPI
- 🤖 Integração com modelos de LLM **OpenAI** -> **gpt-4o-mini**
- 🤖 Integração com modelos de LLM **Ollama** -> **llama3.1:8b**
- 🤖 Integração com modelos de embeddings **OpenAI** -> **text-embedding-ada-002**
- 🤖 Integração com modelos de embeddings **HuggingFace** -> **sentence-transformers/all-MiniLM-L6-v2**
- 🧪 Testes prontos para validar endpoints e lógica

## 📚 Sobre o projeto

Este projeto responde perguntas em linguagem natural sobre obras, autores e movimentos históricos da ficção científica, extraindo dados diretamente de um livro processado e indexado. Ideal para leitores apaixonados por sci-fi, como eu.

## 🧰 Tecnologias Utilizadas

| Tecnologia  | Descrição                                    |
| ----------- | -------------------------------------------- |
| 🐍 Python 3  | Linguagem principal                          |
| ⚡ FastAPI   | Criação da API REST                          |
| 🔗 LangChain | Framework para RAG e pipelines de LLMs       |
| 🌲 Pinecone  | Banco vetorial para buscas semânticas        |
| 🧠 OpenAI    | Geração de linguagem natural com modelos LLM (gpt-4o-mini) |
| 🧠 Ollama    | Geração de linguagem natural com modelos LLM (llama3.1:8b) |
| 🧠 OpenAI    | Geração de modelos de embeddings (text-embedding-ada-002) |
| 🧠 HuggingFace    | Geração de modelos de embeddings (sentence-transformers/all-MiniLM-L6-v2) |

---

## 🚀 Como rodar localmente

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/sci-fi-history-rag-api.git
cd sci-fi-history-rag-api
````

### 2. Instale as dependências

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Instale o Ollama com o modelo Llama3.1:8b

Basta executar o comando:

`docker-compose up --build -d`

E por fim rodar o comando:

`docker exec -it ollama ollama pull llama3.1:8b`

Ou então, instale manualmente o Ollama e baixe o modelo llama3.1 com o comando `ollama pull llama3.1:8b`.

### 4. Configure as variáveis de ambiente

Crie um arquivo `.env` com:

```env
OPENAI_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
PINECONE_API_KEY=pcsk_...
PINECONE_OPENAI_INDEX=scifi-rag
PINECONE_HUGGINGFACE_INDEX=scifi-rag-huggingface
PROJECT_ROOT=.
LLAMA_MODEL=llama3.1:8b
HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 5. Rode o servidor

```bash
uvicorn app:app --reload
```

---

## 📥 Exemplo de uso

### `POST /api/query`

```json
{
  "query": "Liste os livros de Isaac Asimov que foram citados",
  "filter_id": "36eafca6-5848-4e58-be2c-fdc8888588ff",
  "top_k": 15,
  "history": [
    "user: Isaac Asimov é citado no documento?",
    "assistant: Sim, alguns livros de Isaac Asimov são citados!"
  ]
}
```

### Resposta

```json
{
  "content": "Os livros de Isaac Asimov que foram citados nos documentos são:\n\n1. **The Bicentennial Man** (1976) - Incluído em \"The Complete Robot\".\n2. **The Caves of Steel** (1954).",
  "total_docs": 15,
  "usage": {
    "input_tokens": 10490,
    "output_tokens": 50,
    "total_tokens": 10540
  },
  "docs": [
    "caótico. E embora não possamos supor que Asimov tenha previsto a Teoria do\nCaos, o fato é que na série Duna,...",
    "determinante original da FC parece tender para a vertente materialista ou\nprotestante; mas de fato aquela vertente católica mística/fantástica está muito...",
    "e Peter Gallison, com Amy Slaton (orgs.). Picturing Science and Producing Art. Nova..."
  ]
}
```


## 📌 Estrutura do projeto

```
openai-scifi-rag/
├── files/
│   └── scifi.txt                  # Texto de entrada usado no projeto
├── app.py                         # Entrypoint da aplicação FastAPI
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config.py              # Configurações gerais do projeto
│   │   └── prompt.py              # Prompts customizados para o LLM
│   ├── llm/
│   │   ├── __init__.py
│   │   └── llm_integration.py     # Integração com LLM (OpenAI ou Llama)
│   ├── models/
│   │   ├── endpoint.py            # Schemas para requests/responses da API
│   │   └── llm_models.py          # Modelos auxiliares relacionados ao LLM
│   │   └── providers.py           # Enums que definem quais providers serão usados na LLM e nos embeddings
│   ├── routes/
│   │   └── rag_route.py           # Definição de endpoints da API
│   ├── services/
│   │   ├── __init__.py
│   │   └── rag_service.py         # Regras de negócio do RAG
│   ├── vector_store/
│   │   ├── __init__.py
│   │   └── vector_store.py        # Integração com o repositório vetorial (ex: Pinecone)
│   └── __init__.py
├── requirements.txt               # Dependências do projeto
└── README.md                      # Instruções e descrição do projeto
```

---

## 📖 Licença

[MIT](LICENSE)

---

## ✍️ Autor

**Victor Hugo Negrisoli**
🔗 [LinkedIn](https://www.linkedin.com/in/victorhugonegrisoli/) | 🐙 [GitHub](https://github.com/vhnegrisoli/)
