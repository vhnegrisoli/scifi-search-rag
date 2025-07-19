# 🤖📚 Sci-Fi History RAG API

Um microserviço em Python que utiliza **LangChain**, **Pinecone**, **FastAPI** e **OpenAI** para responder perguntas sobre a história da ficção científica com base em um livro indexado.

> 🔍 Powered by Retrieval-Augmented Generation (RAG) para oferecer respostas contextualizadas e precisas.

## ✨ Funcionalidades

- 🔎 **RAG (Retrieval-Augmented Generation)** com LangChain
- 📖 Baseado em um livro sobre a história da ficção científica
- 🧠 Vetorização com Pinecone
- ⚡ API REST com FastAPI
- 🤖 Integração com modelos da OpenAI (GPT)
- 🧪 Testes prontos para validar endpoints e lógica

## 📚 Sobre o projeto

Este projeto responde perguntas em linguagem natural sobre obras, autores e movimentos históricos da ficção científica, extraindo dados diretamente de um livro processado e indexado. Ideal para:

- Estudantes e pesquisadores
- Leitores apaixonados por sci-fi
- Aplicações educacionais

## 🧰 Tecnologias Utilizadas

| Tecnologia  | Descrição                                      |
|-------------|------------------------------------------------|
| 🐍 Python 3  | Linguagem principal                            |
| ⚡ FastAPI   | Criação da API REST                            |
| 🔗 LangChain | Framework para RAG e pipelines de LLMs         |
| 🌲 Pinecone  | Banco vetorial para buscas semânticas          |
| 🧠 OpenAI    | Geração de linguagem natural com modelos LLM   |

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

### 3. Configure as variáveis de ambiente

Crie um arquivo `.env` com:

```env
OPENAI_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX=scifi-rag
PROJECT_ROOT=.
```

### 4. Rode o servidor

```bash
uvicorn app.main:app --reload
```

---

## 📥 Exemplo de uso

### `POST /chat`

```json
{
  "question": "Quem foi o precursor da ficção científica moderna?"
}
```

### Resposta

```json
{
  "answer": "H.G. Wells é frequentemente citado como um dos precursores...",
  "consumption": {
    "input": 1550,
    "output": 445,
    "total": 1995
  },
  "docs": [
    "Doc chunk 1",
    "Doc chunk 2"
  ]
}
```


## 📌 Estrutura do projeto

```
sci-fi-history-rag-api/
├── app/
│   ├── main.py            # Entrypoint da API
│   ├── rag_chain.py       # Lógica do LangChain
│   ├── vector_store.py    # Integração com Pinecone
│   └── utils.py
├── data/
│   └── sci-fi-book.pdf    # Livro usado como fonte
├── tests/
│   └── test_api.py
├── requirements.txt
└── README.md
```

---

## 📖 Licença

[MIT](LICENSE)

---

## ✍️ Autor

**Victor Hugo Negrisoli**
🔗 [LinkedIn](https://www.linkedin.com/in/victorhugonegrisoli/) | 🐙 [GitHub](https://github.com/vhnegrisoli/)
