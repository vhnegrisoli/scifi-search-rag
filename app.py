from fastapi import FastAPI
from src.routes.rag_route import router as rag_router

app = FastAPI(
    title="Sci-Fi RAG API",
    description="RAG com LangChain, Pinecone e OpenAI para responder sobre ficção científica",
    version="1.0.0"
)

app.include_router(rag_router, prefix="/api", tags=["RAG"])