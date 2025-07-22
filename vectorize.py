from src.vector_store.vector_store import VectorStore
from src.models.providers import EmbeddingProvider

if __name__ == "__main__":
    provider = EmbeddingProvider.HUGGINFACE_EMBEDDINGS
    vector_store = VectorStore(provider=provider)
    filter_id = "975936a4-6fce-4d9f-8b50-a4dc0fceea67"
    #vector_store.create_embeddings(filter_id=filter_id)
    #print(f"Embeddings created for filter ID: {filter_id}")


    docs = vector_store.search_vector_store(
        query="Quais livros de Arthur C. Clarke são citados no documento?",
        k=5,
        filter_id='975936a4-6fce-4d9f-8b50-a4dc0fceea67'
    )

    docs_str = ''
    for doc in docs:
        docs_str = docs_str + doc.page_content + "\n==========================================\n\n"

    print(docs_str)