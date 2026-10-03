from langchain_pinecone import PineconeVectorStore
from source.embeddings import embeddings

def create_vector_store(index_name="aerointel-v1"):
    return PineconeVectorStore(
        index_name=index_name,
        embeddings=embeddings
    )

def create_retreiver(vector_store,k=3):
    return vector_store.as_retreiver(
        search_kwargs={"k":k}
    )