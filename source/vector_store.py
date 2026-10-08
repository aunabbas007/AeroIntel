import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from source.embeddings import embeddings

load_dotenv()


def create_vector_store(index_name="aerointel-v2"):

    api_key = os.getenv("PINECONE_API_KEY")

    pc = Pinecone(api_key=api_key)

    existing_indexes = [
        index["name"]
        for index in pc.list_indexes()
    ]

    if index_name not in existing_indexes:

        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    return vector_store


def create_retriever(vector_store, k=3):

    return vector_store.as_retriever(
        search_kwargs={
            "k": k
        }
    )