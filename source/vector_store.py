
import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from source.embeddings import embeddings


load_dotenv()


DEFAULT_INDEX_NAME = "aerointel-v2"
EMBEDDING_DIMENSION = 384


def create_vector_store(index_name=DEFAULT_INDEX_NAME):
    """Connect to an existing Pinecone index or create it if missing."""

    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY is missing. Check your .env file."
        )

    pc = Pinecone(api_key=api_key)

    existing_indexes = [
        index["name"] for index in pc.list_indexes()
    ]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
    )


def create_retriever(
    vector_store,
    k=3,
    entity_type=None,
):
    """Create a retriever with optional metadata filtering.

    Args:
        vector_store: Connected Pinecone vector store.
        k: Number of documents to retrieve.
        entity_type: Optional metadata type, such as
            'aircraft' or 'airline'.

    Returns:
        A LangChain vector-store retriever.
    """

    if k < 1:
        raise ValueError("k must be at least 1.")

    search_kwargs = {"k": k}

    if entity_type is not None:
        if entity_type not in {"aircraft", "airline"}:
            raise ValueError(
                "entity_type must be 'aircraft', 'airline', "
                "or None."
            )

        search_kwargs["filter"] = {"type": entity_type}

    return vector_store.as_retriever(
        search_kwargs=search_kwargs,
    )
