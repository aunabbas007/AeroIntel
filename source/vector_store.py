
import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from source.embeddings import embeddings


load_dotenv()


DEFAULT_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "aerointel-v2")
EMBEDDING_DIMENSION = 384


def _pinecone_client() -> Pinecone:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is missing from the environment.")
    return Pinecone(api_key=api_key)


def ensure_vector_index(index_name: str = DEFAULT_INDEX_NAME) -> None:
    """Create the configured index for the explicit ingestion command."""
    pc = _pinecone_client()
    if index_name in pc.list_indexes().names():
        return

    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        ),
    )

    timeout = int(os.getenv("PINECONE_INDEX_READY_TIMEOUT", "180"))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        description = pc.describe_index(index_name)
        if getattr(getattr(description, "status", None), "ready", False):
            return
        time.sleep(2)

    raise TimeoutError(
        f"Pinecone index '{index_name}' did not become ready in time."
    )


def create_vector_store(index_name: str | None = None):
    """Connect to an existing, dimension-compatible Pinecone index."""
    index_name = index_name or DEFAULT_INDEX_NAME
    pc = _pinecone_client()

    if index_name not in pc.list_indexes().names():
        raise RuntimeError(
            f"Pinecone index '{index_name}' does not exist. "
            "Run the explicit indexing command after configuring the index."
        )

    description = pc.describe_index(index_name)
    if description.dimension != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Pinecone index '{index_name}' has dimension "
            f"{description.dimension}; expected {EMBEDDING_DIMENSION}."
        )

    if not getattr(getattr(description, "status", None), "ready", True):
        raise RuntimeError(f"Pinecone index '{index_name}' is not ready.")

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
