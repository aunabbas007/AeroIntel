
import re

from langchain.tools import tool

from source.vector_store import create_vector_store


vector_store = create_vector_store("aerointel-v2")


def normalize_entity_name(value: str) -> str:
    """Normalize an entity name for matching."""
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def find_exact_entity(entity_name: str, entity_type: str):
    """Find a document using existing Pinecone metadata."""

    normalized_name = normalize_entity_name(entity_name)

    # Query the known entities using their indexed metadata.
    # Fetch a bounded set of candidates within the requested category.
    candidates = vector_store.similarity_search(
        entity_name,
        k=20,
        filter={"type": entity_type},
    )

    for document in candidates:
        entity_id = normalize_entity_name(
            document.metadata.get("entity_id", "")
        )
        display_name = normalize_entity_name(
            document.metadata.get("display_name", "")
        )

        if normalized_name in {entity_id, display_name}:
            return document

    return None


def search_specific_entity(
    entity_name: str,
    entity_type: str,
) -> str:
    """Prefer an exact entity match, then use semantic fallback."""

    exact_document = find_exact_entity(entity_name, entity_type)

    if exact_document is not None:
        return exact_document.page_content

    query = f"Information about {entity_type} {entity_name}"

    documents = vector_store.similarity_search(
        query,
        k=3,
        filter={"type": entity_type},
    )

    if not documents:
        return f"No information found for {entity_type}: {entity_name}"

    return "\n\n".join(
        document.page_content for document in documents
    )


@tool
def search_aviation_knowledge(query: str) -> str:
    """Search the AeroIntel knowledge base for general aviation information."""

    documents = vector_store.similarity_search(query, k=3)

    if not documents:
        return "No relevant information was found."

    return "\n\n".join(
        document.page_content for document in documents
    )


@tool
def search_airline(airline: str) -> str:
    """Search AeroIntel for information about a specific airline."""

    return search_specific_entity(airline, "airline")


@tool
def search_aircraft(aircraft: str) -> str:
    """Search AeroIntel for information about a specific aircraft."""

    return search_specific_entity(aircraft, "aircraft")
