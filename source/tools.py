from langchain.tools import tool

from source.vector_store import create_vector_store


vector_store = create_vector_store("aerointel-v2")


@tool
def search_aviation_knowledge(query: str):
    """
    Search the AeroIntel aviation knowledge base for
    general information about airlines and aircraft.
    """

    documents = vector_store.similarity_search(
        query,
        k=3
    )

    if not documents:
        return "No relevant information was found."

    return "\n\n".join(
        document.page_content
        for document in documents
    )


@tool
def search_airline(airline: str):
    """
    Search the AeroIntel knowledge base for information
    about a specific airline.
    """

    query = f"Information about airline {airline}"

    documents = vector_store.similarity_search(
        query,
        k=5,
        filter={
            "type": "airline"
        }
    )

    if not documents:
        return f"No information found for airline: {airline}"

    return "\n\n".join(
        document.page_content
        for document in documents
    )


@tool
def search_aircraft(aircraft: str):
    """
    Search the AeroIntel knowledge base for information
    about a specific aircraft.
    """

    query = f"Information about aircraft {aircraft}"

    documents = vector_store.similarity_search(
        query,
        k=5,
        filter={
            "type": "aircraft"
        }
    )

    if not documents:
        return f"No information found for aircraft: {aircraft}"

    return "\n\n".join(
        document.page_content
        for document in documents
    )