
import re
import sys

from mcp.server.mcpserver import MCPServer
from source.vector_store import create_vector_store


mcp = MCPServer("Aerointel")
vector_store = create_vector_store()


def format_documents(documents) -> str:
    formatted = []
    for document in documents:
        metadata = document.metadata or {}
        source = metadata.get("source") or metadata.get("display_name")
        if source:
            source = str(source).replace("\\", "/").rsplit("/", 1)[-1]
            formatted.append(
                f"[Source: {source}]\n{document.page_content}"
            )
        else:
            formatted.append(document.page_content)
    return "\n\n".join(formatted)


def normalize_name(value: str) -> str:
    """Normalize names for comparison."""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def get_entity_id(entity_name: str, entity_type: str) -> str | None:
    """Resolve a known entity name to its existing entity_id."""

    name = normalize_name(entity_name)

    # Known aliases for the entities in the current AeroIntel dataset.
    aliases = {
        "a350 900": "a350_900",
        "airbus a350 900": "a350_900",
        "a350 1000": "a350_1000",
        "airbus a350 1000": "a350_1000",
        "a320neo": "a320neo",
        "airbus a320neo": "a320neo",
        "787 9": "787_9",
        "boeing 787 9": "787_9",
        "787 10": "787_10",
        "boeing 787 10": "787_10",
        "777 200lr": "777_200lr",
        "boeing 777 200lr": "777_200lr",
        "777 300er": "777_300er",
        "boeing 777 300er": "777_300er",
        "a350 900": "a350_900",
        "a350 1000": "a350_1000",
        "air india": "air_india",
        "british airways": "british_airways",
        "emirates": "emirates",
        "etihad": "etihad_airways",
        "etihad airways": "etihad_airways",
        "lufthansa": "lufthansa",
        "qatar": "qatar_airways",
        "qatar airways": "qatar_airways",
        "singapore airlines": "singapore_airlines",
        "turkish airlines": "turkish_airlines",
    }

    entity_id = aliases.get(name)

    if entity_id is None:
        return None

    # Ensure the alias belongs to the requested category.
    if entity_type == "aircraft" and entity_id in {
        "emirates", "air_india", "british_airways",
        "etihad_airways", "lufthansa", "qatar_airways",
        "singapore_airlines", "turkish_airlines",
    }:
        return None

    if entity_type == "airline" and entity_id not in {
        "emirates", "air_india", "british_airways",
        "etihad_airways", "lufthansa", "qatar_airways",
        "singapore_airlines", "turkish_airlines",
    }:
        return None

    return entity_id


def search_entity(name: str, entity_type: str) -> str:
    """Try a known exact entity first, then use semantic retrieval."""

    entity_id = get_entity_id(name, entity_type)

    if entity_id:
        # Pinecone metadata filtering retrieves the exact indexed entity.
        documents = vector_store.similarity_search(
            name,
            k=1,
            filter={
                "type": entity_type,
                "entity_id": entity_id,
            },
        )

        if documents:
            return format_documents(documents)

    # Fallback for names not covered by the alias table.
    documents = vector_store.similarity_search(
        f"Information about {entity_type} {name}",
        k=3,
        filter={"type": entity_type},
    )

    if not documents:
        return f"No {entity_type} information found for {name}."

    return format_documents(documents)


@mcp.tool()
def search_aircraft(aircraft: str) -> str:
    """Search for a specific aircraft in the AeroIntel knowledge base."""
    try:
        return search_entity(aircraft, "aircraft")
    except Exception as error:
        print(
            f"Aircraft search failed error_type={type(error).__name__}",
            file=sys.stderr,
            flush=True,
        )
        raise


@mcp.tool()
def search_airline(airline: str) -> str:
    """Search for a specific airline in the AeroIntel knowledge base."""
    try:
        return search_entity(airline, "airline")
    except Exception as error:
        print(
            f"Airline search failed error_type={type(error).__name__}",
            file=sys.stderr,
            flush=True,
        )
        raise


@mcp.tool()
def search_aviation_knowledge(query: str) -> str:
    """Search the knowledge base for general aviation information."""
    try:
        documents = vector_store.similarity_search(query, k=3)

        if not documents:
            return "No relevant aviation information found."

        return format_documents(documents)
    except Exception as error:
        print(
            f"General aviation search failed error_type={type(error).__name__}",
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    print("MCP_MAIN_ENTERED", file=sys.stderr, flush=True)
    mcp.run()
    print("MCP_RUN_RETURNED", file=sys.stderr, flush=True)
