from mcp.server.mcpserver import MCPServer

from source.vector_store import create_vector_store

#1. Create the MCP Server
mcp=MCPServer("Aerointel")

#Connecting to existing pinecone index
vector_store=create_vector_store("aerointel-v2")

#2. Register a tool
@mcp.tool()
def search_aircraft(aircraft: str) -> str:
    """Search the AeroIntel knowledge base for aircraft information."""

    try:
        documents = vector_store.similarity_search(
            f"Information about aircraft {aircraft}",
            k=3,
            filter={"type": "aircraft"}
        )

        if not documents:
            return f"No aircraft information found for {aircraft}."

        return "\n\n".join(
            document.page_content
            for document in documents
        )

    except Exception as error:
        print(f"Aircraft search failed: {error!r}", flush=True)
        raise


@mcp.tool()
def search_airline(airline: str) -> str:
    """Search the AeroIntel knowledge base for information about an airline."""

    try:
        documents = vector_store.similarity_search(
            f"Information about airline {airline}",
            k=3,
            filter={"type": "airline"}
        )

        if not documents:
            return f"No airline information found for {airline}."

        return "\n\n".join(
            document.page_content
            for document in documents
        )

    except Exception as error:
        print(f"Airline search failed: {error!r}", flush=True)
        raise


@mcp.tool()
def search_aviation_knowledge(query: str) -> str:
    """Search the AeroIntel knowledge base for general aviation information."""

    try:
        documents = vector_store.similarity_search(
            query,
            k=3
        )

        if not documents:
            return "No relevant aviation information found."

        return "\n\n".join(
            document.page_content
            for document in documents
        )

    except Exception as error:
        print(f"General aviation search failed: {error!r}", flush=True)
        raise



if __name__ == "__main__":
    mcp.run()