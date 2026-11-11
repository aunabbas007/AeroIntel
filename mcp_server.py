
import hashlib
import logging
import os
import re
import sys

import redis
from mcp.server.mcpserver import MCPServer
from source.vector_store import create_vector_store


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aerointel.mcp")

mcp = MCPServer("Aerointel")
vector_store = create_vector_store()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL_SECONDS", "3600"))
CACHE_VERSION = os.getenv("AEROINTEL_CACHE_VERSION", "v1")

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=2,
    socket_timeout=2,
    health_check_interval=30,
)


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
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def make_cache_key(search_type: str, query: str, entity_type: str = ""):
    normalized = normalize_name(query)
    raw = f"{CACHE_VERSION}:{search_type}:{entity_type}:{normalized}"
    digest = hashlib.sha256(raw.encode()).hexdigest()

    return f"aerointel:retrieval:{digest}"


def cached_retrieval(cache_key: str, retrieval_function) -> str:
    """Read from Redis, retrieve from Pinecone on a miss."""

    try:
        cached = redis_client.get(cache_key)

        if cached is not None:
            logger.info("retrieval_cache_hit")
            return cached

    except redis.RedisError as error:
        logger.warning(
            "redis_read_failed error_type=%s",
            type(error).__name__,
        )

    logger.info("retrieval_cache_miss")

    # Pinecone retrieval
    result = retrieval_function()

    try:
        redis_client.setex(cache_key, CACHE_TTL, result)

    except redis.RedisError as error:
        logger.warning(
            "redis_write_failed error_type=%s",
            type(error).__name__,
        )

    return result


def get_entity_id(entity_name: str, entity_type: str) -> str | None:
    name = normalize_name(entity_name)

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

    airline_ids = {
        "emirates",
        "air_india",
        "british_airways",
        "etihad_airways",
        "lufthansa",
        "qatar_airways",
        "singapore_airlines",
        "turkish_airlines",
    }

    if entity_type == "aircraft" and entity_id in airline_ids:
        return None

    if entity_type == "airline" and entity_id not in airline_ids:
        return None

    return entity_id


def search_entity_uncached(name: str, entity_type: str) -> str:
    entity_id = get_entity_id(name, entity_type)

    if entity_id:
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

    documents = vector_store.similarity_search(
        f"Information about {entity_type} {name}",
        k=3,
        filter={"type": entity_type},
    )

    if not documents:
        return f"No {entity_type} information found for {name}."

    return format_documents(documents)


def search_entity(name: str, entity_type: str) -> str:
    cache_key = make_cache_key("entity", name, entity_type)

    return cached_retrieval(
        cache_key,
        lambda: search_entity_uncached(name, entity_type),
    )


@mcp.tool()
def search_aircraft(aircraft: str) -> str:
    """Search for a specific aircraft in AeroIntel."""

    try:
        return search_entity(aircraft, "aircraft")

    except Exception as error:
        print(
            f"Aircraft search failed: {type(error).__name__}",
            file=sys.stderr,
            flush=True,
        )
        raise


@mcp.tool()
def search_airline(airline: str) -> str:
    """Search for a specific airline in AeroIntel."""

    try:
        return search_entity(airline, "airline")

    except Exception as error:
        print(
            f"Airline search failed: {type(error).__name__}",
            file=sys.stderr,
            flush=True,
        )
        raise


@mcp.tool()
def search_aviation_knowledge(query: str) -> str:
    """Search general aviation information in AeroIntel."""

    try:
        cache_key = make_cache_key("knowledge", query)

        def retrieve():
            documents = vector_store.similarity_search(query, k=3)

            if not documents:
                return "No relevant aviation information found."

            return format_documents(documents)

        return cached_retrieval(cache_key, retrieve)

    except Exception as error:
        print(
            f"General aviation search failed: {type(error).__name__}",
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    print("MCP_MAIN_ENTERED", file=sys.stderr, flush=True)
    mcp.run()
    print("MCP_RUN_RETURNED", file=sys.stderr, flush=True)
