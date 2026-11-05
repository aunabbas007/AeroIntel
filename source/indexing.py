from collections import defaultdict
from hashlib import sha256


def stable_document_ids(documents) -> list[str]:
    """Return repeatable IDs so re-indexing replaces existing chunks."""
    positions: dict[str, int] = defaultdict(int)
    ids = []

    for document in documents:
        metadata = document.metadata or {}
        source = str(metadata.get("source", "unknown"))
        source = source.replace("\\", "/")
        position = positions[source]
        positions[source] += 1
        ids.append(
            sha256(f"{source}\0{position}".encode("utf-8")).hexdigest()
        )

    return ids
