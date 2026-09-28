import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


load_dotenv()


def create_index(index_name, dimension=384):
    api_key = os.getenv("PINECONE_API_KEY")

    pc = Pinecone(api_key=api_key)

    existing_indexes = [index["name"] for index in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    return pc.Index(index_name)


def prepare_vectors(chunks, embeddings):
    vectors = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding.tolist(),
            "metadata": {
                "text": chunk["text"],
                "source": chunk["source"],
                "type": chunk["type"],
                "entity_id": chunk["entity_id"],
                "display_name": chunk["display_name"]
            }
        })

    return vectors


def upsert_vectors(index, vectors):
    index.upsert(vectors=vectors)


def search_vectors(
    index,
    query_embedding,
    top_k=3,
    score_threshold=0.5,
    document_type=None,
    entity=None
):
    filters = {}

    if document_type:
        filters["type"] = document_type

    if entity:
        filters["entity_id"] = entity

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
        filter=filters if filters else None
    )

    filtered_matches = [
        match
        for match in results["matches"]
        if match["score"] >= score_threshold
    ]

    results["matches"] = filtered_matches

    return results

def clear_index(index):
    index.delete(delete_all=True)