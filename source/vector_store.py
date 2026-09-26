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
                "source": chunk["source"]
            }
        })

    return vectors

def upsert_vectors(index,vectors):
    index.upsert(vectors=vectors)

def search_vectors(index,query_embeddings,top_k=3):
    results=index.query(
        vector=query_embeddings.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    return results