from source.loader import load_documents
from source.chunking import chunk_text
from source.embeddings import create_embeddings
from source.vector_store import create_index, prepare_vectors, upsert_vectors,clear_index


documents = load_documents()

print(f"Documents loaded: {len(documents)}")


all_chunks = []

for document in documents:
    chunks = chunk_text(document["text"])

    for chunk in chunks:
        all_chunks.append({
            "source": document["source"],
            "text": chunk,
            "type": document["type"],
            "entity_id": document["entity_id"],
            "display_name": document["display_name"]
        })


print(f"Total chunks created: {len(all_chunks)}")


embeddings = create_embeddings(all_chunks)

print(f"Embeddings shape: {embeddings.shape}")

index = create_index("aerointel-v1")

clear_index(index)

vectors = prepare_vectors(all_chunks, embeddings)

upsert_vectors(index, vectors)

print(f"Uploaded {len(vectors)} vectors to Pinecone")