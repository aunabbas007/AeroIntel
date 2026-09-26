from source.embeddings import model
from source.vector_store import create_index,search_vectors

index = create_index("aerointel-v1")

query="Which aircarft does Emirates operate?"

query_embedding=model.encode(query)

results=search_vectors(index,query_embedding,top_k=3)

for match in results["matches"]:
    print(f"Score: {match['score']}")
    print(f"Source: {match['metadata']['source']}")
    print(f"Text: {match['metadata']['text']}")
    print("-----")
