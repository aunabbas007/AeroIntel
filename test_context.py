from source.embeddings import model
from source.vector_store import create_index, search_vectors
from source.context import build_context


index = create_index("aerointel-v1")

query = "Which aircraft does Emirates operate?"

query_embedding = model.encode(query)

results = search_vectors(index, query_embedding, top_k=3)

context = build_context(results)

print(context)