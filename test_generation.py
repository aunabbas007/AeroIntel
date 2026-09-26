from source.embeddings import model
from source.vector_store import create_index, search_vectors
from source.context import build_context
from source.generator import generate_answer


index = create_index("aerointel-v1")


question = "Which aircraft does Emirates operate?"


query_embedding = model.encode(question)


results = search_vectors(
    index,
    query_embedding,
    top_k=3
)


context = build_context(results)


answer = generate_answer(
    question,
    context
)


print("\nAnswer:")
print(answer)