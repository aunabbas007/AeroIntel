from source.embeddings import model
from source.vector_store import create_index, search_vectors


index = create_index("aerointel-v1")


questions = [
    "Which aircraft does Emirates operate?",
    "Who won the 2026 World Cup?"
]


for question in questions:

    print(f"\nQuestion: {question}")

    query_embedding = model.encode(question)

    results = search_vectors(
        index,
        query_embedding,
        top_k=3,
        score_threshold=0.5
    )

    if not results["matches"]:
        print("No sufficiently relevant information found.")

    else:
        for match in results["matches"]:
            print(
                f"Score: {match['score']:.3f} | "
                f"Source: {match['metadata']['source']}"
            )