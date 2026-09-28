from source.embeddings import model
from source.vector_store import create_index, search_vectors
from source.context import build_context
from source.generator import generate_answer


index = create_index("aerointel-v1")


def ask_aerointel(question):
    query_embedding = model.encode(question)

    results = search_vectors(
        index,
        query_embedding,
        top_k=3,
        score_threshold=0.5
    )

    context = build_context(results)

    answer = generate_answer(
        question,
        context
    )

    return answer


if __name__ == "__main__":
    question = input("Ask AeroIntel: ")

    answer = ask_aerointel(question)

    print("\nAeroIntel:")
    print(answer)