
from source.vector_store import (
    create_vector_store,
    create_retriever,
)


def main():
    vector_store = create_vector_store()

    tests = [
        ("aircraft", "What is the range of the Airbus A350-900?"),
        ("airline", "What is the primary hub of Emirates?"),
    ]

    for entity_type, question in tests:
        print(f"\n{'=' * 60}")
        print(f"Type: {entity_type}")
        print(f"Question: {question}")

        retriever = create_retriever(
            vector_store,
            k=3,
            entity_type=entity_type,
        )

        documents = retriever.invoke(question)

        for rank, doc in enumerate(documents, start=1):
            print(
                f"{rank}. "
                f"Entity: {doc.metadata.get('entity_id')} | "
                f"Type: {doc.metadata.get('type')}"
            )


if __name__ == "__main__":
    main()
