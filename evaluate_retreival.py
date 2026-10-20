
from source.vector_store import create_vector_store

vector_store = create_vector_store("aerointel-v2")

TEST_CASES = [
    {
        "question": "What is the maximum range of the Airbus A350-900?",
        "expected_terms": ["8,600", "nautical miles"],
        "expected_entity": "a350_900",
    },
    {
        "question": "What is the primary hub of Emirates?",
        "expected_terms": ["Dubai International Airport"],
        "expected_entity": "emirates",
    },
    {
        "question": "Which engine powers the Airbus A350-900?",
        "expected_terms": ["Trent XWB-84"],
        "expected_entity": "a350_900",
    },
    {
        "question": "What is the range of the Boeing 787-9?",
        "expected_terms": ["8,300", "nautical miles"],
        "expected_entity": "787_9",
    },
    {
        "question": "What are the dimensions of the Boeing 787-9?",
        "expected_terms": ["62.8", "60.1"],
        "expected_entity": "787_9",
    },
]


def evaluate_retrieval():
    keyword_passes = 0
    top1_passes = 0

    print("=" * 65)
    print("AEROINTEL RETRIEVAL EVALUATION")
    print("=" * 65)

    for number, test in enumerate(TEST_CASES, start=1):
        documents = vector_store.similarity_search(
            test["question"], k=3
        )

        retrieved_text = "\n".join(
            doc.page_content for doc in documents
        ).lower()

        matched_terms = [
            term for term in test["expected_terms"]
            if term.lower() in retrieved_text
        ]

        keyword_pass = (
            len(matched_terms) == len(test["expected_terms"])
        )

        top_entity = (
            documents[0].metadata.get("entity_id")
            if documents else None
        )

        top1_pass = top_entity == test["expected_entity"]

        keyword_passes += keyword_pass
        top1_passes += top1_pass

        print(f"\nTest {number}: {test['question']}")
        print(f"  Keyword coverage: {'PASS' if keyword_pass else 'FAIL'}")
        print(f"  Expected entity:  {test['expected_entity']}")
        print(f"  Top-1 entity:     {top_entity}")
        print(f"  Top-1 accuracy:   {'PASS' if top1_pass else 'FAIL'}")

        print("  Retrieved entities:")
        for rank, doc in enumerate(documents, start=1):
            entity = doc.metadata.get("entity_id", "unknown")
            print(f"    {rank}. {entity}")

    total = len(TEST_CASES)

    print("\n" + "=" * 65)
    print(f"Keyword coverage: {keyword_passes}/{total} "
          f"({keyword_passes / total * 100:.1f}%)")
    print(f"Top-1 accuracy:   {top1_passes}/{total} "
          f"({top1_passes / total * 100:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_retrieval()
