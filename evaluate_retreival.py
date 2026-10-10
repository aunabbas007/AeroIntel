
from source.vector_store import create_vector_store


# Connect to the existing AeroIntel Pinecone index
vector_store = create_vector_store("aerointel-v2")


# Small evaluation dataset
TEST_CASES = [
    {
        "question": "What is the maximum range of the Airbus A350-900?",
        "expected_terms": ["8,600", "nautical miles"],
    },
    
    {
        "question": "What is the primary hub of Emirates?",
        "expected_terms": ["Dubai International Airport"],
    },
    {
        "question": "Which engine powers the Airbus A350-900?",
        "expected_terms": ["Trent XWB-84"],
    },
    {
        "question": "What is the range of the Boeing 787-9?",
        "expected_terms": ["8,300", "nautical miles"],
    },
    {
        "question": "What are the dimensions of the Boeing 787-9?",
        "expected_terms": ["62.8", "60.1"],
    },
]


def evaluate_retrieval():
    passed = 0

    print("=" * 65)
    print("AEROINTEL RETRIEVAL EVALUATION")
    print("=" * 65)

    for number, test in enumerate(TEST_CASES, start=1):
        question = test["question"]
        expected_terms = test["expected_terms"]

        # Retrieve the top 3 matching chunks
        documents = vector_store.similarity_search(
            question,
            k=3,
        )

        # Combine the retrieved text for a simple keyword check
        retrieved_text = "\n".join(
            document.page_content
            for document in documents
        ).lower()

        matched_terms = [
            term
            for term in expected_terms
            if term.lower() in retrieved_text
        ]

        # A test passes only if all expected terms are present
        passed_test = len(matched_terms) == len(expected_terms)

        if passed_test:
            passed += 1

        status = "PASS" if passed_test else "FAIL"

        print(f"\nTest {number}: {status}")
        print(f"Question: {question}")
        print(
            f"Expected terms found: "
            f"{len(matched_terms)}/{len(expected_terms)}"
        )

        if not passed_test:
            missing = [
                term
                for term in expected_terms
                if term.lower() not in retrieved_text
            ]
            print(f"Missing terms: {missing}")

        # Show the actual retrieved chunks for manual inspection
        for rank, document in enumerate(documents, start=1):
            print(f"\n  Retrieved chunk {rank}")
            print(f"  Metadata: {document.metadata}")
            print(f"  Text: {document.page_content[:500]}")

    total = len(TEST_CASES)
    score = (passed / total) * 100

    print("\n" + "=" * 65)
    print(f"RESULT: {passed}/{total} tests passed")
    print(f"Keyword coverage pass rate: {score:.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_retrieval()
