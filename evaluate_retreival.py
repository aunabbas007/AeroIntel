
from source.vector_store import create_vector_store


INDEX_NAME = "aerointel-v2"
K = 3

# Expected entity IDs and factual terms are derived from
# the AeroIntel source documents supplied for this evaluation.
TEST_CASES = [
    # -------------------------------------------------
    # 1-6: Aircraft specifications
    # -------------------------------------------------
    {
        "question": "What is the maximum range of the Airbus A350-900?",
        "type": "aircraft",
        "relevant_entities": ["a350_900"],
        "expected_terms": ["8,600", "nautical miles"],
        "answerable": True,
    },
    {
        "question": "Which engine powers the Airbus A350-900?",
        "type": "aircraft",
        "relevant_entities": ["a350_900"],
        "expected_terms": ["Trent XWB-84"],
        "answerable": True,
    },
    {
        "question": "What is the maximum seating capacity of the Airbus A350-1000?",
        "type": "aircraft",
        "relevant_entities": ["a350_1000"],
        "expected_terms": ["480 passengers"],
        "answerable": True,
    },
    {
        "question": "What is the range of the Airbus A350-1000?",
        "type": "aircraft",
        "relevant_entities": ["a350_1000"],
        "expected_terms": ["9,100", "nautical miles"],
        "answerable": True,
    },
    {
        "question": "What is the range of the Boeing 787-9?",
        "type": "aircraft",
        "relevant_entities": ["787_9"],
        "expected_terms": ["8,300", "nautical miles"],
        "answerable": True,
    },
    {
        "question": "What is the length of the Boeing 787-10?",
        "type": "aircraft",
        "relevant_entities": ["787_10"],
        "expected_terms": ["68.3 metres"],
        "answerable": True,
    },

    # -------------------------------------------------
    # 7-10: Airline information
    # -------------------------------------------------
    {
        "question": "What is the primary hub of Emirates?",
        "type": "airline",
        "relevant_entities": ["emirates"],
        "expected_terms": ["Dubai International Airport"],
        "answerable": True,
    },
    {
        "question": "Which airport is the hub of British Airways?",
        "type": "airline",
        "relevant_entities": ["british_airways"],
        "expected_terms": ["London Heathrow Airport"],
        "answerable": True,
    },
    {
        "question": "What is the hub of Etihad Airways?",
        "type": "airline",
        "relevant_entities": ["etihad_airways"],
        "expected_terms": ["Abu Dhabi International Airport"],
        "answerable": True,
    },
    {
        "question": "Which hubs are listed for Lufthansa?",
        "type": "airline",
        "relevant_entities": ["lufthansa"],
        "expected_terms": ["Frankfurt Airport", "Munich Airport"],
        "answerable": True,
    },

    # -------------------------------------------------
    # 11-14: Comparisons
    # -------------------------------------------------
    {
        "question": "Compare the range of the Airbus A350-900 and A350-1000.",
        "type": "aircraft",
        "relevant_entities": ["a350_900", "a350_1000"],
        "expected_terms": ["8,600", "9,100"],
        "answerable": True,
    },
    {
        "question": "Compare the range and length of the Boeing 787-9 and 787-10.",
        "type": "aircraft",
        "relevant_entities": ["787_9", "787_10"],
        "expected_terms": ["8,300", "7,500", "62.8", "68.3"],
        "answerable": True,
    },
    {
        "question": "Compare the hubs of Emirates and Etihad Airways.",
        "type": "airline",
        "relevant_entities": ["emirates", "etihad_airways"],
        "expected_terms": [
            "Dubai International Airport",
            "Abu Dhabi International Airport",
        ],
        "answerable": True,
    },
    {
        "question": "Compare the fleet information available for Qatar Airways and Singapore Airlines.",
        "type": "airline",
        "relevant_entities": ["qatar_airways", "singapore_airlines"],
        "expected_terms": [
            "Airbus A350-900",
            "Boeing 787-10",
        ],
        "answerable": True,
    },

    # -------------------------------------------------
    # 15-17: Cross-document questions
    # -------------------------------------------------
    {
        "question": "Which airlines in the knowledge base list the Airbus A380 in their fleets?",
        "type": "airline",
        "relevant_entities": [
            "emirates",
            "british_airways",
            "etihad_airways",
            "lufthansa",
            "qatar_airways",
            "singapore_airlines",
        ],
        "expected_terms": ["Airbus A380"],
        "answerable": True,
    },
    {
        "question": "Which airlines list the Boeing 787-9 in their fleets?",
        "type": "airline",
        "relevant_entities": [
            "air_india",
            "qatar_airways",
            "lufthansa",
            "etihad_airways",
            "turkish_airlines",
        ],
        "expected_terms": ["787-9"],
        "answerable": True,
    },
    {
        "question": "Which aircraft does Singapore Airlines list in its fleet?",
        "type": "airline",
        "relevant_entities": ["singapore_airlines"],
        "expected_terms": [
            "Airbus A350-900",
            "Airbus A380-800",
            "Boeing 777-300ER",
        ],
        "answerable": True,
    },

    # -------------------------------------------------
    # 18-20: Unsupported questions
    # These are for a separate answerability evaluation.
    # Do not include them in retrieval relevance averages.
    # -------------------------------------------------
    {
        "question": "What is the cockpit crew's average age on the A350-900?",
        "type": None,
        "relevant_entities": [],
        "expected_terms": [],
        "answerable": False,
    },
    {
        "question": "What was Emirates' exact operating profit in 2025?",
        "type": None,
        "relevant_entities": [],
        "expected_terms": [],
        "answerable": False,
    },
    {
        "question": "What is the hourly maintenance cost of the Boeing 787-9?",
        "type": None,
        "relevant_entities": [],
        "expected_terms": [],
        "answerable": False,
    },
]



def evaluate_retrieval(vector_store, test, mode, k):
    """Evaluate semantic, type-filtered, or oracle entity-filtered retrieval."""

    search_kwargs = {"k": k}

    if mode == "type_filtered" and test["type"] is not None:
        search_kwargs["filter"] = {"type": test["type"]}

    elif mode == "entity_oracle":
        # Only evaluate single-entity questions in this mode.
        # This uses the known answer entity as an oracle.
        if len(test["relevant_entities"]) != 1:
            return None

        search_kwargs["filter"] = {
            "entity_id": test["relevant_entities"][0]
        }

    documents = vector_store.similarity_search(
        test["question"],
        **search_kwargs
    )

    retrieved_entities = [
        doc.metadata.get("entity_id")
        for doc in documents
    ]

    if not test["answerable"]:
        return None

    relevant_entities = set(test["relevant_entities"])
    retrieved_set = set(retrieved_entities)

    relevant_chunks = sum(
        entity in relevant_entities
        for entity in retrieved_entities
    )

    precision = (
        relevant_chunks / len(retrieved_entities)
        if retrieved_entities else 0.0
    )

    recall = (
        len(retrieved_set & relevant_entities)
        / len(relevant_entities)
        if relevant_entities else 0.0
    )

    expected_terms = test["expected_terms"]
    retrieved_text = "\n".join(
        doc.page_content for doc in documents
    ).lower()

    keyword_coverage = (
        sum(term.lower() in retrieved_text for term in expected_terms)
        / len(expected_terms)
        if expected_terms else None
    )

    return {
        "entities": retrieved_entities,
        "precision": precision,
        "recall": recall,
        "hit": float(bool(retrieved_set & relevant_entities)),
        "top1": float(
            bool(retrieved_entities)
            and retrieved_entities[0] in relevant_entities
        ),
        "keyword_coverage": keyword_coverage,
    }


def print_summary(label, results):
    """Print macro averages over supported questions."""

    metric_names = [
        ("precision", f"Precision@{K}"),
        ("recall", f"Recall@{K}"),
        ("hit", f"Hit Rate@{K}"),
        ("top1", "Top-1 entity accuracy"),
        ("keyword_coverage", "Keyword coverage"),
    ]

    print(f"\n{'=' * 72}")
    print(label)
    print("=" * 72)

    for key, name in metric_names:
        values = [
            result[key]
            for result in results
            if result[key] is not None
        ]

        if values:
            average = sum(values) / len(values)
            print(f"{name:<25} {average:.1%} ({len(values)} cases)")



def main():
    vector_store = create_vector_store(INDEX_NAME)
    k_values = [3, 5, 8]

    modes = [
        ("semantic", "Semantic retrieval"),
        ("type_filtered", "Type-filtered retrieval"),
        ("entity_oracle", "Entity-filtered oracle baseline"),
    ]

    supported = [t for t in TEST_CASES if t["answerable"]]

    print("=" * 72)
    print("AEROINTEL RETRIEVAL STRATEGY COMPARISON")
    print(f"Index: {INDEX_NAME}")
    print(f"Supported cases: {len(supported)}")
    print(f"K values: {k_values}")
    print("=" * 72)

    all_results = {}

    for mode, label in modes:
        for k in k_values:
            results = []

            for test in supported:
                result = evaluate_retrieval(
                    vector_store,
                    test,
                    mode=mode,
                    k=k,
                )

                # Oracle mode applies only to single-entity questions.
                if result is not None:
                    results.append(result)

            key = (mode, k)
            all_results[key] = results

            print(f"\n{label} | k={k}")
            print("-" * 72)

            for metric in [
                "precision",
                "recall",
                "hit",
                "top1",
                "keyword_coverage",
            ]:
                values = [
                    r[metric] for r in results
                    if r[metric] is not None
                ]

                if values:
                    average = sum(values) / len(values)
                    print(
                        f"{metric:<22} {average:.1%} "
                        f"({len(values)} cases)"
                    )

    print("\n" + "=" * 72)
    print("SUMMARY: TYPE-FILTERED RETRIEVAL BY K")
    print("=" * 72)
    print(f"{'K':<8}{'Precision':<14}{'Recall':<14}"
          f"{'Top-1':<14}{'Keyword coverage':<18}")

    for k in k_values:
        results = all_results[("type_filtered", k)]

        def avg(metric):
            values = [
                r[metric] for r in results
                if r[metric] is not None
            ]
            return sum(values) / len(values) if values else 0.0

        print(
            f"{k:<8}{avg('precision'):<14.1%}"
            f"{avg('recall'):<14.1%}"
            f"{avg('top1'):<14.1%}"
            f"{avg('keyword_coverage'):<18.1%}"
        )

    print("\nNotes:")
    print("- The oracle baseline uses the expected entity ID.")
    print("- Oracle results are not comparable to normal retrieval.")
    print("- These metrics measure retrieval, not answer correctness.")
    print("- Larger k can improve recall while reducing precision.")


if __name__ == "__main__":
    main()
