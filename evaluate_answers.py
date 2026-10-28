
import time
import requests


API_URL = "http://127.0.0.1:8000/ask"

# Initial regression set based on the AeroIntel source documents.
TEST_CASES = [
    {
        "question": "What is the maximum range of the Airbus A350-900?",
        "expected_terms": ["8,600", "nautical miles"],
        "answerable": True,
    },
    {
        "question": "Which engine powers the Airbus A350-900?",
        "expected_terms": ["Trent XWB-84"],
        "answerable": True,
    },
    {
        "question": "What is the maximum seating capacity of the Airbus A350-1000?",
        "expected_terms": ["480"],
        "answerable": True,
    },
    {
        "question": "What is the primary hub of Emirates?",
        "expected_terms": ["Dubai International Airport"],
        "answerable": True,
    },
    {
        "question": "Compare the range of the Airbus A350-900 and A350-1000.",
        "expected_terms": ["8,600", "9,100"],
        "answerable": True,
    },
    {
        "question": "What is the cockpit crew's average age on the A350-900?",
        "expected_terms": [],
        "answerable": False,
    },
    {
        "question": "What was Emirates' exact operating profit in 2025?",
        "expected_terms": [],
        "answerable": False,
    },
    {
        "question": "What is the hourly maintenance cost of the Boeing 787-9?",
        "expected_terms": [],
        "answerable": False,
    },
]

ABSTENTION_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "not enough information",
    "information is not available",
    "cannot answer",
    "can't answer",
    "unable to determine",
    "not specified in the knowledge base",
]


def evaluate_answer(session, test, token):
    headers = {"Authorization": f"Bearer {token}"}

    started = time.perf_counter()

    try:
        response = session.post(
            API_URL,
            headers=headers,
            json={"question": test["question"]},
            timeout=180,
        )

        latency = time.perf_counter() - started

        if response.status_code != 200:
            return {
                "status": response.status_code,
                "latency": latency,
                "answer": response.text,
                "passed": False,
            }

        answer = response.json()["answer"]
        answer_lower = answer.lower()

        if test["answerable"]:
            matched = [
                term for term in test["expected_terms"]
                if term.lower() in answer_lower
            ]

            coverage = (
                len(matched) / len(test["expected_terms"])
                if test["expected_terms"] else 0.0
            )

            passed = coverage == 1.0
            abstained = False

        else:
            matched = []
            coverage = None

            abstained = any(
                phrase in answer_lower
                for phrase in ABSTENTION_PHRASES
            )

            passed = abstained

        return {
            "status": response.status_code,
            "latency": latency,
            "answer": answer,
            "coverage": coverage,
            "abstained": abstained,
            "passed": passed,
        }

    except requests.RequestException as exc:
        return {
            "status": "REQUEST_ERROR",
            "latency": time.perf_counter() - started,
            "answer": str(exc),
            "passed": False,
        }


def main():
    token = input("Paste a valid AeroIntel JWT access token: ").strip()

    if not token:
        print("A token is required.")
        return

    passed = 0
    latencies = []

    with requests.Session() as session:
        for index, test in enumerate(TEST_CASES, start=1):
            print(f"\n{'=' * 65}")
            print(f"Test {index}: {test['question']}")

            result = evaluate_answer(session, test, token)

            print(f"HTTP status: {result['status']}")
            print(f"Latency: {result['latency']:.2f}s")
            print(f"Answer:\n{result['answer']}")

            if result["status"] == 200:
                latencies.append(result["latency"])

                if test["answerable"]:
                    print(
                        "Expected-term coverage: "
                        f"{result['coverage']:.0%}"
                    )
                    print(f"Term check passed: {result['passed']}")
                else:
                    print(
                        "Abstention phrase detected: "
                        f"{result['abstained']}"
                    )
                    print(f"Abstention check passed: {result['passed']}")

            if result["status"] == 200:
                if result["passed"]:
                    passed += 1
            else:
                print("Test status: INCONCLUSIVE (API/request failure)")

    print(f"\n{'=' * 65}")
    print("AEROINTEL ANSWER EVALUATION SUMMARY")

    total = len(TEST_CASES)
    inconclusive = total - len(latencies)
    evaluated = len(latencies)

    print(f"Total tests: {total}")
    print(f"Evaluated responses: {evaluated}")
    print(f"Inconclusive tests: {inconclusive}")
    print(f"Passed: {passed}")
    print(f"Failed among evaluated: {evaluated - passed}")

    if evaluated:
        print(f"Pass rate among evaluated: {passed / evaluated:.1%}")
        print(f"Average latency: {sum(latencies) / evaluated:.2f}s")

if __name__ == "__main__":
    main()
