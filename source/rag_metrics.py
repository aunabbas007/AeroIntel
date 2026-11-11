"""Small, dependency-light retrieval metrics used by AeroIntel evaluations."""


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    top = retrieved[:k]
    return sum(item in relevant for item in top) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(rankings: list[list[str]], relevant_sets: list[set[str]]) -> float:
    if len(rankings) != len(relevant_sets):
        raise ValueError("rankings and relevant_sets must have equal length")
    if not rankings:
        return 0.0
    return sum(
        reciprocal_rank(ranking, relevant)
        for ranking, relevant in zip(rankings, relevant_sets)
    ) / len(rankings)
