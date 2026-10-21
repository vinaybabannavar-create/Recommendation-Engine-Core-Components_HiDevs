"""Small, dependency-free similarity measures for recommendation systems."""

from __future__ import annotations

import math
from numbers import Real
from typing import Iterable, Sequence


def _numeric_values(values: Sequence[Real], name: str) -> list[float]:
    """Convert a numeric sequence to finite floats with a useful error."""
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must contain only numeric values")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name} must contain only finite numeric values")
        result.append(converted)
    return result


class SimilarityCalculator:
    """Calculate common vector and set similarities."""

    @staticmethod
    def cosine_similarity(vec1: Sequence[Real], vec2: Sequence[Real]) -> float:
        """Return cosine similarity clamped to [0, 1]."""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same length")
        first = _numeric_values(vec1, "vec1")
        second = _numeric_values(vec2, "vec2")
        if not first:
            return 0.0
        magnitude_first = math.sqrt(sum(value * value for value in first))
        magnitude_second = math.sqrt(sum(value * value for value in second))
        if magnitude_first == 0.0 or magnitude_second == 0.0:
            return 0.0
        dot_product = sum(left * right for left, right in zip(first, second))
        similarity = dot_product / (magnitude_first * magnitude_second)
        return min(1.0, max(0.0, similarity))

    @staticmethod
    def jaccard_similarity(set1: Iterable[object], set2: Iterable[object]) -> float:
        """Return intersection-over-union; two empty inputs have similarity 1."""
        first = set(set1)
        second = set(set2)
        union = first | second
        if not union:
            return 1.0
        return len(first & second) / len(union)

    @staticmethod
    def pearson_correlation(
        ratings1: Sequence[Real], ratings2: Sequence[Real]
    ) -> float:
        """Return Pearson correlation, or 0 when correlation is undefined."""
        if len(ratings1) != len(ratings2):
            raise ValueError("Rating sequences must have the same length")
        first = _numeric_values(ratings1, "ratings1")
        second = _numeric_values(ratings2, "ratings2")
        if len(first) < 2:
            return 0.0
        mean_first = sum(first) / len(first)
        mean_second = sum(second) / len(second)
        deviations_first = [value - mean_first for value in first]
        deviations_second = [value - mean_second for value in second]
        numerator = sum(
            left * right for left, right in zip(deviations_first, deviations_second)
        )
        sum_first = sum(value * value for value in deviations_first)
        sum_second = sum(value * value for value in deviations_second)
        denominator = math.sqrt(sum_first * sum_second)
        if denominator == 0.0:
            return 0.0
        correlation = numerator / denominator
        return min(1.0, max(-1.0, correlation))


if __name__ == "__main__":
    print("Cosine:", SimilarityCalculator.cosine_similarity([1, 2], [1, 2]))
    print("Jaccard:", SimilarityCalculator.jaccard_similarity({1, 2}, {2, 3}))
    print("Pearson:", SimilarityCalculator.pearson_correlation([1, 2, 3], [3, 2, 1]))
