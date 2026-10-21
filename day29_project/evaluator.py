"""Offline ranking metrics for recommendation lists."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


class RecommendationEvaluator:
    """Calculate precision, recall, NDCG, and macro-averaged metrics."""

    @staticmethod
    def precision_at_k(
        recommendations: Iterable[str], relevant_items: Iterable[str], k: int
    ) -> float:
        """Return relevant hits divided by the number of returned items at k."""
        if k <= 0:
            return 0.0
        considered = list(recommendations)[:k]
        if not considered:
            return 0.0
        relevant = set(relevant_items)
        hits = len(set(considered) & relevant)
        return hits / len(considered)

    @staticmethod
    def recall_at_k(
        recommendations: Iterable[str], relevant_items: Iterable[str], k: int
    ) -> float:
        """Return relevant hits at k divided by all relevant items."""
        relevant = set(relevant_items)
        if k <= 0 or not relevant:
            return 0.0
        hits = len(set(list(recommendations)[:k]) & relevant)
        return hits / len(relevant)

    @staticmethod
    def ndcg_at_k(
        recommendations: Iterable[str], relevant_items: Iterable[str], k: int
    ) -> float:
        """Return binary normalized discounted cumulative gain at k."""
        relevant = set(relevant_items)
        if k <= 0 or not relevant:
            return 0.0
        ranked = list(recommendations)[:k]
        seen: set[str] = set()
        dcg = 0.0
        for position, item_id in enumerate(ranked):
            if item_id in relevant and item_id not in seen:
                dcg += 1.0 / math.log2(position + 2)
            seen.add(item_id)
        ideal_hits = min(len(relevant), k)
        ideal_dcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_hits))
        if ideal_dcg == 0.0:
            return 0.0
        return min(1.0, max(0.0, dcg / ideal_dcg))

    def evaluate_all(
        self,
        recommendations_dict: Mapping[str, Iterable[str]],
        ground_truth_dict: Mapping[str, Iterable[str]],
        k: int,
    ) -> dict[str, Any]:
        """Evaluate users with ground truth; empty truth sets score as zero.

        Missing ground-truth users are omitted. Users explicitly present with
        an empty truth set are included with zero metrics and in the macro
        denominator, making the policy explicit and deterministic.
        """
        per_user: dict[str, dict[str, float]] = {}
        for user_id, relevant_items in ground_truth_dict.items():
            recommendations = list(recommendations_dict.get(user_id, []))
            relevant_items = set(relevant_items)
            per_user[user_id] = {
                "precision": self.precision_at_k(recommendations, relevant_items, k),
                "recall": self.recall_at_k(recommendations, relevant_items, k),
                "ndcg": self.ndcg_at_k(recommendations, relevant_items, k),
            }
        evaluated_users = len(per_user)
        macro = {
            metric: (
                sum(values[metric] for values in per_user.values()) / evaluated_users
                if evaluated_users
                else 0.0
            )
            for metric in ("precision", "recall", "ndcg")
        }
        return {
            "per_user": per_user,
            "macro_average": macro,
            "evaluated_users": evaluated_users,
        }


if __name__ == "__main__":
    example = RecommendationEvaluator().evaluate_all(
        {"user_1": ["item_103", "item_107", "item_105"]},
        {"user_1": {"item_103", "item_105"}},
        k=3,
    )
    print("Example evaluation:", example)
