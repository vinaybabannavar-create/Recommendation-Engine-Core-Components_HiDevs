"""Weighted recommendation scoring and deterministic ranking."""

from __future__ import annotations

import math
from datetime import date
from numbers import Real
from typing import Any, Callable, Mapping

from candidate_gen import CandidateGenerator, _tags_for
from similarity import SimilarityCalculator

ScoringFunction = Callable[[str, str, Mapping[str, Any]], float]


class RecommendationScorer:
    """Score and rank candidates using normalized weighted factors.

    Scorers use the signature ``function(user_id, item_id, context)``. Each
    result must lie in [0, 1]. Weights are divided by their sum, so weights
    such as 6, 2, and 1 behave like 0.60, 0.20, and 0.10.
    """

    def __init__(
        self,
        candidate_generator: CandidateGenerator | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        """Attach a dataset and register relevance, popularity, and recency."""
        self.generator = candidate_generator or CandidateGenerator()
        configured = (
            {"relevance": 0.60, "popularity": 0.25, "recency": 0.15}
            if weights is None
            else dict(weights)
        )
        self.scorers: dict[str, tuple[ScoringFunction, float]] = {}
        defaults: dict[str, ScoringFunction] = {
            "relevance": self._relevance_score,
            "popularity": self._popularity_score,
            "recency": self._recency_score,
        }
        for name, weight in configured.items():
            if name not in defaults:
                raise ValueError(f"Unknown default scoring factor: {name}")
            self.add_scorer(name, defaults[name], weight)

    def add_scorer(self, name: str, function: ScoringFunction, weight: float) -> None:
        """Register or replace a factor after validating its name and weight."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Scorer name must be a nonempty string")
        if not callable(function):
            raise TypeError("Scoring function must be callable")
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise TypeError("Scorer weight must be numeric")
        if not math.isfinite(float(weight)) or weight < 0:
            raise ValueError("Scorer weight must be finite and nonnegative")
        self.scorers[name.strip()] = (function, float(weight))

    def _user_tags(self, user_id: str) -> set[str]:
        tags: set[str] = set()
        for item_id in self.generator._positive_items(user_id):
            tags.update(_tags_for(self.generator.items[item_id]))
        return tags

    def _relevance_score(
        self, user_id: str, item_id: str, context: Mapping[str, Any]
    ) -> float:
        return SimilarityCalculator.jaccard_similarity(
            self._user_tags(user_id), _tags_for(self.generator.items[item_id])
        )

    def _popularity_score(
        self, user_id: str, item_id: str, context: Mapping[str, Any]
    ) -> float:
        scores = self.generator._popularity_scores()
        maximum = max(scores.values(), default=0.0)
        if maximum <= 0.0:
            return 0.0
        return scores.get(item_id, 0.0) / maximum

    def _recency_score(
        self, user_id: str, item_id: str, context: Mapping[str, Any]
    ) -> float:
        added = self.generator.items[item_id].get("added")
        if not added:
            return 0.0
        try:
            added_date = date.fromisoformat(str(added))
            as_of = date.fromisoformat(
                str(context.get("as_of", date.today().isoformat()))
            )
        except (TypeError, ValueError):
            return 0.0
        age_days = max(0, (as_of - added_date).days)
        return max(0.0, 1.0 - age_days / 365.0)

    def _component_scores(
        self, user_id: str, item_id: str, context: Mapping[str, Any]
    ) -> dict[str, float]:
        if item_id not in self.generator.items:
            return {}
        scores: dict[str, float] = {}
        for name, (function, _) in self.scorers.items():
            value = function(user_id, item_id, context)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Scorer '{name}' must return a numeric score")
            score = float(value)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Scorer '{name}' must return a finite value from 0.0 to 1.0"
                )
            scores[name] = score
        return scores

    def _weighted_score(self, component_scores: Mapping[str, float]) -> float:
        total_weight = sum(weight for _, weight in self.scorers.values())
        if total_weight == 0.0:
            return 0.0
        weighted_sum = sum(
            component_scores[name] * weight
            for name, (_, weight) in self.scorers.items()
        )
        return weighted_sum / total_weight

    def calculate_score(
        self, user_id: str, item_id: str, context: Mapping[str, Any] | None = None
    ) -> float:
        """Return the normalized weighted score, or 0 for missing items."""
        if context is not None and not isinstance(context, Mapping):
            raise TypeError("context must be a mapping")
        components = self._component_scores(user_id, item_id, context or {})
        return self._weighted_score(components)

    def _explanation(self, components: Mapping[str, float]) -> str:
        relevance = components.get("relevance", 0.0)
        popularity = components.get("popularity", 0.0)
        recency = components.get("recency", 0.0)
        reasons: list[str] = []
        if relevance >= 0.2 and recency >= 0.65:
            reasons.append("Recently added and relevant to your interests")
        elif relevance >= 0.2:
            reasons.append("Matches your interests")
        elif recency >= 0.65:
            reasons.append("Recently added")
        if popularity >= 0.5:
            reasons.append("Popular among other users")
        return "; ".join(reasons[:2]) or "Recommended based on available signals"

    def rank_candidates(
        self,
        user_id: str,
        candidates: list[str],
        limit: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return scored candidate dictionaries, excluding previously rated items."""
        if context is not None and not isinstance(context, Mapping):
            raise TypeError("context must be a mapping")
        if limit is not None and limit < 0:
            raise ValueError("limit must be nonnegative")
        result_limit = len(candidates) if limit is None else limit
        already_rated = set(self.generator.interactions.get(user_id, {}))
        results: list[tuple[float, str, str]] = []
        seen: set[str] = set()
        for item_id in candidates:
            if (
                item_id in seen
                or item_id in already_rated
                or item_id not in self.generator.items
            ):
                continue
            seen.add(item_id)
            components = self._component_scores(user_id, item_id, context or {})
            score = self._weighted_score(components)
            reason = self._explanation(components)
            results.append((score, item_id, reason))
        results.sort(key=lambda result: (-result[0], result[1]))
        return [
            {"item_id": item_id, "score": round(score, 4), "reason": reason}
            for score, item_id, reason in results[:result_limit]
        ]


if __name__ == "__main__":
    generator = CandidateGenerator()
    scorer = RecommendationScorer(generator)
    candidates = generator.hybrid_candidates("user_1")
    print(scorer.rank_candidates("user_1", candidates, context={"as_of": "2026-10-21"}))
