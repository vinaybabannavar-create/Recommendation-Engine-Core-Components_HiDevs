"""Candidate generation from collaborative, content, and popularity signals."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from similarity import SimilarityCalculator

SAMPLE_ITEMS: dict[str, dict[str, Any]] = {
    "item_101": {
        "title": "Python Basics",
        "category": "programming",
        "tags": ["python", "coding"],
        "added": "2026-05-10",
    },
    "item_102": {
        "title": "Practical Data Analysis",
        "category": "data",
        "tags": ["python", "data"],
        "added": "2026-04-20",
    },
    "item_103": {
        "title": "Introduction to Machine Learning",
        "category": "AI",
        "tags": ["AI", "python", "data"],
        "added": "2026-08-12",
    },
    "item_104": {
        "title": "Designing Accessible Interfaces",
        "category": "design",
        "tags": ["design", "accessibility"],
        "added": "2026-02-18",
    },
    "item_105": {
        "title": "SQL for Analysts",
        "category": "data",
        "tags": ["SQL", "data"],
        "added": "2026-06-01",
    },
    "item_106": {
        "title": "Neural Networks Explained",
        "category": "AI",
        "tags": ["AI", "deep learning"],
        "added": "2026-09-01",
    },
    "item_107": {
        "title": "Building Web APIs",
        "category": "programming",
        "tags": ["python", "web", "API"],
        "added": "2026-03-15",
    },
    "item_108": {
        "title": "Cloud Computing Foundations",
        "category": "cloud",
        "tags": ["cloud", "infrastructure"],
        "added": "2026-07-01",
    },
    "item_109": {
        "title": "Product Design Workshop",
        "category": "design",
        "tags": ["design", "product"],
        "added": "2026-05-22",
    },
    "item_110": {
        "title": "Data Visualization",
        "category": "data",
        "tags": ["data", "visualization"],
        "added": "2026-08-05",
    },
}

# Ratings of 4 or 5 are positive interactions by default. The target user's
# held-out item_103 is deliberately absent; see the integration workflow below.
SAMPLE_INTERACTIONS: dict[str, dict[str, int]] = {
    "user_1": {"item_101": 5, "item_102": 4, "item_104": 2},
    "user_2": {"item_101": 5, "item_103": 5, "item_105": 4, "item_107": 4},
    "user_3": {"item_102": 4, "item_106": 5, "item_108": 4},
    "user_4": {"item_103": 4, "item_109": 5, "item_110": 4},
    "user_5": {"item_105": 5, "item_107": 4, "item_108": 4, "item_110": 5},
}


def _tags_for(item: Mapping[str, Any]) -> set[str]:
    """Collect an item's tags and category as normalized interest tokens."""
    tags = item.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    result = {str(tag).strip().casefold() for tag in tags if str(tag).strip()}
    category = item.get("category")
    if category:
        result.add(str(category).strip().casefold())
    return result


class CandidateGenerator:
    """Generate valid, unseen item IDs using in-memory user and item data.

    Interactions are ratings; ratings at or above ``interaction_threshold``
    count as positive for profiles and neighbors. Any rated item is excluded
    from a user's recommendations, including low-rated items.
    """

    def __init__(
        self,
        interactions: Mapping[str, Mapping[str, float]] | None = None,
        items: Mapping[str, Mapping[str, Any]] | None = None,
        popularity: Mapping[str, float] | None = None,
        interaction_threshold: float = 4.0,
        max_candidates: int = 20,
    ) -> None:
        """Copy supplied datasets and configure positive ratings and limits."""
        if interaction_threshold < 0:
            raise ValueError("interaction_threshold must be nonnegative")
        if max_candidates < 0:
            raise ValueError("max_candidates must be nonnegative")
        self.interactions = deepcopy(
            dict(interactions if interactions is not None else SAMPLE_INTERACTIONS)
        )
        self.items = deepcopy(dict(items if items is not None else SAMPLE_ITEMS))
        self.popularity = dict(popularity or {})
        self.interaction_threshold = interaction_threshold
        self.max_candidates = max_candidates

    def _limit(self, limit: int | None) -> int:
        actual = self.max_candidates if limit is None else limit
        if actual < 0:
            raise ValueError("limit must be nonnegative")
        return actual

    def _positive_items(self, user_id: str) -> set[str]:
        return {
            item_id
            for item_id, rating in self.interactions.get(user_id, {}).items()
            if rating >= self.interaction_threshold and item_id in self.items
        }

    def _popularity_scores(self) -> dict[str, float]:
        counts = {item_id: 0.0 for item_id in self.items}
        for ratings in self.interactions.values():
            for item_id, rating in ratings.items():
                if item_id in counts and rating >= self.interaction_threshold:
                    counts[item_id] += 1.0
        return {
            item_id: float(self.popularity.get(item_id, count))
            for item_id, count in counts.items()
        }

    def collaborative_candidates(
        self, user_id: str, limit: int | None = None
    ) -> list[str]:
        """Rank unseen items by similar users' positive ratings."""
        result_limit = self._limit(limit)
        target_history = self._positive_items(user_id)
        rated_items = set(self.interactions.get(user_id, {}))
        if not target_history or result_limit == 0:
            return []
        scores: dict[str, float] = {}
        for other_user, ratings in self.interactions.items():
            if other_user == user_id:
                continue
            neighbor_history = self._positive_items(other_user)
            similarity = SimilarityCalculator.jaccard_similarity(
                target_history, neighbor_history
            )
            if similarity <= 0.0:
                continue
            for item_id, rating in ratings.items():
                if (
                    item_id in self.items
                    and item_id not in rated_items
                    and rating >= self.interaction_threshold
                ):
                    scores[item_id] = scores.get(item_id, 0.0) + similarity * float(
                        rating
                    )
        ordered = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))
        return ordered[:result_limit]

    def content_based_candidates(
        self, user_id: str, limit: int | None = None
    ) -> list[str]:
        """Recommend unseen items with tags matching positive item interests."""
        result_limit = self._limit(limit)
        profile: set[str] = set()
        rated_items = set(self.interactions.get(user_id, {}))
        for item_id in self._positive_items(user_id):
            profile.update(_tags_for(self.items[item_id]))
        if not profile or result_limit == 0:
            return []
        scores: dict[str, float] = {}
        for item_id, item in self.items.items():
            if item_id not in rated_items:
                score = SimilarityCalculator.jaccard_similarity(
                    profile, _tags_for(item)
                )
                if score > 0.0:
                    scores[item_id] = score
        ordered = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))
        return ordered[:result_limit]

    def popularity_candidates(self, limit: int | None = None) -> list[str]:
        """Return all items ordered by positive-interaction popularity."""
        result_limit = self._limit(limit)
        scores = self._popularity_scores()
        ordered = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))
        return ordered[:result_limit]

    def hybrid_candidates(self, user_id: str, limit: int | None = None) -> list[str]:
        """Blend personalized rank signals with lower-weight popularity.

        Reciprocal-rank contributions from collaborative and content results
        each have weight 1.0; popularity receives weight 0.35. Thus popularity
        is a useful fallback for cold-start users but does not dominate matches.
        """
        result_limit = self._limit(limit)
        if result_limit == 0:
            return []
        rated_items = set(self.interactions.get(user_id, {}))
        ranked_sources = (
            (self.collaborative_candidates(user_id), 1.0),
            (self.content_based_candidates(user_id), 1.0),
            (self.popularity_candidates(), 0.35),
        )
        scores: dict[str, float] = {}
        for candidates, weight in ranked_sources:
            for rank, item_id in enumerate(candidates, start=1):
                if item_id not in rated_items and item_id in self.items:
                    scores[item_id] = scores.get(item_id, 0.0) + weight / rank
        ordered = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))
        return ordered[:result_limit]


if __name__ == "__main__":
    generator = CandidateGenerator()
    print("Hybrid candidates for user_1:", generator.hybrid_candidates("user_1"))
