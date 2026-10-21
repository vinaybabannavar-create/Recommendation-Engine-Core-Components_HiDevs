"""Unit tests and a deterministic end-to-end recommendation demonstration."""

from __future__ import annotations

import unittest
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen
import json

from app import RecommendationDashboardHandler
from candidate_gen import CandidateGenerator, SAMPLE_INTERACTIONS, SAMPLE_ITEMS
from evaluator import RecommendationEvaluator
from scorer import RecommendationScorer
from similarity import SimilarityCalculator


class SimilarityTests(unittest.TestCase):
    def test_cosine_identical_vectors(self) -> None:
        self.assertAlmostEqual(
            SimilarityCalculator.cosine_similarity([1, 2], [1, 2]), 1.0
        )

    def test_cosine_orthogonal_vectors(self) -> None:
        self.assertEqual(SimilarityCalculator.cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_cosine_empty_and_zero_vectors(self) -> None:
        self.assertEqual(SimilarityCalculator.cosine_similarity([], []), 0.0)
        self.assertEqual(SimilarityCalculator.cosine_similarity([0, 0], [2, 3]), 0.0)

    def test_cosine_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            SimilarityCalculator.cosine_similarity([1], [1, 2])

    def test_cosine_rejects_non_numeric_values(self) -> None:
        with self.assertRaises(TypeError):
            SimilarityCalculator.cosine_similarity([1, "x"], [1, 2])  # type: ignore[list-item]

    def test_jaccard_identical_sets(self) -> None:
        self.assertEqual(SimilarityCalculator.jaccard_similarity({1, 2}, {1, 2}), 1.0)

    def test_jaccard_disjoint_sets(self) -> None:
        self.assertEqual(SimilarityCalculator.jaccard_similarity({1}, {2}), 0.0)

    def test_jaccard_empty_sets(self) -> None:
        self.assertEqual(SimilarityCalculator.jaccard_similarity([], []), 1.0)
        self.assertEqual(SimilarityCalculator.jaccard_similarity([], [1]), 0.0)

    def test_pearson_positive_correlation(self) -> None:
        self.assertAlmostEqual(
            SimilarityCalculator.pearson_correlation([1, 2, 3], [2, 4, 6]), 1.0
        )

    def test_pearson_negative_and_constant_correlation(self) -> None:
        self.assertAlmostEqual(
            SimilarityCalculator.pearson_correlation([1, 2, 3], [3, 2, 1]), -1.0
        )
        self.assertEqual(SimilarityCalculator.pearson_correlation([2, 2], [1, 3]), 0.0)

    def test_pearson_short_and_mismatched_inputs(self) -> None:
        self.assertEqual(SimilarityCalculator.pearson_correlation([1], [1]), 0.0)
        with self.assertRaises(ValueError):
            SimilarityCalculator.pearson_correlation([1], [1, 2])


class CandidateGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = CandidateGenerator()

    def test_collaborative_recommendations_are_valid(self) -> None:
        candidates = self.generator.collaborative_candidates("user_1")
        self.assertIn("item_103", candidates)
        self.assertTrue(all(item in SAMPLE_ITEMS for item in candidates))

    def test_content_recommendations_match_profile(self) -> None:
        candidates = self.generator.content_based_candidates("user_1")
        self.assertTrue(candidates)
        self.assertIn("item_103", candidates)

    def test_popularity_ordering_is_deterministic(self) -> None:
        self.assertEqual(
            self.generator.popularity_candidates()[:3],
            ["item_101", "item_102", "item_103"],
        )

    def test_hybrid_deduplicates_items(self) -> None:
        candidates = self.generator.hybrid_candidates("user_1")
        self.assertEqual(len(candidates), len(set(candidates)))

    def test_previously_rated_items_are_excluded(self) -> None:
        for method in (
            self.generator.collaborative_candidates,
            self.generator.content_based_candidates,
            self.generator.hybrid_candidates,
        ):
            self.assertTrue(
                set(method("user_1")).isdisjoint(SAMPLE_INTERACTIONS["user_1"])
            )

    def test_missing_and_cold_start_users(self) -> None:
        self.assertEqual(self.generator.collaborative_candidates("missing"), [])
        self.assertEqual(self.generator.content_based_candidates("missing"), [])
        self.assertEqual(
            self.generator.hybrid_candidates("missing")[:3],
            self.generator.popularity_candidates()[:3],
        )

    def test_empty_dataset_behavior(self) -> None:
        empty = CandidateGenerator(interactions={}, items={})
        self.assertEqual(empty.collaborative_candidates("any"), [])
        self.assertEqual(empty.content_based_candidates("any"), [])
        self.assertEqual(empty.popularity_candidates(), [])
        self.assertEqual(empty.hybrid_candidates("any"), [])

    def test_configured_limit_and_unknown_items(self) -> None:
        generator = CandidateGenerator(
            interactions={"u": {"valid": 5, "unknown": 5}},
            items={"valid": {"tags": ["x"]}, "candidate": {"tags": ["x"]}},
            max_candidates=1,
        )
        self.assertEqual(generator.content_based_candidates("u"), ["candidate"])

    def test_input_data_is_not_mutated(self) -> None:
        interactions = {"u": {"a": 5}}
        items = {"a": {"tags": ["topic"]}}
        CandidateGenerator(interactions=interactions, items=items)
        self.assertEqual(interactions, {"u": {"a": 5}})
        self.assertEqual(items, {"a": {"tags": ["topic"]}})


class RecommendationScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = CandidateGenerator()
        self.scorer = RecommendationScorer(self.generator)

    def test_weighted_score_and_normalization(self) -> None:
        scorer = RecommendationScorer(self.generator, weights={})
        scorer.add_scorer("one", lambda user, item, context: 1.0, 2)
        scorer.add_scorer("zero", lambda user, item, context: 0.0, 1)
        self.assertAlmostEqual(scorer.calculate_score("u", "item_101"), 2 / 3)

    def test_invalid_scorer_registration(self) -> None:
        with self.assertRaises(ValueError):
            self.scorer.add_scorer(" ", lambda user, item, context: 0.5, 1)
        with self.assertRaises(TypeError):
            self.scorer.add_scorer("bad", None, 1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self.scorer.add_scorer("negative", lambda user, item, context: 0.5, -1)

    def test_invalid_scorer_outputs(self) -> None:
        scorer = RecommendationScorer(self.generator, weights={})
        scorer.add_scorer("string", lambda user, item, context: "bad", 1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            scorer.calculate_score("u", "item_101")
        scorer.scorers.clear()
        scorer.add_scorer("range", lambda user, item, context: 1.1, 1)
        with self.assertRaises(ValueError):
            scorer.calculate_score("u", "item_101")

    def test_rank_order_and_result_limit(self) -> None:
        scorer = RecommendationScorer(self.generator, weights={})
        scorer.add_scorer(
            "fixed",
            lambda user, item, context: {"item_103": 0.3, "item_106": 0.9}.get(
                item, 0.5
            ),
            1,
        )
        ranked = scorer.rank_candidates(
            "user_1", ["item_103", "item_106", "item_105"], limit=2
        )
        self.assertEqual(
            [entry["item_id"] for entry in ranked], ["item_106", "item_105"]
        )

    def test_rank_explanations_and_exclusion(self) -> None:
        ranked = self.scorer.rank_candidates(
            "user_1", ["item_103", "item_101"], context={"as_of": "2026-10-21"}
        )
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["item_id"], "item_103")
        self.assertTrue(ranked[0]["reason"])
        self.assertGreaterEqual(ranked[0]["score"], 0.0)

    def test_no_scorer_and_missing_item(self) -> None:
        scorer = RecommendationScorer(self.generator, weights={})
        self.assertEqual(scorer.calculate_score("u", "item_101"), 0.0)
        self.assertEqual(scorer.calculate_score("u", "missing"), 0.0)

    def test_zero_weights_return_zero(self) -> None:
        scorer = RecommendationScorer(self.generator, weights={})
        scorer.add_scorer("zero", lambda user, item, context: 1.0, 0)
        self.assertEqual(scorer.calculate_score("u", "item_101"), 0.0)


class RecommendationEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = RecommendationEvaluator()

    def test_precision_at_k_uses_actual_list_length(self) -> None:
        self.assertEqual(self.evaluator.precision_at_k(["a", "x"], {"a", "b"}, 5), 0.5)

    def test_recall_at_k(self) -> None:
        self.assertEqual(self.evaluator.recall_at_k(["a", "x"], {"a", "b"}, 2), 0.5)

    def test_ndcg_at_k_ideal_and_discounted(self) -> None:
        self.assertEqual(self.evaluator.ndcg_at_k(["a", "b"], {"a", "b"}, 2), 1.0)
        self.assertLess(self.evaluator.ndcg_at_k(["x", "a"], {"a"}, 2), 1.0)

    def test_empty_and_nonpositive_metric_inputs(self) -> None:
        self.assertEqual(self.evaluator.precision_at_k([], {"a"}, 3), 0.0)
        self.assertEqual(self.evaluator.recall_at_k(["a"], set(), 3), 0.0)
        self.assertEqual(self.evaluator.ndcg_at_k(["a"], {"a"}, 0), 0.0)

    def test_missing_and_empty_ground_truth_policy(self) -> None:
        result = self.evaluator.evaluate_all(
            {"known": ["a"], "unknown": ["b"]}, {"known": {"a"}, "empty": set()}, 1
        )
        self.assertNotIn("unknown", result["per_user"])
        self.assertEqual(result["per_user"]["empty"]["recall"], 0.0)
        self.assertEqual(result["evaluated_users"], 2)

    def test_macro_averages_and_no_evaluable_users(self) -> None:
        result = self.evaluator.evaluate_all(
            {"a": ["x"], "b": ["y"]}, {"a": {"x"}, "b": {"z"}}, 1
        )
        self.assertEqual(result["macro_average"]["precision"], 0.5)
        empty = self.evaluator.evaluate_all({}, {}, 3)
        self.assertEqual(empty["evaluated_users"], 0)
        self.assertEqual(empty["macro_average"]["ndcg"], 0.0)

    def test_evaluate_all_accepts_one_shot_iterables(self) -> None:
        result = self.evaluator.evaluate_all(
            {"u": iter(["a", "b"])}, {"u": iter(["a"])}, 2
        )
        self.assertEqual(result["per_user"]["u"]["recall"], 1.0)


def run_demo() -> None:
    """Run a leakage-free toy train/test recommendation workflow."""
    generator = CandidateGenerator()
    scorer = RecommendationScorer(generator)
    user_id = "user_1"
    candidates = generator.hybrid_candidates(user_id)
    ranked = scorer.rank_candidates(
        user_id, candidates, limit=5, context={"as_of": "2026-10-21"}
    )
    recommendations = [entry["item_id"] for entry in ranked]
    # item_103 is held out for user_1 and is not in that user's training data.
    ground_truth = {user_id: {"item_103"}}
    metrics = RecommendationEvaluator().evaluate_all(
        {user_id: recommendations}, ground_truth, k=5
    )
    print("Recommendation Engine Core Components demo")
    print(f"User: {user_id}; candidates: {len(candidates)}; ranked: {recommendations}")
    print(f"Held-out relevant items: {sorted(ground_truth[user_id])}")
    print(f"Metrics at 5: {metrics['per_user'][user_id]}")
    print(f"Macro averages: {metrics['macro_average']}")


class IntegrationTests(unittest.TestCase):
    def test_end_to_end_candidate_ranking_and_evaluation(self) -> None:
        generator = CandidateGenerator()
        candidates = generator.hybrid_candidates("user_1")
        ranked = RecommendationScorer(generator).rank_candidates(
            "user_1", candidates, context={"as_of": "2026-10-21"}
        )
        item_ids = [entry["item_id"] for entry in ranked]
        result = RecommendationEvaluator().evaluate_all(
            {"user_1": item_ids}, {"user_1": {"item_103"}}, 5
        )
        self.assertTrue(candidates)
        self.assertTrue(all("score" in entry and "reason" in entry for entry in ranked))
        self.assertEqual(result["evaluated_users"], 1)
        self.assertGreaterEqual(result["per_user"]["user_1"]["recall"], 0.0)


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), RecommendationDashboardHandler
        )
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_dashboard_page_loads(self) -> None:
        with urlopen(self.base_url) as response:
            page = response.read().decode("utf-8")
        self.assertIn("Recommendation lab", page)

    def test_personalized_api_uses_engine_and_evaluator(self) -> None:
        url = f"{self.base_url}/api/recommendations?user_id=user_1&strategy=hybrid&limit=5"
        with urlopen(url) as response:
            result = json.loads(response.read())
        self.assertEqual(result["recommendations"][0]["item_id"], "item_103")
        self.assertEqual(result["evaluation"]["recall"], 1.0)

    def test_cold_start_api_returns_popular_items(self) -> None:
        url = f"{self.base_url}/api/recommendations?user_id=new_user&strategy=hybrid&limit=3"
        with urlopen(url) as response:
            result = json.loads(response.read())
        self.assertEqual(result["rated_count"], 0)
        self.assertEqual(len(result["recommendations"]), 3)
        self.assertIsNone(result["evaluation"])

    def test_api_rejects_invalid_strategy(self) -> None:
        url = f"{self.base_url}/api/recommendations?user_id=user_1&strategy=unknown"
        with self.assertRaises(HTTPError) as error:
            urlopen(url)
        error.exception.close()
        self.assertEqual(error.exception.code, 400)


if __name__ == "__main__":
    run_demo()
