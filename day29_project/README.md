# Recommendation Engine Core Components

A beginner-friendly, standard-library-only Python project that demonstrates four reusable parts of a recommendation system: similarity calculation, candidate generation, weighted ranking, and offline evaluation. Everything runs on small in-memory dictionaries; there is no database, external network dependency, paid service, or machine-learning dependency. A local web dashboard lets you interact with the same components in a browser.

## Problem and Objectives

A recommender needs to identify useful items, order them, explain why they were selected, and measure how well the ranked list matches known relevant items. This project separates those responsibilities so each component can be understood, tested, and replaced independently.

## Architecture

- `similarity.py` provides cosine similarity, Jaccard similarity, and Pearson correlation.
- `candidate_gen.py` builds personalized and popularity-based item lists from ratings and item metadata.
- `scorer.py` combines relevance, popularity, and recency into a normalized score and explanation.
- `evaluator.py` computes precision@k, recall@k, NDCG@k, and macro averages.
- `app.py` provides a standard-library HTTP API and serves the interactive dashboard.
- `index.html` contains the responsive browser interface.
- `test.py` contains the `unittest` suite and a runnable end-to-end demonstration.

## Project Structure

```text
day29_project/
|-- similarity.py
|-- candidate_gen.py
|-- scorer.py
|-- evaluator.py
|-- app.py
|-- index.html
|-- test.py
|-- README.md
`-- requirements.txt
```

## Requirements and Setup

Python 3.10 or later is required. There are no external dependencies; `requirements.txt` documents this.

Run commands from the `day29_project/` directory:

```sh
python --version
python -m unittest -v test
python test.py
python app.py
```

Open [http://localhost:8000](http://localhost:8000) after starting `python app.py`. The dashboard lets you select a sample user, choose a hybrid, collaborative, content-based, or popularity strategy, and adjust the result count. It displays ranked items, scores, reasons, user interests, and held-out metrics for `user_1`. Press `Ctrl+C` in the server terminal to stop it.

Each core module can also be run directly for a small example:

```sh
python similarity.py
python candidate_gen.py
python scorer.py
python evaluator.py
```

## Public Components

### SimilarityCalculator

All methods are static. `cosine_similarity(vec1, vec2)` compares numeric vectors and clamps results to [0, 1], returning zero for empty or zero-magnitude vectors. `jaccard_similarity(set1, set2)` compares iterable sets; two empty inputs score 1. `pearson_correlation(ratings1, ratings2)` measures linear correlation in [-1, 1], returning zero when it is undefined. Mismatched vectors and invalid numeric values raise clear errors.

Example: `{python, data}` and `{python, ai}` have Jaccard similarity $1/3$, since one of three unique tags overlaps.

### CandidateGenerator

The constructor accepts optional `interactions`, `items`, `popularity`, `interaction_threshold` (default 4.0), and `max_candidates` (default 20). Data is copied to avoid mutating caller-owned dictionaries. Ratings at or above the threshold count as positive for profiles and neighbor recommendations; every rated item, including a low rating, is excluded for that user.

- `collaborative_candidates(user_id, limit=None)` finds users with overlapping positive histories and accumulates their positive item ratings weighted by Jaccard similarity.
- `content_based_candidates(user_id, limit=None)` compares tags and categories in the user's positive-item profile with unseen items.
- `popularity_candidates(limit=None)` counts positive interactions, or uses any supplied per-item popularity values, sorting by score then item ID.
- `hybrid_candidates(user_id, limit=None)` blends reciprocal ranks. Collaborative and content results each have weight 1.0; popularity has weight 0.35. With no profile, popularity supplies cold-start results.

Unknown users and empty datasets are supported. Only known item IDs are returned.

### RecommendationScorer

`RecommendationScorer(candidate_generator=None, weights=None)` uses the generator's dataset and registers relevance (0.60), popularity (0.25), and recency (0.15) by default. Weights are normalized by their total when computing the weighted average. `add_scorer(name, function, weight)` registers additional factors using the callable signature `function(user_id, item_id, context)`. Functions must return numeric values from 0 through 1. `calculate_score(...)` returns a normalized score. `rank_candidates(user_id, candidates, limit=None, context=None)` returns dictionaries containing `item_id`, rounded display `score`, and a short `reason`; internal precision determines ordering. Existing interactions and unknown IDs are omitted. The optional `context["as_of"]` ISO date makes recency scoring deterministic in tests and examples.

### RecommendationEvaluator

`precision_at_k`, `recall_at_k`, and `ndcg_at_k` operate on ranked item ID iterables and a relevant-item iterable. Precision divides by the actual number of returned recommendations up to k. NDCG uses binary relevance and logarithmic rank discounts. `evaluate_all(recommendations_dict, ground_truth_dict, k)` returns `per_user`, `macro_average`, and `evaluated_users`. Users missing from ground truth are omitted. Explicitly empty ground-truth sets are included with zero metrics and count in the macro average.

## Sample Data and Integration

The sample dataset has five users, ten items, five-point ratings, categories, tags, and fixed publication dates. In the demo and dashboard, `user_1`'s ratings in the candidate-generation dataset are training data. `item_103` is held out as the test relevant item; it is not included in `user_1`'s interaction history. Recommendations are evaluated against that held-out item. The fixed `as_of` date keeps the output reproducible. Evaluation is shown only for `user_1`; other user profiles do not have ground truth.

Expected output is a concise summary containing the user, ranked item IDs, the held-out relevant item, per-user precision/recall/NDCG, and macro averages. Exact scores are computed from the algorithms and sample data, not hardcoded.

For the included dataset and fixed date, the key output is:

```text
User: user_1; candidates: 7; ranked: ['item_103', 'item_107', 'item_110', 'item_105', 'item_108']
Held-out relevant items: ['item_103']
Metrics at 5: {'precision': 0.2, 'recall': 1.0, 'ndcg': 1.0}
Macro averages: {'precision': 0.2, 'recall': 1.0, 'ndcg': 1.0}
```

## Metrics and Limitations

Precision@k measures the fraction of returned top-k items that are relevant. Recall@k measures the fraction of all known relevant items retrieved. NDCG@k rewards relevant items near the top more strongly. Macro averages give each evaluated user equal weight.

This is an educational in-memory implementation. Jaccard overlap and simple ratings do not model confidence or implicit feedback in depth; popularity can favor already popular items; the recency formula is a simple one-year linear decay; and the toy data is too small for meaningful real-world quality claims. Future work could add configurable decay functions, richer evaluation splits, persistence, and larger sparse datasets.

## Submission Packaging

From the repository root, create the required archive with:

```sh
zip -r recommendation_engine_day29.zip day29_project
```

The archive should contain `day29_project/` as its top-level directory, all seven required core files, and the `app.py` and `index.html` dashboard files.