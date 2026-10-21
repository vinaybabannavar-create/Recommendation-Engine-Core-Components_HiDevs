"""Serve a small interactive dashboard for the recommendation components."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from typing import Any

from candidate_gen import CandidateGenerator, _tags_for
from evaluator import RecommendationEvaluator
from scorer import RecommendationScorer

AS_OF_DATE = "2026-10-21"
GROUND_TRUTH = {"user_1": {"item_103"}}
STRATEGIES = {"hybrid", "collaborative", "content", "popularity"}


class RecommendationDashboardHandler(BaseHTTPRequestHandler):
    """Handle dashboard assets and JSON requests using shared in-memory data."""

    generator = CandidateGenerator()
    scorer = RecommendationScorer(generator)
    evaluator = RecommendationEvaluator()

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _send_error_json(self, message: str, status: int) -> None:
        self._send_json({"error": message}, status)

    def _overview(self) -> None:
        users = [
            {"user_id": user_id, "label": user_id.replace("_", " ").title()}
            for user_id in sorted(self.generator.interactions)
        ]
        users.append({"user_id": "new_user", "label": "New user (cold start)"})
        interactions = sum(
            len(ratings) for ratings in self.generator.interactions.values()
        )
        self._send_json(
            {
                "users": users,
                "user_count": len(self.generator.interactions),
                "item_count": len(self.generator.items),
                "interaction_count": interactions,
            }
        )

    def _recommendations(self, query: dict[str, list[str]]) -> None:
        user_id = query.get("user_id", ["user_1"])[0]
        strategy = query.get("strategy", ["hybrid"])[0]
        try:
            limit = int(query.get("limit", ["5"])[0])
        except ValueError:
            self._send_error_json("limit must be an integer", 400)
            return
        if strategy not in STRATEGIES:
            self._send_error_json("Unknown recommendation strategy", 400)
            return
        if not 1 <= limit <= 20:
            self._send_error_json("limit must be between 1 and 20", 400)
            return
        if user_id not in self.generator.interactions and user_id != "new_user":
            self._send_error_json("Unknown user", 404)
            return

        if strategy == "hybrid":
            candidates = self.generator.hybrid_candidates(user_id)
        elif strategy == "collaborative":
            candidates = self.generator.collaborative_candidates(user_id)
        elif strategy == "content":
            candidates = self.generator.content_based_candidates(user_id)
        else:
            candidates = self.generator.popularity_candidates(
                limit=len(self.generator.items)
            )

        ranked = self.scorer.rank_candidates(
            user_id,
            candidates,
            limit=limit,
            context={"as_of": AS_OF_DATE},
        )
        for recommendation in ranked:
            item = self.generator.items[recommendation["item_id"]]
            recommendation["title"] = item.get("title", recommendation["item_id"])
            recommendation["category"] = item.get("category", "Uncategorized")
            recommendation["tags"] = item.get("tags", [])

        profile_tags: set[str] = set()
        for item_id in self.generator._positive_items(user_id):
            profile_tags.update(_tags_for(self.generator.items[item_id]))

        evaluation = None
        if user_id in GROUND_TRUTH:
            evaluation = self.evaluator.evaluate_all(
                {user_id: [item["item_id"] for item in ranked]},
                {user_id: GROUND_TRUTH[user_id]},
                limit,
            )["per_user"][user_id]

        self._send_json(
            {
                "user_id": user_id,
                "strategy": strategy,
                "limit": limit,
                "candidate_count": len(candidates),
                "rated_count": len(self.generator.interactions.get(user_id, {})),
                "interests": sorted(profile_tags),
                "recommendations": ranked,
                "evaluation": evaluation,
                "evaluation_note": (
                    "Toy held-out evaluation is available for user_1 only."
                ),
                "as_of": AS_OF_DATE,
            }
        )

    def do_GET(self) -> None:
        request = urlsplit(self.path)
        if request.path in ("/", "/index.html"):
            page = Path(__file__).with_name("index.html").read_bytes()
            self._send_bytes(page, "text/html; charset=utf-8")
        elif request.path == "/api/overview":
            self._overview()
        elif request.path == "/api/recommendations":
            self._recommendations(parse_qs(request.query))
        else:
            self._send_error_json("Not found", 404)

    def log_message(self, format_string: str, *args: object) -> None:
        """Keep the terminal log concise while retaining request visibility."""
        print(f"{self.address_string()} - {format_string % args}")


def main() -> None:
    """Start the dashboard server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    arguments = parser.parse_args()

    server = ThreadingHTTPServer(
        (arguments.host, arguments.port), RecommendationDashboardHandler
    )
    print(f"Recommendation dashboard running at http://localhost:{arguments.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping recommendation dashboard.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
