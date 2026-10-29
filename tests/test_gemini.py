"""
Comprehensive Unit Tests for Google Gemini Enterprise Engine.

Tests:
- Client configuration and content generation.
- Long-context 2M token needle-in-a-haystack retrieval simulation.
- Context Caching creation and token reduction.
- Function calling and Google Search Grounding metadata.
- Enterprise token cost calculator with caching discounts.
- Pedagogical roadmap stage completion.
"""

import unittest

from src.learn_gemini.client import (
    Content,
    FunctionCallPart,
    GeminiClient,
    GeminiCostCalculator,
    TextPart,
    Tool,
)
from src.learn_gemini.roadmap import GeminiRoadmap


class TestGeminiEngine(unittest.TestCase):

    def setUp(self) -> None:
        self.client = GeminiClient(api_key="ai-studio-test-key")

    def test_client_initialization(self) -> None:
        self.assertEqual(self.client.api_key, "ai-studio-test-key")

    def test_generate_content_standard(self) -> None:
        response = self.client.generate_content(
            model="gemini-1.5-pro",
            contents="Explain TPU v5p architecture.",
        )
        self.assertTrue(len(response.candidates) > 0)
        self.assertEqual(response.candidates[0].finish_reason, "STOP")
        self.assertIn("Gemini", response.text)
        self.assertTrue(response.usage_metadata.prompt_token_count > 0)
        self.assertTrue(response.usage_metadata.candidates_token_count > 0)

    def test_simulate_needle_retrieval(self) -> None:
        result = self.client.simulate_needle_retrieval(
            context_length_tokens=2_000_000,
            needle="MISSION_CRITICAL_KEY_8832",
            needle_depth_pct=45.0,
        )
        self.assertEqual(result["context_length_tokens"], 2_000_000)
        self.assertEqual(result["recall_score"], 1.0)
        self.assertTrue(result["retrieved_successfully"])
        self.assertIn("MISSION_CRITICAL_KEY_8832", result["retrieved_text"])

    def test_context_caching(self) -> None:
        doc = Content(role="user", parts=[TextPart(text="Enterprise documentation " * 500)])
        cached = self.client.create_cached_content(
            model="gemini-1.5-pro",
            contents=[doc],
            ttl_seconds=1800,
        )
        self.assertIn(cached.name, self.client.cached_contents)
        self.assertTrue(cached.token_count >= 32768)

        # Generate using cached resource
        res = self.client.generate_content(
            model="gemini-1.5-pro",
            contents="Summarize documentation.",
            cached_content=cached.name,
        )
        self.assertEqual(res.usage_metadata.cached_content_token_count, cached.token_count)
        self.assertTrue(res.usage_metadata.prompt_token_count > cached.token_count)

    def test_tool_calling_function(self) -> None:
        tool = Tool(
            function_declarations=[
                {
                    "name": "lookup_customer_balance",
                    "description": "Returns account balance for a given customer ID.",
                    "parameters": {"type": "object", "properties": {"customer_id": {"type": "string"}}},
                }
            ]
        )
        response = self.client.generate_content(
            model="gemini-1.5-flash",
            contents="Please call lookup_customer_balance for customer 451.",
            tools=[tool],
        )
        self.assertEqual(len(response.candidates), 1)
        part = response.candidates[0].content.parts[0]
        self.assertIsInstance(part, FunctionCallPart)
        self.assertEqual(part.name, "lookup_customer_balance")

    def test_google_search_grounding(self) -> None:
        tool = Tool(google_search=True)
        response = self.client.generate_content(
            model="gemini-1.5-flash",
            contents="Latest AI news today.",
            tools=[tool],
        )
        self.assertIsNotNone(response.candidates[0].grounding_metadata)
        gm = response.candidates[0].grounding_metadata
        self.assertTrue(len(gm.web_search_queries) > 0)
        self.assertTrue(len(gm.grounding_chunks) > 0)

    def test_cost_calculator(self) -> None:
        # 1M prompt, 1M completion on gemini-1.5-pro with 800k cached
        cost = GeminiCostCalculator.calculate_cost(
            model="gemini-1.5-pro",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            cached_tokens=800_000,
        )

        # Baseline: 1M (>128k) @ $2.50 + 1M output @ $5.00 = $7.50
        self.assertAlmostEqual(cost["baseline_cost"], 7.50, places=2)
        # With 800k cached @ $0.3125/1M ($0.25) + 200k uncached @ $2.50/1M ($0.50) + 1M output @ $5.00 = $5.75
        self.assertAlmostEqual(cost["total_cost"], 5.75, places=2)
        self.assertTrue(cost["savings_usd"] > 0)

    def test_roadmap_progress(self) -> None:
        roadmap = GeminiRoadmap()
        self.assertEqual(len(roadmap.stages), 7)
        self.assertEqual(roadmap.total_progress(), 100.0)
        summary = roadmap.get_summary()
        self.assertEqual(summary["total_stages"], 7)
        self.assertEqual(summary["completed_stages"], 7)
        self.assertEqual(summary["overall_progress_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
