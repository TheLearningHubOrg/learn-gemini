"""
Google Gemini Enterprise Masterclass: CLI & Demonstrator.

Demonstrates:
1. 7-Stage Curriculum progress tracking.
2. Gemini 1.5 Pro generation with system instructions and generation config.
3. 2M+ Token Long-Context Needle-in-a-Haystack (NIAH) retrieval.
4. Context Caching with 75% cost reduction on large document stores.
5. Google Search Grounding with verified source citations.
6. Multi-tier token cost economics.
"""

from src.learn_gemini.client import (
    Content,
    GeminiClient,
    GeminiCostCalculator,
    GenerationConfig,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
    TextPart,
    Tool,
)
from src.learn_gemini.roadmap import GeminiRoadmap


def run_roadmap_showcase() -> None:
    roadmap = GeminiRoadmap()
    summary = roadmap.get_summary()

    print("=" * 80)
    print("GOOGLE GEMINI ENTERPRISE MASTERCLASS: 7-STAGE CURRICULUM")
    print("=" * 80)
    print(f"Curriculum Completion: {summary['overall_progress_pct']}% | Stages: {summary['completed_stages']}/{summary['total_stages']} Completed\n")

    for s in summary["stages"]:
        status_symbol = "[x]" if s["status"] == "COMPLETED" else "[ ]"
        print(f"  {status_symbol} Stage {s['stage']}: {s['title']} ({s['topics_count']} topics, {s['progress_pct']}%)")
    print("=" * 80 + "\n")


def run_generation_demo(client: GeminiClient) -> None:
    print("-" * 80)
    print("STAGE 1: GEMINI 1.5 PRO GENERATION & SYSTEM INSTRUCTIONS")
    print("-" * 80)

    config = GenerationConfig(temperature=0.2, top_p=0.95, top_k=40)
    safety = [
        SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE)
    ]

    response = client.generate_content(
        model="gemini-1.5-pro",
        contents="Explain Sparse Mixture-of-Experts (MoE) routing in Gemini 1.5.",
        config=config,
        safety_settings=safety,
        system_instruction="You are a Google DeepMind principal research scientist.",
    )

    print(f"Model: gemini-1.5-pro | Finish Reason: {response.candidates[0].finish_reason}")
    print(f"Text: \"{response.text}\"")
    print(f"Usage: Prompt={response.usage_metadata.prompt_token_count}, Output={response.usage_metadata.candidates_token_count}, Total={response.usage_metadata.total_token_count}\n")


def run_long_context_demo(client: GeminiClient) -> None:
    print("-" * 80)
    print("STAGE 2: 2M+ TOKEN LONG-CONTEXT NEEDLE-IN-A-HAYSTACK (NIAH)")
    print("-" * 80)

    secret_key = "DEEPMIND_QUANTUM_CORE_9921_ACTIVATED"
    context_tokens = 2_000_000
    depth_pct = 73.4

    result = client.simulate_needle_retrieval(
        context_length_tokens=context_tokens,
        needle=secret_key,
        needle_depth_pct=depth_pct,
    )

    print(f"Long-Context Window: {result['context_length_tokens']:,} tokens (2M context limit)")
    print(f"Needle Placed At:     Token {result['token_offset']:,} ({result['needle_depth_pct']}% depth)")
    print(f"Recall Accuracy:      {result['recall_score'] * 100:.1f}%")
    print(f"Model Retrieval:      \"{result['retrieved_text']}\"\n")


def run_context_caching_demo(client: GeminiClient) -> None:
    print("-" * 80)
    print("STAGE 3: CONTEXT CACHING (75% TOKEN COST REDUCTION)")
    print("-" * 80)

    # Cache large codebase / legal repository
    large_doc = Content(
        role="user",
        parts=[TextPart(text="Enterprise architectural specifications and legal agreements " * 2000)]
    )
    cache_resource = client.create_cached_content(
        model="gemini-1.5-pro",
        contents=[large_doc],
        ttl_seconds=3600,
    )

    print(f"Created Context Cache: {cache_resource.name}")
    print(f"Cached Tokens Pin:     {cache_resource.token_count:,} tokens in TPU HBM")
    print(f"Cache TTL:             {cache_resource.ttl_seconds} seconds (1 hour)\n")

    # Query against cache
    query_response = client.generate_content(
        model="gemini-1.5-pro",
        contents="What are the SLA terms for cross-region disaster recovery?",
        cached_content=cache_resource.name,
    )

    usage = query_response.usage_metadata
    print(f"Query executed against Context Cache:")
    print(f"  Total Prompt Tokens:  {usage.prompt_token_count:,}")
    print(f"  Cached Tokens (75% off): {usage.cached_content_token_count:,}")
    print(f"  Uncached Delta Tokens: {usage.prompt_token_count - usage.cached_content_token_count:,}\n")


def run_grounding_demo(client: GeminiClient) -> None:
    print("-" * 80)
    print("STAGE 5: GOOGLE SEARCH GROUNDING WITH VERIFIED WEB CITATIONS")
    print("-" * 80)

    search_tool = Tool(google_search=True)
    response = client.generate_content(
        model="gemini-1.5-flash",
        contents="What are the latest updates in Gemini 2.0 release?",
        tools=[search_tool],
    )

    print(f"Answer: {response.text}\n")
    if response.candidates[0].grounding_metadata:
        gm = response.candidates[0].grounding_metadata
        print(f"Verified Grounding Queries: {gm.web_search_queries}")
        print("Citations:")
        for chunk in gm.grounding_chunks:
            print(f"  - [{chunk.title}]({chunk.uri})")
    print()


def run_cost_demo() -> None:
    print("-" * 80)
    print("STAGE 3 & 6: ENTERPRISE GEMINI TOKEN COST ECONOMICS (50,000 REQUESTS)")
    print("-" * 80)

    reqs = 50_000
    prompt_tokens = reqs * 100_000
    cached_tokens = reqs * 80_000
    completion_tokens = reqs * 500

    print(f"Workload: {reqs:,} requests (100k prompt tokens, 80k cached in TPU HBM, 500 output tokens)\n")
    print(f"{'Model':<18} | {'Baseline Cost':<15} | {'With Cache (-75%)':<20} | {'Savings':<15}")
    print("-" * 72)

    for model in ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"]:
        std = GeminiCostCalculator.calculate_cost(model, prompt_tokens, completion_tokens, cached_tokens=0)
        cached = GeminiCostCalculator.calculate_cost(model, prompt_tokens, completion_tokens, cached_tokens=cached_tokens)

        print(f"{model:<18} | ${std['total_cost']:<14.2f} | ${cached['total_cost']:<19.2f} | ${cached['savings_usd']:<8.2f} ({cached['savings_pct']}%)")
    print()


def main() -> None:
    client = GeminiClient()
    run_roadmap_showcase()
    run_generation_demo(client)
    run_long_context_demo(client)
    run_context_caching_demo(client)
    run_grounding_demo(client)
    run_cost_demo()


if __name__ == "__main__":
    main()
