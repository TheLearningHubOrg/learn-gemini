"""
learn_gemini: Production-Grade Google Gemini Platform Engineering Engine.

Provides 2M+ long-context retrieval, native multimodality, context caching (85% discount),
function calling, Google Search grounding metadata, and enterprise pricing calculators.
"""

from .client import (
    GeminiClient,
    ContentPart,
    TextPart,
    BlobPart,
    Content,
    GenerationConfig,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
    CachedContent,
    Tool,
    GenerateContentResponse,
    GeminiCostCalculator,
)
from .roadmap import GeminiRoadmap

__all__ = [
    "GeminiClient",
    "ContentPart",
    "TextPart",
    "BlobPart",
    "Content",
    "GenerationConfig",
    "SafetySetting",
    "HarmCategory",
    "HarmBlockThreshold",
    "CachedContent",
    "Tool",
    "GenerateContentResponse",
    "GeminiCostCalculator",
    "GeminiRoadmap",
]
