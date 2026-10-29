"""
Pedagogical Roadmap: 7-Stage Google Gemini Enterprise Masterclass.

Tracks curriculum progress across Architecture, 2M+ Long-Context Retrieval,
Context Caching, Native Multimodality, Search Grounding, and Vertex AI Enterprise Production.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class StageStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


@dataclass
class Topic:
    id: str
    title: str
    summary: str
    completed: bool = False


@dataclass
class Stage:
    stage_number: int
    title: str
    description: str
    topics: List[Topic]
    status: StageStatus = StageStatus.NOT_STARTED

    @property
    def progress_percentage(self) -> float:
        if not self.topics:
            return 0.0
        completed = sum(1 for t in self.topics if t.completed)
        return (completed / len(self.topics)) * 100.0


class GeminiRoadmap:
    """Manages progress and milestones for the 7-Stage Google Gemini curriculum."""

    def __init__(self) -> None:
        self.stages: Dict[int, Stage] = {
            1: Stage(
                stage_number=1,
                title="Architecture, Model Topology & Google GenAI SDK",
                description="Gemini 1.5 Pro, Flash, Flash-Lite topology, MoE architectures, google-genai unified SDK, and system instructions.",
                topics=[
                    Topic("1.1", "Model Topology & MoE Architecture", "Sparse Mixture-of-Experts (MoE) routing, throughput, and memory bandwidth efficiency.", True),
                    Topic("1.2", "Unified Google GenAI SDK", "Migrating from legacy google-generativeai to google-genai unified client interfaces.", True),
                    Topic("1.3", "Generation Controls", "Configuring temperature, top-p, top-k (Gemini specific), candidate_count, and stop_sequences.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            2: Stage(
                stage_number=2,
                title="2M+ Token Long-Context & Needle-in-a-Haystack Retrieval",
                description="2M token context window, Needle-in-a-Haystack (NIAH) 99.7% recall, full codebase analysis, and multi-hour video processing.",
                topics=[
                    Topic("2.1", "Long-Context Transformer Mechanics", "Rotary Position Embeddings (RoPE) scaling, RingAttention, and infinite memory illusion.", True),
                    Topic("2.2", "Needle-in-a-Haystack Benchmarking", "Evaluating recall accuracy across varying token depths (0% to 100%) and context lengths.", True),
                    Topic("2.3", "In-Context Learning vs RAG", "Architectural trade-offs: when to dump entire codebases/books into context vs vector chunking.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            3: Stage(
                stage_number=3,
                title="85% Cost Reduction with Context Caching",
                description="Creating explicit CachedContent resources, TTL expiration, 75-85% token discount, and multi-turn sharing.",
                topics=[
                    Topic("3.1", "Context Caching Architecture", "Precomputing and pinning KV states in TPU HBM across requests with explicit resource IDs.", True),
                    Topic("3.2", "Cache Lifecycle & TTL Management", "Setting TTL duration (default 1h), extending TTL dynamically, and automated cache cleanup.", True),
                    Topic("3.3", "Cost Modeling & Minimum Tokens", "Minimum 32,768 token threshold for Flash / 1.5 Pro and calculating ROI across repeat queries.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            4: Stage(
                stage_number=4,
                title="Native Multimodal Ingestion: Video, Audio, PDFs & Spatial Reasoning",
                description="Processing 1 hour of video at 1 FPS, audio transcription with timestamps, multi-page PDFs, and 2D bounding boxes.",
                topics=[
                    Topic("4.1", "Native Video Processing", "1 frame per second token sampling (258 tokens per second) and visual action retrieval.", True),
                    Topic("4.2", "Audio & Speech Ingestion", "Raw audio waveform processing, multilingual acoustic understanding, and background noise isolation.", True),
                    Topic("4.3", "Spatial Object Detection", "Generating 2D bounding box coordinates [ymin, xmin, ymax, xmax] normalized to [0, 1000].", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            5: Stage(
                stage_number=5,
                title="Tool Use, Code Execution & Google Search Grounding",
                description="Function declarations, Python sandbox code execution, dynamic grounding with Google Search, and source citations.",
                topics=[
                    Topic("5.1", "Function Calling Orchestration", "Declaring tools with OpenAPI schemas and executing function calls with auto/forced modes.", True),
                    Topic("5.2", "Python Code Execution Sandbox", "Enabling code_execution tool to compute complex math and data transformations securely.", True),
                    Topic("5.3", "Google Search Grounding", "Enabling search grounding to retrieve fresh web data with grounding_metadata & search_entry_point.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            6: Stage(
                stage_number=6,
                title="Enterprise Production: Vertex AI, Safety Guardrails & Quotas",
                description="Google Cloud Vertex AI enterprise hosting, safety filter categories & thresholds, rate limits, and private VPC peering.",
                topics=[
                    Topic("6.1", "Vertex AI vs Google AI Studio", "Enterprise IAM governance, HIPAA/SOC2 compliance, private service connect, and billing.", True),
                    Topic("6.2", "Safety Settings & Block Thresholds", "Configuring HARM_CATEGORY thresholds (Hate Speech, Harassment, Dangerous Content).", True),
                    Topic("6.3", "Production Quota Management", "Tier 1-4 rate limits, Request per Minute (RPM) governors, and automatic fallback failover.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            7: Stage(
                stage_number=7,
                title="Staff-Level Interview Prep, Cheatsheet & Appendix",
                description="Architectural comparison matrix, production outage mitigation runbooks, and staff engineering interview mastery.",
                topics=[
                    Topic("7.1", "Architectural Comparisons", "Gemini vs GPT-4o vs Claude 3.5 Sonnet across context length, multimodality, and pricing.", True),
                    Topic("7.2", "Incident Runbooks", "Handling safety filter false-positives, TPU capacity degradation, and cache evictions.", True),
                    Topic("7.3", "Master Interview Q&A", "Deep-dive answers to 40+ staff-level long-context, multimodality, and systems questions.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
        }

    def get_stage(self, stage_number: int) -> Optional[Stage]:
        return self.stages.get(stage_number)

    def total_progress(self) -> float:
        total_topics = sum(len(s.topics) for s in self.stages.values())
        if total_topics == 0:
            return 0.0
        completed = sum(sum(1 for t in s.topics if t.completed) for s in self.stages.values())
        return (completed / total_topics) * 100.0

    def get_summary(self) -> Dict[str, object]:
        return {
            "total_stages": len(self.stages),
            "completed_stages": sum(1 for s in self.stages.values() if s.status == StageStatus.COMPLETED),
            "overall_progress_pct": round(self.total_progress(), 2),
            "stages": [
                {
                    "stage": s.stage_number,
                    "title": s.title,
                    "status": s.status.value,
                    "progress_pct": round(s.progress_percentage, 1),
                    "topics_count": len(s.topics),
                }
                for s in self.stages.values()
            ],
        }
