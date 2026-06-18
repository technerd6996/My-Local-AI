import math
import os
from services.embedder import embed, embed_batch

# ─────────────────────────────────────────────
# Labelled examples per tier.
# These teach the classifier what each tier
# looks like without hardcoding keywords.
# Add more examples here to improve accuracy.
# ─────────────────────────────────────────────

EXAMPLES = {
    "simple": [
        "Write a Python script to rename files in a directory",
        "What is a load balancer and how does it work",
        "Give me a bash script to check disk usage",
        "Summarize this text for me",
        "What are the benefits of containerization",
        "Write documentation for this function",
        "Brainstorm names for a tech startup",
        "What is the difference between TCP and UDP",
        "Write a regex to validate email addresses",
        "Create a PowerShell script to list running processes",
        "Explain what DNS is",
        "Write a Terraform resource block for an S3 bucket",
        "What is idempotency",
        "Convert this JSON to YAML",
        "Write unit tests for this function",
        "What does this error message mean",
        "Generate a sample Nginx config",
        "Explain what a container image is",
        "Write a cron job to run every 5 minutes",
        "What is the OSI model",
    ],
    "medium": [
        "Debug why my Kubernetes pod keeps crashing with OOMKilled",
        "Review my architecture for a high-availability web application",
        "Analyze the performance bottleneck in this database query",
        "Compare microservices vs monolithic architecture for my use case",
        "Walk me through debugging this memory leak step by step",
        "What are the trade-offs between eventual and strong consistency",
        "Review this code for security vulnerabilities",
        "Explain why my distributed system has split-brain issues",
        "Design a caching strategy for a high-traffic API",
        "Trace through this multi-threaded race condition",
        "Analyze why this CI pipeline is flaky",
        "What approach should I take to migrate this monolith to microservices",
        "Diagnose why my service has intermittent 503 errors",
        "Review my Terraform module for best practices",
        "Explain the trade-offs of different database sharding strategies",
        "How do I refactor this tightly coupled code",
        "Why does my API have high latency under load",
        "Debug this async race condition in my Node service",
        "Analyze the security posture of this cloud architecture",
        "Help me design an event-driven system for order processing",
    ],
    "hard": [
        "Create a product strategy for launching a B2B SaaS platform",
        "Analyze this entire codebase and suggest a complete refactoring plan",
        "Write a comprehensive business plan for a cloud services company",
        "Design the complete architecture for a global real-time messaging system",
        "What acquisition strategy should a startup use to compete with AWS",
        "Write a detailed technical specification for a distributed consensus system",
        "Plan a zero-downtime migration of a 10TB production database",
        "Create a comprehensive security audit framework for a fintech company",
        "Design an ML pipeline for real-time fraud detection at scale",
        "Write a detailed post-mortem report for a major production outage",
    ],
}

MODEL_MAP = {
    "simple": "qwen2.5:7b",
    "medium": "deepseek-r1:7b",
    "hard":   "deepseek-r1:7b",  # Phase 4 will escalate hard → cloud
}

# Cache precomputed example embeddings after startup
_example_embeddings: dict[str, list[list[float]]] | None = None


async def precompute_examples():
    """
    Called once at app startup. Embeds all labelled examples
    and caches them in memory. Subsequent calls are instant.
    """
    global _example_embeddings
    print("[CLASSIFIER] Precomputing example embeddings...")
    _example_embeddings = {}
    for tier, examples in EXAMPLES.items():
        _example_embeddings[tier] = await embed_batch(examples)
    print("[CLASSIFIER] Ready.")


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


async def classify(messages: list[dict]) -> tuple[str, str, float]:
    """
    Returns (tier, model_name, confidence_score).
    tier:       simple | medium | hard
    model_name: ollama model string
    confidence: 0.0–1.0 cosine similarity to closest example
    """
    if _example_embeddings is None:
        # Fallback if called before startup (shouldn't happen)
        return "simple", MODEL_MAP["simple"], 0.0

    # Extract the last user message
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            last_user_msg = content if isinstance(content, str) else ""
            break

    if not last_user_msg.strip():
        return "simple", MODEL_MAP["simple"], 0.0

    query_emb = await embed(last_user_msg)
    

    # For each tier, find the highest similarity to any example
    tier_scores: dict[str, float] = {}
    for tier, embs in _example_embeddings.items():
        best = max(_cosine_sim(query_emb, ex_emb) for ex_emb in embs)
        tier_scores[tier] = best

    best_tier = max(tier_scores, key=tier_scores.get)
    confidence = tier_scores[best_tier]

    return best_tier, MODEL_MAP[best_tier], confidence