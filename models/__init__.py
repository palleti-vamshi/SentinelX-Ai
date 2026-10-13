"""SentinelX-Ai AI Models Module.

Responsible for network-flow attack detection, model training,
feature explanations (Phase 3), and LLM investigation summaries (Phase 4).
"""

from models.interface import explain, predict

__all__ = ["predict", "explain"]

