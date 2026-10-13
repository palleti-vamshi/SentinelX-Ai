"""SentinelX AI — Global Configuration."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Ollama LLM Local Service Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT", "15"))

# Models Module Defaults
DEFAULT_MODEL_VERSION = "rf-v1"
DEFAULT_DECISION_THRESHOLD = 0.60
