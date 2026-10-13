"""SentinelX AI - AI Models Interface.

Exposes the standardized prediction interface matching the shared architecture contract
in Part 5.3 and 5.6 of the SentinelX AI Team Brief.

Contract shape:
{
  "record_id": "R-100",
  "label": "attack" | "benign",
  "probability": float (0.0 to 1.0),
  "attack_category": null,
  "model_version": "rf-v1",
  "top_features": []
}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from models.train_rf import PIPELINE_PATH, load_pipeline

MODEL_VERSION = "rf-v1"
DEFAULT_THRESHOLD: float = 0.60

# Required input feature columns in expected order
CATEGORICAL_FEATURES: list[str] = ["proto", "service", "state"]
NUMERICAL_FEATURES: list[str] = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports",
]
ALL_INPUT_FEATURES: list[str] = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

_CACHED_PIPELINE: Pipeline | None = None


def get_pipeline(pipeline_path: Path = PIPELINE_PATH) -> Pipeline:
    """Retrieve or lazily load the trained detection pipeline."""
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is None:
        _CACHED_PIPELINE = load_pipeline(pipeline_path)
    return _CACHED_PIPELINE


def set_pipeline(pipeline: Pipeline | None) -> None:
    """Explicitly set or reset the cached pipeline (useful for testing)."""
    global _CACHED_PIPELINE
    _CACHED_PIPELINE = pipeline


def _extract_record_id(record: Mapping[str, Any]) -> str:
    """Extract a canonical record_id string from record inputs."""
    if "record_id" in record and record["record_id"] is not None:
        return str(record["record_id"])
    if "id" in record and record["id"] is not None:
        raw_id = record["id"]
        if isinstance(raw_id, (int, np.integer)) or (isinstance(raw_id, str) and raw_id.isdigit()):
            return f"R-{raw_id}"
        return str(raw_id)
    return "R-UNKNOWN"


def validate_flow_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that record contains all required network flow features.

    Raises:
        TypeError: If record is not a mapping/dict.
        ValueError: If required flow features are missing or non-convertible.
    """
    if not isinstance(record, Mapping):
        raise TypeError(
            f"record must be a dictionary or Mapping, got '{type(record).__name__}'."
        )

    # Check for missing features
    missing_features = [col for col in ALL_INPUT_FEATURES if col not in record]
    if missing_features:
        raise ValueError(
            f"Record is missing {len(missing_features)} required flow feature(s): {missing_features}. "
            "Note: The network flow model requires UNSW-NB15 flow features and cannot "
            "directly classify non-network alerts (such as cloud or host-only logs)."
        )

    cleaned: dict[str, Any] = {}

    # Validate categorical features
    for col in CATEGORICAL_FEATURES:
        val = record[col]
        cleaned[col] = str(val) if val is not None else "-"

    # Validate numerical features
    for col in NUMERICAL_FEATURES:
        val = record[col]
        try:
            cleaned[col] = float(val)
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"Incompatible value for numerical feature '{col}': expected numeric, got {val!r}"
            ) from err

    return cleaned


def predict(
    record: Mapping[str, Any] | pd.Series | pd.DataFrame,
    threshold: float | None = None,
    pipeline: Pipeline | None = None,
    include_explanations: bool = False,
) -> dict[str, Any]:
    """Predict attack probability and label for a validated network flow record.

    Conforms to Part 5.3 of the SentinelX AI contract:
    {
      "record_id": str,
      "label": "attack" | "benign",
      "probability": float (0.0 to 1.0),
      "attack_category": null,
      "model_version": "rf-v1",
      "top_features": []
    }

    Args:
        record: Mapping of flow feature names to values.
        threshold: Decision threshold for the attack class (defaults to DEFAULT_THRESHOLD = 0.60).
        pipeline: Optional custom fitted Pipeline (defaults to loaded cached artifact).

    Returns:
        Structured prediction dictionary.
    """
    # Normalize input if pandas Series or DataFrame
    if isinstance(record, pd.DataFrame):
        if len(record) != 1:
            raise ValueError("predict() accepts a single record. For batch, pass records individually.")
        record_mapping = record.iloc[0].to_dict()
    elif isinstance(record, pd.Series):
        record_mapping = record.to_dict()
    elif isinstance(record, Mapping):
        record_mapping = record
    else:
        raise TypeError(
            f"record must be a dictionary, pandas Series, or single-row DataFrame, got '{type(record).__name__}'."
        )

    record_id = _extract_record_id(record_mapping)
    validated_features = validate_flow_record(record_mapping)

    # Construct single-row DataFrame preserving exact column ordering
    row_df = pd.DataFrame([validated_features], columns=ALL_INPUT_FEATURES)

    # Use active or cached pipeline
    active_pipeline = pipeline if pipeline is not None else get_pipeline()
    active_threshold = threshold if threshold is not None else DEFAULT_THRESHOLD

    # Predict attack-class probability
    # Estimator classes: [0 (benign), 1 (attack)]
    probabilities = active_pipeline.predict_proba(row_df)[0]
    attack_probability = float(probabilities[1])

    predicted_label = "attack" if attack_probability >= active_threshold else "benign"

    top_features: list[dict[str, Any]] = []
    if include_explanations:
        from models.explain import explain as explain_fn
        explanation_result = explain_fn(
            record=record_mapping,
            threshold=active_threshold,
            pipeline=active_pipeline,
        )
        top_features = explanation_result["top_features"]

    return {
        "record_id": record_id,
        "label": predicted_label,
        "probability": round(attack_probability, 4),
        "attack_category": None,
        "model_version": MODEL_VERSION,
        "top_features": top_features,
    }


def explain(
    record: Mapping[str, Any] | pd.Series | pd.DataFrame,
    top_k: int = 8,
    threshold: float | None = None,
    pipeline: Pipeline | None = None,
) -> dict[str, Any]:
    """Explain model prediction for a network flow record using SHAP.

    Delegates to models.explain.explain().
    """
    from models.explain import explain as explain_fn
    return explain_fn(
        record=record,
        top_k=top_k,
        threshold=threshold,
        pipeline=pipeline,
    )


def summarize_incident(
    incident: Mapping[str, Any],
    use_cache: bool = True,
    force_template: bool = False,
) -> dict[str, Any]:
    """Generate structured investigation summary for a security incident.

    Delegates to models.llm.summarize_incident().
    """
    from models.llm import summarize_incident as summarize_fn
    return summarize_fn(
        incident=incident,
        use_cache=use_cache,
        force_template=force_template,
    )


def get_example_incident() -> dict[str, Any]:
    """Return a representative incident object matching Yasaswi's Part 5.2 schema."""
    from models.llm import get_example_incident as get_inc_fn
    return get_inc_fn()


def get_example_record(as_attack: bool = True) -> dict[str, Any]:
    """Return a representative network flow record for testing and collaboration."""
    if as_attack:
        return {
            "record_id": "R-100",
            "dur": 0.921987,
            "proto": "ospf",
            "service": "-",
            "state": "INT",
            "spkts": 20,
            "dpkts": 0,
            "sbytes": 1280,
            "dbytes": 0,
            "rate": 20.607666,
            "sttl": 254,
            "dttl": 0,
            "sload": 10551.125,
            "dload": 0.0,
            "sloss": 0,
            "dloss": 0,
            "sinpkt": 48.525633,
            "dinpkt": 0.0,
            "sjit": 52.253805,
            "djit": 0.0,
            "swin": 0,
            "stcpb": 0,
            "dtcpb": 0,
            "dwin": 0,
            "tcprtt": 0.0,
            "synack": 0.0,
            "ackdat": 0.0,
            "smean": 64,
            "dmean": 0,
            "trans_depth": 0,
            "response_body_len": 0,
            "ct_srv_src": 1,
            "ct_state_ttl": 2,
            "ct_dst_ltm": 1,
            "ct_src_dport_ltm": 1,
            "ct_dst_sport_ltm": 1,
            "ct_dst_src_ltm": 2,
            "is_ftp_login": 0,
            "ct_ftp_cmd": 0,
            "ct_flw_http_mthd": 0,
            "ct_src_ltm": 1,
            "ct_srv_dst": 1,
            "is_sm_ips_ports": 0,
        }
    return {
        "record_id": "R-200",
        "dur": 0.0,
        "proto": "arp",
        "service": "-",
        "state": "INT",
        "spkts": 1,
        "dpkts": 0,
        "sbytes": 46,
        "dbytes": 0,
        "rate": 0.0,
        "sttl": 0,
        "dttl": 0,
        "sload": 0.0,
        "dload": 0.0,
        "sloss": 0,
        "dloss": 0,
        "sinpkt": 60000.688,
        "dinpkt": 0.0,
        "sjit": 0.0,
        "djit": 0.0,
        "swin": 0,
        "stcpb": 0,
        "dtcpb": 0,
        "dwin": 0,
        "tcprtt": 0.0,
        "synack": 0.0,
        "ackdat": 0.0,
        "smean": 46,
        "dmean": 0,
        "trans_depth": 0,
        "response_body_len": 0,
        "ct_srv_src": 2,
        "ct_state_ttl": 2,
        "ct_dst_ltm": 2,
        "ct_src_dport_ltm": 2,
        "ct_dst_sport_ltm": 2,
        "ct_dst_src_ltm": 2,
        "is_ftp_login": 0,
        "ct_ftp_cmd": 0,
        "ct_flw_http_mthd": 0,
        "ct_src_ltm": 2,
        "ct_srv_dst": 2,
        "is_sm_ips_ports": 1,
    }
