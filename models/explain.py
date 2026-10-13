"""SentinelX AI — Phase 3: SHAP Feature Explainability.

Generates local and global feature attribution explanations for the fitted
Random Forest model using shap.TreeExplainer.
Attributes contributions directly to the attack-class probability, ensuring
mathematical additivity (base_value + sum(shap_values) == probability).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from models.interface import (
    ALL_INPUT_FEATURES,
    CATEGORICAL_FEATURES,
    DEFAULT_THRESHOLD,
    NUMERICAL_FEATURES,
    _extract_record_id,
    get_pipeline,
    validate_flow_record,
)
from models.train_rf import ARTIFACTS_DIR, MODEL_VERSION

GLOBAL_IMPORTANCE_PATH = ARTIFACTS_DIR / "global_feature_importance.json"

# Cached singleton TreeExplainer instance
_CACHED_EXPLAINER: shap.TreeExplainer | None = None

# Security-focused feature descriptions for SOC analysts
FEATURE_GLOSSARY: dict[str, str] = {
    "sttl": "Source Time-to-Live (TTL)",
    "dttl": "Destination Time-to-Live (TTL)",
    "ct_state_ttl": "Connections with identical state & TTL window",
    "ct_srv_dst": "Connections to same destination service",
    "ct_dst_src_ltm": "Connections between source & destination",
    "ct_srv_src": "Connections from same source service",
    "ct_dst_ltm": "Connections to same destination address",
    "ct_src_dport_ltm": "Connections from source to destination port",
    "ct_dst_sport_ltm": "Connections to destination from source port",
    "ct_src_ltm": "Connections from same source address",
    "sbytes": "Bytes sent (Source -> Destination)",
    "dbytes": "Bytes received (Destination -> Source)",
    "rate": "Flow packet transmission rate",
    "sload": "Source bits per second",
    "dload": "Destination bits per second",
    "dur": "Connection duration (seconds)",
    "spkts": "Source packet count",
    "dpkts": "Destination packet count",
    "smean": "Mean packet size (Source -> Destination)",
    "dmean": "Mean packet size (Destination -> Source)",
    "tcprtt": "TCP round-trip time",
    "synack": "TCP SYN-ACK handshake latency",
    "ackdat": "TCP ACK data latency",
    "sinpkt": "Source inter-packet arrival time",
    "dinpkt": "Destination inter-packet arrival time",
    "sjit": "Source packet jitter",
    "djit": "Destination packet jitter",
    "swin": "Source TCP window advertisement",
    "dwin": "Destination TCP window advertisement",
    "trans_depth": "HTTP transaction pipelining depth",
    "response_body_len": "HTTP response content length",
    "is_ftp_login": "FTP user session authenticated",
    "ct_ftp_cmd": "FTP command flow count",
    "ct_flw_http_mthd": "HTTP requests count in flow",
    "is_sm_ips_ports": "Source & destination IP/port identical",
    "state": "TCP/IP connection state flag",
    "proto": "Transport layer protocol",
    "service": "Application layer service",
}


def get_tree_explainer(pipeline: Pipeline | None = None) -> shap.TreeExplainer:
    """Retrieve or build the cached shap.TreeExplainer for the Random Forest classifier.

    Args:
        pipeline: Optional custom fitted Pipeline (defaults to project's cached pipeline).

    Returns:
        Fitted shap.TreeExplainer.
    """
    global _CACHED_EXPLAINER
    if _CACHED_EXPLAINER is None or pipeline is not None:
        active_pipeline = pipeline if pipeline is not None else get_pipeline()
        classifier = active_pipeline.named_steps["classifier"]
        explainer = shap.TreeExplainer(classifier)
        if pipeline is None:
            _CACHED_EXPLAINER = explainer
        return explainer
    return _CACHED_EXPLAINER


def extract_attack_shap_values(
    shap_output: np.ndarray | list[np.ndarray],
    row_index: int = 0,
) -> np.ndarray:
    """Extract 1D SHAP values corresponding strictly to the Attack class (Class 1).

    Handles variations across SHAP versions:
    - 3D array: shape (N, num_features, 2) -> output[:, :, 1]
    - List of 2 arrays: [class_0_values, class_1_values] -> list[1]
    - 2D array: single output
    """
    if isinstance(shap_output, list):
        # List of [class_0, class_1]
        attack_arr = shap_output[1]
        if attack_arr.ndim == 2:
            return attack_arr[row_index]
        return np.squeeze(attack_arr)

    if isinstance(shap_output, np.ndarray):
        if shap_output.ndim == 3:
            # Shape: (N, n_features, n_classes) -> extract class index 1 (attack)
            return shap_output[row_index, :, 1]
        if shap_output.ndim == 2:
            return shap_output[row_index]
        if shap_output.ndim == 1:
            return shap_output

    raise ValueError(f"Unexpected SHAP values structure: {type(shap_output)}")


def map_feature_metadata(
    encoded_feature_name: str,
    original_record: Mapping[str, Any],
) -> tuple[str, Any, str]:
    """Map one-hot encoded or raw feature name back to original feature name and value.

    Returns:
        tuple of (original_column_name, original_value, display_label)
    """
    # Check if this is a one-hot encoded column (e.g. proto_tcp, service_http, state_INT)
    for cat_col in CATEGORICAL_FEATURES:
        prefix = f"{cat_col}_"
        if encoded_feature_name.startswith(prefix):
            cat_val = encoded_feature_name[len(prefix):]
            orig_val = original_record.get(cat_col, cat_val)
            glossary_desc = FEATURE_GLOSSARY.get(cat_col, cat_col)
            display = f"{glossary_desc} ({cat_col}='{cat_val}')"
            return cat_col, orig_val, display

    # Numerical feature
    orig_val = original_record.get(encoded_feature_name, None)
    glossary_desc = FEATURE_GLOSSARY.get(encoded_feature_name, encoded_feature_name)
    display = f"{glossary_desc} ({encoded_feature_name}={orig_val})"
    return encoded_feature_name, orig_val, display


def explain(
    record: Mapping[str, Any] | pd.Series | pd.DataFrame,
    top_k: int = 8,
    threshold: float | None = None,
    pipeline: Pipeline | None = None,
) -> dict[str, Any]:
    """Generate SHAP local explanation for a single network flow record.

    Conforms to SentinelX AI Team Brief Part 5.3 contract and Phase 3 requirements.

    Args:
        record: Network flow record mapping.
        top_k: Number of top influential features to return (default: 8).
        threshold: Decision threshold (defaults to DEFAULT_THRESHOLD = 0.60).
        pipeline: Optional custom fitted Pipeline.

    Returns:
        Dictionary containing:
        - record_id
        - model_version
        - label / predicted_label ("attack" | "benign")
        - probability / predicted_probability
        - base_value (attack prior probability)
        - decision_threshold
        - attack_category (null)
        - top_features: list of top features with contribution and direction
        - summary: analyst-readable natural language justification
        - additivity_check_passed: bool indicating additivity validation
    """
    # Normalize input
    if isinstance(record, pd.DataFrame):
        if len(record) != 1:
            raise ValueError("explain() accepts a single record. For batch explanations, iterate over rows.")
        record_mapping = record.iloc[0].to_dict()
    elif isinstance(record, pd.Series):
        record_mapping = record.to_dict()
    elif isinstance(record, Mapping):
        record_mapping = record
    else:
        raise TypeError(
            f"record must be a Mapping, pandas Series, or single-row DataFrame, got '{type(record).__name__}'."
        )

    record_id = _extract_record_id(record_mapping)
    validated_record = validate_flow_record(record_mapping)

    active_pipeline = pipeline if pipeline is not None else get_pipeline()
    active_threshold = threshold if threshold is not None else DEFAULT_THRESHOLD

    preprocessor = active_pipeline.named_steps["preprocessor"]
    classifier = active_pipeline.named_steps["classifier"]

    # Transform single row
    row_df = pd.DataFrame([validated_record], columns=ALL_INPUT_FEATURES)
    X_transformed = preprocessor.transform(row_df)
    encoded_feature_names = preprocessor.get_feature_names_out().tolist()

    # Model prediction
    probabilities = classifier.predict_proba(X_transformed)[0]
    attack_probability = float(probabilities[1])
    predicted_label = "attack" if attack_probability >= active_threshold else "benign"

    # Compute SHAP values
    explainer = get_tree_explainer(pipeline=active_pipeline)
    raw_shap_output = explainer.shap_values(X_transformed)
    attack_shap_values = extract_attack_shap_values(raw_shap_output, row_index=0)

    # Base value for attack class (index 1)
    if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
        base_value = float(explainer.expected_value[1])
    else:
        base_value = float(explainer.expected_value)

    # Additivity verification: base_value + sum(shap_values) == attack_probability
    shap_sum = float(np.sum(attack_shap_values))
    reconstructed_prob = base_value + shap_sum
    additivity_passed = bool(np.isclose(reconstructed_prob, attack_probability, atol=1e-3))

    # Rank features by absolute SHAP attribution
    ranked_indices = np.argsort(np.abs(attack_shap_values))[::-1]

    top_features: list[dict[str, Any]] = []
    for rank, idx in enumerate(ranked_indices[:top_k], start=1):
        feat_name = encoded_feature_names[idx]
        shap_val = float(attack_shap_values[idx])
        orig_col, orig_val, display_label = map_feature_metadata(feat_name, record_mapping)

        direction = "attack" if shap_val > 0 else "benign"
        impact_desc = "pushed_toward_attack" if shap_val > 0 else "pushed_toward_benign"

        val_repr = f"'{orig_val}'" if isinstance(orig_val, str) else f"{orig_val}"
        sign_str = f"+{shap_val * 100:.2f}%" if shap_val > 0 else f"{shap_val * 100:.2f}%"
        direction_word = "increased" if shap_val > 0 else "decreased"
        explanation_phrase = f"{orig_col}={val_repr} {direction_word} attack probability by {sign_str}"

        top_features.append({
            "rank": rank,
            "feature": feat_name,
            "original_feature": orig_col,
            "value": orig_val,
            "display_name": display_label,
            "shap_value": round(shap_val, 4),
            "contribution": round(shap_val, 4),
            "direction": direction,
            "impact": impact_desc,
            "explanation": explanation_phrase,
        })

    # Build human-readable analyst narrative summary
    top_pushing_attack = [f for f in top_features if f["shap_value"] > 0]
    top_pushing_benign = [f for f in top_features if f["shap_value"] < 0]

    narrative_parts: list[str] = [
        f"Record {record_id} is classified as {predicted_label.upper()} "
        f"with attack probability {attack_probability * 100:.2f}% "
        f"(Decision threshold: {active_threshold:.2f}, Baseline prior: {base_value * 100:.2f}%)."
    ]

    if top_pushing_attack:
        attack_factors = ", ".join(
            f"{f['original_feature']}={f['value']!r} (+{f['shap_value'] * 100:.2f}%)"
            for f in top_pushing_attack[:3]
        )
        narrative_parts.append(f"Top indicators elevating attack risk: {attack_factors}.")

    if top_pushing_benign:
        benign_factors = ", ".join(
            f"{f['original_feature']}={f['value']!r} ({f['shap_value'] * 100:.2f}%)"
            for f in top_pushing_benign[:3]
        )
        narrative_parts.append(f"Top mitigating normal indicators: {benign_factors}.")

    summary_narrative = " ".join(narrative_parts)

    return {
        "record_id": record_id,
        "model_version": MODEL_VERSION,
        "label": predicted_label,
        "predicted_label": predicted_label,
        "probability": round(attack_probability, 4),
        "predicted_probability": round(attack_probability, 4),
        "base_value": round(base_value, 4),
        "decision_threshold": round(active_threshold, 2),
        "attack_category": None,
        "top_features": top_features,
        "summary": summary_narrative,
        "additivity_check_passed": additivity_passed,
    }


def compute_global_feature_importance(
    sample_size: int = 100,
    save_artifact: bool = True,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Compute global mean absolute SHAP feature importance on a representative sample."""
    from src.detection.prepare_data import load_dataset

    train_df = load_dataset("train").sample(n=sample_size, random_state=random_state)
    active_pipeline = get_pipeline()
    preprocessor = active_pipeline.named_steps["preprocessor"]

    X_transformed = preprocessor.transform(train_df[ALL_INPUT_FEATURES])
    encoded_feature_names = preprocessor.get_feature_names_out().tolist()

    explainer = get_tree_explainer(pipeline=active_pipeline)
    raw_shap_output = explainer.shap_values(X_transformed)

    attack_shap_matrix = (
        raw_shap_output[:, :, 1]
        if isinstance(raw_shap_output, np.ndarray) and raw_shap_output.ndim == 3
        else (raw_shap_output[1] if isinstance(raw_shap_output, list) else raw_shap_output)
    )

    mean_abs_shap = np.mean(np.abs(attack_shap_matrix), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1]

    results: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_indices[:25], start=1):
        feat_name = encoded_feature_names[idx]
        mean_val = float(mean_abs_shap[idx])
        glossary_desc = FEATURE_GLOSSARY.get(feat_name.split("_")[0], feat_name)

        results.append({
            "rank": rank,
            "feature": feat_name,
            "mean_abs_shap": round(mean_val, 4),
            "description": glossary_desc,
        })

    if save_artifact:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(GLOBAL_IMPORTANCE_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_version": MODEL_VERSION,
                    "sample_size": sample_size,
                    "global_top_features": results,
                },
                f,
                indent=2,
            )

    return results


def main() -> None:
    """CLI test demonstrating SHAP explanation on representative flow records."""
    from models.interface import get_example_record

    print("=" * 76)
    print("SentinelX AI — Phase 3: SHAP Feature Explainability Demo")
    print("=" * 76)

    # 1. Attack record explanation
    attack_sample = get_example_record(as_attack=True)
    exp_attack = explain(attack_sample, top_k=6)

    print("\n[1] Attack Record Explanation:")
    print(f"  Record ID:             {exp_attack['record_id']}")
    print(f"  Predicted Label:       {exp_attack['predicted_label']}")
    print(f"  Attack Probability:    {exp_attack['predicted_probability'] * 100:.2f}%")
    print(f"  Base Value (Prior):    {exp_attack['base_value'] * 100:.2f}%")
    print(f"  Decision Threshold:    {exp_attack['decision_threshold']:.2f}")
    print(f"  Additivity Validated:  {exp_attack['additivity_check_passed']}")
    print(f"  Summary: {exp_attack['summary']}\n")
    print("  Top Contributing Features:")
    for f in exp_attack["top_features"]:
        print(f"    #{f['rank']} {f['feature']:20s} | Value: {f['value']!r:8s} | SHAP: {f['shap_value']:+0.4f} | {f['impact']}")

    # 2. Benign record explanation
    benign_sample = get_example_record(as_attack=False)
    exp_benign = explain(benign_sample, top_k=6)

    print("\n[2] Benign Record Explanation:")
    print(f"  Record ID:             {exp_benign['record_id']}")
    print(f"  Predicted Label:       {exp_benign['predicted_label']}")
    print(f"  Attack Probability:    {exp_benign['predicted_probability'] * 100:.2f}%")
    print(f"  Base Value (Prior):    {exp_benign['base_value'] * 100:.2f}%")
    print(f"  Decision Threshold:    {exp_benign['decision_threshold']:.2f}")
    print(f"  Additivity Validated:  {exp_benign['additivity_check_passed']}")
    print(f"  Summary: {exp_benign['summary']}\n")
    print("  Top Contributing Features:")
    for f in exp_benign["top_features"]:
        print(f"    #{f['rank']} {f['feature']:20s} | Value: {f['value']!r:8s} | SHAP: {f['shap_value']:+0.4f} | {f['impact']}")

    print("\n" + "=" * 76)


if __name__ == "__main__":
    main()
