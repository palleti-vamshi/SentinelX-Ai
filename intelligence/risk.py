"""Transparent risk score for an incident. Owner: Yasaswi (Alert Intelligence).

score = 100 * (0.30*asset + 0.30*severity + 0.25*threat_intel + 0.15*count)
Every part is between 0 and 1, so anyone can see WHY a score is high or low.
"""
SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
WEIGHTS = {"asset": 0.30, "severity": 0.30, "threat_intel": 0.25, "alert_count": 0.15}


def label_for(score):
    if score >= 75:
        return "critical"
    if score >= 50:
        return "high"
    if score >= 25:
        return "medium"
    return "low"


def score_incident(alerts):
    """Return (score, label, components) for a list of enriched alerts."""
    max_crit = max(a["enrichment"]["asset_criticality"] for a in alerts)
    max_sev = max(SEVERITY_RANK[a["severity"]] for a in alerts)
    ti_hits = sum(1 for a in alerts if a["enrichment"]["ti_match"])
    parts = {
        "asset": (max_crit - 1) / 4,
        "severity": (max_sev - 1) / 3,
        "threat_intel": min(ti_hits, 3) / 3,
        "alert_count": min(len(alerts), 10) / 10,
    }
    score = round(100 * sum(WEIGHTS[k] * v for k, v in parts.items()))
    components = {k: round(v, 2) for k, v in parts.items()}
    return score, label_for(score), components
