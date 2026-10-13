import json

from intelligence.build_incidents import build
from intelligence.correlate import correlate
from intelligence.risk import score_incident, label_for


def load():
    with open("data/synthetic/alerts.json") as f:
        return json.load(f)


def incidents_for(tag):
    _, incs = build(load())
    return [i for i in incs if i["scenario_tag"] == tag]


def test_compromise_is_one_critical_incident():
    incs = incidents_for("account_compromise")
    assert len(incs) == 1
    assert len(incs[0]["alert_ids"]) == 5
    assert incs[0]["priority_label"] in ("high", "critical")


def test_network_attack_is_one_high_incident():
    incs = incidents_for("network_attack")
    assert len(incs) == 1 and len(incs[0]["alert_ids"]) == 3
    assert incs[0]["priority_label"] in ("high", "critical")


def test_benign_stays_low():
    for inc in incidents_for("benign_routine"):
        assert inc["priority_label"] == "low"


def test_noise_never_high_or_critical():
    for inc in incidents_for("noise"):
        assert inc["priority_label"] in ("low", "medium")


def test_no_giant_incident():
    _, incs = build(load())
    assert max(len(i["alert_ids"]) for i in incs) <= 15


def test_every_alert_in_exactly_one_incident():
    alerts = load()
    _, incs = build(alerts)
    ids = [a for i in incs for a in i["alert_ids"]]
    assert sorted(ids) == sorted(a["alert_id"] for a in alerts)


def test_shared_external_ip_links_alerts():
    a = {"alert_id": "A-1", "timestamp": "2026-10-20T09:00:00Z", "severity": "high",
         "host": None, "user": None, "src_ip": "203.0.113.9", "dst_ip": None}
    b = dict(a, alert_id="A-2", timestamp="2026-10-20T09:10:00Z", host="X")
    assert len(correlate([a, b])) == 1


def test_far_apart_alerts_not_linked():
    a = {"alert_id": "A-1", "timestamp": "2026-10-20T09:00:00Z", "severity": "high",
         "host": "H", "user": None, "src_ip": None, "dst_ip": None}
    b = dict(a, alert_id="A-2", timestamp="2026-10-20T11:00:00Z")
    assert len(correlate([a, b])) == 2


def test_score_components_and_labels():
    al = [{"severity": "critical",
           "enrichment": {"asset_criticality": 5, "ti_match": True}}]
    score, label, parts = score_incident(al)
    assert parts["asset"] == 1.0 and parts["severity"] == 1.0
    assert label == label_for(score)
    assert label_for(24) == "low" and label_for(25) == "medium"
    assert label_for(50) == "high" and label_for(75) == "critical"
