"""Run the whole pipeline: alerts -> enrich -> correlate -> score -> incidents.json
Owner: Yasaswi (Alert Intelligence)."""
import json
import os
from collections import Counter

from intelligence.correlate import correlate
from intelligence.enrich import enrich_all
from intelligence.risk import score_incident

IN_PATH = "data/synthetic/alerts.json"
OUT_DIR = "data/processed"


def scenario_tag(alerts):
    tags = Counter(a["details"]["scenario"] for a in alerts)
    return tags.most_common(1)[0][0]


def build(alerts):
    enriched = enrich_all(alerts)
    groups = correlate(enriched)
    incidents = []
    for number, group in enumerate(groups, start=1):
        al = group["alerts"]
        score, label, parts = score_incident(al)
        incidents.append({
            "incident_id": f"INC-{number:03d}",
            "alert_ids": [a["alert_id"] for a in al],
            "hosts": sorted({a["host"] for a in al if a["host"]}),
            "users": sorted({a["user"] for a in al if a["user"]}),
            "start": al[0]["timestamp"],
            "end": al[-1]["timestamp"],
            "link_reasons": group["reasons"],
            "ti_reasons": sorted({a["enrichment"]["ti_reason"] for a in al
                                  if a["enrichment"]["ti_reason"]}),
            "risk_factors": parts,
            "priority_score": score,
            "priority_label": label,
            "scenario_tag": scenario_tag(al),
        })
    incidents.sort(key=lambda i: -i["priority_score"])
    return enriched, incidents


def main():
    with open(IN_PATH) as f:
        alerts = json.load(f)
    enriched, incidents = build(alerts)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/enriched_alerts.json", "w") as f:
        json.dump(enriched, f, indent=2)
    with open(f"{OUT_DIR}/incidents.json", "w") as f:
        json.dump(incidents, f, indent=2)
    print(f"{len(alerts)} alerts in -> {len(incidents)} incidents out")
    print("By priority:", dict(Counter(i["priority_label"] for i in incidents)))
    print()
    print("Top incidents:")
    for i in incidents[:6]:
        print(f"  {i['incident_id']}  {i['priority_score']:>3} {i['priority_label']:<8} "
              f"{len(i['alert_ids'])} alerts  {i['scenario_tag']}")


if __name__ == "__main__":
    main()
