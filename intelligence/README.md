# Alert Intelligence (Yasaswi)

Turns a pile of noisy security alerts into a short, ranked list of incidents, each with a score that explains itself.

## Pipeline

alerts.json -> enrich -> correlate -> risk score -> incidents.json

| Step | File | What it does |
|------|------|--------------|
| Generate | `generate_alerts.py` | Makes 36 fake alerts: account compromise (5), network attack (3), benign routine (3), background noise (25). Seed 42, so the data is the same every run. |
| Enrich | `enrich.py` | Adds asset criticality, owner, user role, privileged flag and threat-intel match using `data/lookups/*.csv`. Unknown host gets criticality 1 plus a note. |
| Correlate | `correlate.py` | Links alerts within 30 minutes that share a host, a user or an external IP. Internal IPs are ignored. Low-severity-only links are ignored, so noise does not merge into big incidents. |
| Score | `risk.py` | Transparent formula, see below. |
| Run all | `build_incidents.py` | Writes `data/processed/incidents.json` and `enriched_alerts.json`. |

## Risk score

score = 100 x (0.30 x asset + 0.30 x severity + 0.25 x threat_intel + 0.15 x alert_count)

- asset = (highest asset criticality - 1) / 4
- severity = (highest severity rank - 1) / 3 (low=1 ... critical=4)
- threat_intel = min(matching alerts, 3) / 3
- alert_count = min(alerts, 10) / 10

Labels: 75 and above critical, 50-74 high, 25-49 medium, below 25 low.
Each incident carries its four parts in `risk_factors`, so the dashboard can show why.

## Run it

    python intelligence/generate_alerts.py
    python -m intelligence.build_incidents
    python -m pytest -q

## Incident fields

incident_id, alert_ids, hosts, users, start, end, link_reasons, ti_reasons, risk_factors, priority_score, priority_label, scenario_tag

## Demo numbers

36 alerts -> 26 incidents (1 critical, 1 high, 2 medium, 22 low). INC-001 is the account compromise (score 92).
