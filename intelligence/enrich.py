"""Add extra context to each alert. Owner: Yasaswi (Alert Intelligence)."""
import csv


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_lookups(folder="data/lookups"):
    assets = {r["host"]: r for r in load_csv(f"{folder}/assets.csv")}
    users = {r["user"]: r for r in load_csv(f"{folder}/users.csv")}
    intel = {r["indicator"]: r for r in load_csv(f"{folder}/threat_intel.csv")}
    return assets, users, intel


def enrich_alert(alert, assets, users, intel):
    """Return a copy of the alert with an `enrichment` section added."""
    asset = assets.get(alert.get("host"))
    user = users.get(alert.get("user"))
    hits = [intel[ip] for ip in (alert.get("src_ip"), alert.get("dst_ip"))
            if ip in intel]
    notes = []
    if alert.get("host") and asset is None:
        notes.append("unknown host: assumed lowest criticality")
    if alert.get("user") and user is None:
        notes.append("unknown user")
    new = dict(alert)
    new["enrichment"] = {
        "asset_criticality": int(asset["criticality"]) if asset else 1,
        "asset_owner": asset["owner"] if asset else None,
        "asset_type": asset["asset_type"] if asset else None,
        "user_role": user["role"] if user else None,
        "user_department": user["department"] if user else None,
        "user_privileged": (user["privileged"] == "true") if user else False,
        "ti_match": bool(hits),
        "ti_reason": "; ".join(h["reason"] for h in hits) or None,
        "notes": notes,
    }
    return new


def enrich_all(alerts, folder="data/lookups"):
    assets, users, intel = load_lookups(folder)
    return [enrich_alert(a, assets, users, intel) for a in alerts]
