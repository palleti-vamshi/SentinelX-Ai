"""Generate fake security alerts. Owner: Yasaswi (Alert Intelligence)."""
import json
from datetime import datetime, timedelta

START = datetime(2026, 10, 20, 9, 0, 0)  # when our story begins
counter = 0  # used to give every alert a new ID


def make_alert(minutes, source, severity, rule,
               host=None, user=None, src_ip=None, dst_ip=None):
    """Build one alert, happening `minutes` after the start."""
    global counter
    counter += 1
    when = START + timedelta(minutes=minutes)
    return {
        "alert_id": f"A-{counter:04d}",
        "source": source,
        "timestamp": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "severity": severity,
        "host": host,
        "user": user,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "rule": rule,
        "details": {},
    }


def account_compromise():
    """Story: an attacker takes over sara.fin's account and steals data."""
    bad_ip = "203.0.113.9"  # this IP is in threat_intel.csv
    return [
        make_alert(0, "identity", "high", "Login from unusual country",
                   user="sara.fin", src_ip=bad_ip),
        make_alert(6, "identity", "medium", "Multiple failed MFA prompts",
                   user="sara.fin", src_ip=bad_ip),
        make_alert(18, "edr", "high", "Suspicious process accessed stored credentials",
                   host="SRV-FIN-01", user="sara.fin"),
        make_alert(31, "cloud", "high", "New access key created",
                   host="CLOUD-VM-01", user="sara.fin", src_ip=bad_ip),
        make_alert(47, "cloud", "critical", "Unusual large data download",
                   host="CLOUD-VM-01", user="sara.fin", src_ip=bad_ip),
    ]


def main():
    alerts = account_compromise()
    with open("data/synthetic/alerts.json", "w") as f:
        json.dump(alerts, f, indent=2)
    print(f"Wrote {len(alerts)} alerts to data/synthetic/alerts.json")


if __name__ == "__main__":
    main()
