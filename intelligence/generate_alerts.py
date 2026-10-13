"""Generate fake security alerts. Owner: Yasaswi (Alert Intelligence)."""
import json
import random
from collections import Counter
from datetime import datetime, timedelta

START = datetime(2026, 10, 20, 9, 0, 0)  # when our day begins
rng = random.Random(42)  # fixed seed: the same data every time we run


def make_alert(minutes, source, severity, rule, scenario,
               host=None, user=None, src_ip=None, dst_ip=None):
    """Build one alert, happening `minutes` after the start."""
    when = START + timedelta(minutes=minutes)
    return {
        "alert_id": None,  # filled in later, after sorting by time
        "source": source,
        "timestamp": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "severity": severity,
        "host": host,
        "user": user,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "rule": rule,
        "details": {"scenario": scenario},  # for our testing only
    }


def account_compromise():
    """Story: an attacker takes over sara.fin's account and steals data."""
    bad_ip = "203.0.113.9"  # this IP is in threat_intel.csv
    s = "account_compromise"
    return [
        make_alert(0, "identity", "high", "Login from unusual country", s,
                   user="sara.fin", src_ip=bad_ip),
        make_alert(6, "identity", "medium", "Multiple failed MFA prompts", s,
                   user="sara.fin", src_ip=bad_ip),
        make_alert(18, "edr", "high", "Suspicious process accessed stored credentials", s,
                   host="SRV-FIN-01", user="sara.fin"),
        make_alert(31, "cloud", "high", "New access key created", s,
                   host="CLOUD-VM-01", user="sara.fin", src_ip=bad_ip),
        make_alert(47, "cloud", "critical", "Unusual large data download", s,
                   host="CLOUD-VM-01", user="sara.fin", src_ip=bad_ip),
    ]


def network_attack():
    """Story: an outside attacker scans and exploits the web server."""
    bad_ip = "198.51.100.77"  # in threat_intel.csv
    s = "network_attack"
    return [
        make_alert(200, "siem", "medium", "Port scan from external IP", s,
                   host="SRV-WEB-01", src_ip=bad_ip, dst_ip="10.0.1.20"),
        make_alert(208, "siem", "high", "Web exploit attempt detected", s,
                   host="SRV-WEB-01", src_ip=bad_ip, dst_ip="10.0.1.20"),
        make_alert(220, "edr", "critical", "Unknown binary executed after exploit", s,
                   host="SRV-WEB-01", user="ravi.admin"),
    ]


def benign_routine():
    """Story: a developer mistypes a password, then logs in fine. Harmless."""
    office_ip = "10.0.0.33"
    s = "benign_routine"
    return [
        make_alert(120, "identity", "low", "Multiple failed logins", s,
                   host="WS-033", user="rkumar", src_ip=office_ip),
        make_alert(122, "identity", "low", "Successful login after failures", s,
                   host="WS-033", user="rkumar", src_ip=office_ip),
        make_alert(135, "edr", "low", "Software update installed", s,
                   host="WS-033", user="rkumar"),
    ]


NOISE_RULES = [
    ("siem", "Firewall rule hit (low risk)"),
    ("siem", "Port scan from internal scanner"),
    ("identity", "Password changed"),
    ("identity", "Single failed login"),
    ("identity", "Login from new browser"),
    ("edr", "USB device connected"),
    ("edr", "Software update installed"),
    ("cloud", "Storage bucket listed"),
]
NOISE_USERS = ["jdoe", "asmith", "mpatel"]
NOISE_HOSTS = ["WS-014", "WS-021", "WS-040", "SRV-HR-01", "CLOUD-VM-02"]


def background_noise(count=25):
    """Unrelated, low-importance alerts spread over the working day."""
    alerts = []
    for _ in range(count):
        source, rule = rng.choice(NOISE_RULES)
        severity = rng.choice(["low", "low", "low", "medium"])
        alerts.append(make_alert(
            rng.randint(0, 480), source, severity, rule, "noise",
            host=rng.choice(NOISE_HOSTS),
            user=rng.choice(NOISE_USERS),
            src_ip=f"10.0.0.{rng.randint(100, 199)}"))
    return alerts


def main():
    alerts = (account_compromise() + network_attack()
              + benign_routine() + background_noise())
    alerts.sort(key=lambda a: a["timestamp"])  # earliest first
    for number, alert in enumerate(alerts, start=1):
        alert["alert_id"] = f"A-{number:04d}"
    with open("data/synthetic/alerts.json", "w") as f:
        json.dump(alerts, f, indent=2)
    print(f"Wrote {len(alerts)} alerts to data/synthetic/alerts.json")
    counts = Counter(a["details"]["scenario"] for a in alerts)
    for name, number in counts.items():
        print(f"  {name}: {number}")


if __name__ == "__main__":
    main()
