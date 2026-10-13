"""Group related alerts into incidents. Owner: Yasaswi (Alert Intelligence).

Two alerts are linked if they are within WINDOW_MINUTES of each other and share
a host, a user or an external IP. Linked alerts form a group (union-find).
"""
from datetime import datetime, timedelta

WINDOW_MINUTES = 30
MAX_GROUP_SIZE = 15  # safety: a bigger group is split by time gaps


def parse_time(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def is_internal(ip):
    return ip is None or ip.startswith("10.") or ip.startswith("192.168.")


def shared_reasons(a, b):
    reasons = []
    if a.get("host") and a["host"] == b.get("host"):
        reasons.append(f"same host {a['host']}")
    if a.get("user") and a["user"] == b.get("user"):
        reasons.append(f"same user {a['user']}")
    for key in ("src_ip", "dst_ip"):
        ip = a.get(key)
        if ip and not is_internal(ip) and ip in (b.get("src_ip"), b.get("dst_ip")):
            reasons.append(f"same external IP {ip}")
    return reasons


def is_noise_link(a, b):
    """Two low-severity alerts sharing only a user are not worth linking."""
    return a["severity"] == "low" and b["severity"] == "low"


def correlate(alerts):
    """Return a list of groups; each group = {'alerts': [...], 'reasons': [...]}."""
    alerts = sorted(alerts, key=lambda a: a["timestamp"])
    parent = list(range(len(alerts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    reasons = {}
    window = timedelta(minutes=WINDOW_MINUTES)
    for i, a in enumerate(alerts):
        for j in range(i + 1, len(alerts)):
            b = alerts[j]
            if parse_time(b["timestamp"]) - parse_time(a["timestamp"]) > window:
                break
            why = shared_reasons(a, b)
            if why and not is_noise_link(a, b):
                parent[find(j)] = find(i)
                reasons.setdefault(i, set()).update(why)
                reasons.setdefault(j, set()).update(why)

    groups = {}
    for i in range(len(alerts)):
        groups.setdefault(find(i), []).append(i)
    result = []
    for members in groups.values():
        why = set()
        for i in members:
            why |= reasons.get(i, set())
        result.append({"alerts": [alerts[i] for i in members],
                       "reasons": sorted(why)})
    result.sort(key=lambda g: g["alerts"][0]["timestamp"])
    return result
