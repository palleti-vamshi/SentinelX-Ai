
# Response Safety Project
# Stores audit events in a JSON Lines file.
# Events are appended; public functions do not edit or delete them.

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


AUDIT_FILE = Path(__file__).resolve().parent.parent / "audit_log.jsonl"


def append_event(
    actor,
    incident_id,
    action_id,
    event_type,
    from_state,
    to_state,
    reason
):
    """Append one audit event to the log."""

    event = {
        "event_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor": actor,
        "incident_id": incident_id,
        "action_id": action_id,
        "event_type": event_type,
        "from_state": from_state,
        "to_state": to_state,
        "reason": reason
    }

    with AUDIT_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")

    return event


def read_events():
    """Read all saved audit events."""

    if not AUDIT_FILE.exists():
        return []

    events = []

    with AUDIT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                events.append(json.loads(line))

    return events


def filter_events(action_id=None, actor=None, event_type=None):
    """Filter audit events without changing the log."""

    events = read_events()

    if action_id is not None:
        events = [
            event for event in events
            if event["action_id"] == action_id
        ]

    if actor is not None:
        events = [
            event for event in events
            if event["actor"] == actor
        ]

    if event_type is not None:
        events = [
            event for event in events
            if event["event_type"] == event_type
        ]

    return events


if __name__ == "__main__":
    print("Audit log file:", AUDIT_FILE)
    print("Existing events:", len(read_events()))

