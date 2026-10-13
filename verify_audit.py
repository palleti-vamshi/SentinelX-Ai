
from response.audit import read_events

events = read_events()

print("AUDIT VERIFICATION REPORT")
print("=" * 35)
print("Total recorded events:", len(events))

for event in events:
    
    print(
        event.get("event_type", "unknown"),
        "| Action:", event.get("action_id", "unknown"),
        "| Actor:", event.get("actor", "unknown"),
        "| State:",
        event.get("from_state", "unknown"),
        "->",
        event.get("to_state", "unknown")
    )


print("=" * 35)

if events:
    print("PASS: Audit log contains recorded events.")
else:
    print("WARNING: No audit events found.")
