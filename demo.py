
from response.playbooks import PLAYBOOKS
from response.state_machine import transition, execute_simulated
from response.audit import read_events

def run_demo():
    print("\nRESPONSE SAFETY DEMO")
    print("=" * 40)

    # CASE 1: Benign activity
    print("\nCASE 1: BENIGN ACTIVITY")
    benign = PLAYBOOKS["PB-BENIGN"]
    print("Playbook:", benign["name"])
    print("Recommended action:", benign["actions"][0]["type"])
    print("Result: Monitor only; no disruptive action.")

    # CASE 2: Suspicious activity with approval
    print("\nCASE 2: SUSPICIOUS ACTIVITY")
    action = {
        "action_id": "DEMO-ACT-001",
        "incident_id": "DEMO-INC-001",
        "action_type": "isolate_host",
        "target": "DEMO-HOST",
        "state": "recommended"
    }

    transition(
        action, "pending_approval",
        "demo_analyst", "Suspicious activity needs review"
    )
    print("State:", action["state"])

    transition(
        action, "approved",
        "demo_analyst", "Demo approval after review"
    )
    print("State:", action["state"])

    result = execute_simulated(action)
    print("Result:", result)
    print("Final state:", action["state"])

    # CASE 3: Attempt execution without approval
    print("\nCASE 3: UNSAFE EXECUTION ATTEMPT")
    unsafe_action = {
        "action_id": "DEMO-ACT-002",
        "incident_id": "DEMO-INC-002",
        "action_type": "block_ip",
        "target": "192.0.2.10",
        "state": "pending_approval"
    }

    try:
        execute_simulated(unsafe_action)
    except ValueError as error:
        print("BLOCKED:", error)
        print("State remains:", unsafe_action["state"])

    # Display audit evidence
    print("\nAUDIT EVIDENCE")
    events = read_events()
    print("Total audit events:", len(events))

    
    for event in events[-10:]:
        print(
            event.get("event_type", "unknown"),
            "|", event.get("action_id", "unknown"),
            "|", event.get("from_state", "unknown"),
            "->", event.get("to_state", "unknown")
        )


    print("\nDemo completed.")

if __name__ == "__main__":
    run_demo()
