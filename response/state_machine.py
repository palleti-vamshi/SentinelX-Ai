
# Response Safety Project
# Controls action states and prevents execution without approval.
# Execution is simulated only. No real system is changed.

from response.audit import append_event


# Allowed state changes
ALLOWED_TRANSITIONS = {
    "recommended": ["pending_approval"],
    "pending_approval": ["approved", "rejected"],
    "approved": ["executed_simulated"],
    "rejected": [],
    "executed_simulated": []
}


def transition(action, new_state, actor, reason):
    """Change an action state only when the transition is allowed."""

    old_state = action["state"]

    if not actor or not actor.strip():
        raise ValueError("Actor is required.")

    if new_state in ["approved", "rejected"]:
        if not reason or not reason.strip():
            raise ValueError("A reason is required.")

    if new_state == "executed_simulated":
        raise ValueError("Use execute_simulated() for execution.")

    if new_state not in ALLOWED_TRANSITIONS.get(old_state, []):
        raise ValueError(
            f"Invalid transition: {old_state} to {new_state}"
        )


    # Change state only after validating the request.
    action["state"] = new_state

    try:
        append_event(
            actor=actor,
            incident_id=action["incident_id"],
            action_id=action["action_id"],
            event_type=new_state,
            from_state=old_state,
            to_state=new_state,
            reason=reason
        )
    except Exception:
        # Restore the old state if audit recording fails.
        action["state"] = old_state
        raise

    return action



def execute_simulated(action):
    """Execute an approved action in simulation only."""

    old_state = action["state"]

    # Block execution when approval is missing.
    if old_state != "approved":
        append_event(
            actor="system",
            incident_id=action["incident_id"],
            action_id=action["action_id"],
            event_type="execution_blocked",
            from_state=old_state,
            to_state=old_state,
            reason="Valid approval required"
        )

        raise ValueError("Execution requires approval.")

    new_state = "executed_simulated"

    # Record the simulated execution before changing state.
    append_event(
        actor="system",
        incident_id=action["incident_id"],
        action_id=action["action_id"],
        event_type="execution_simulated",
        from_state=old_state,
        to_state=new_state,
        reason="Approved simulated execution"
    )

    action["state"] = new_state

    return f'{action["action_type"]} on {action["target"]} (simulated)'


    action["state"] = new_state

    return f'{action["action_type"]} on {action["target"]} (simulated)'
