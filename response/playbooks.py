
# Response Safety Project
# Playbooks define how the system responds to security incidents.
# These are recommendations only. No real systems are changed.

PLAYBOOKS = {
    "PB-PHISH": {
        "name": "Phishing Attack",
        "applies_to": ["phishing"],
        "steps": [
            "Review the reported email",
            "Check suspicious links and attachments",
            "Ask an analyst to review the evidence"
        ],
        "actions": [
            {
                "type": "quarantine_email",
                "target_type": "email",
                "disruptive": True
            }
        ]
    },

    "PB-BRUTE": {
        "name": "Brute Force Attack",
        "applies_to": ["brute_force", "credential_attack"],
        "steps": [
            "Review repeated failed login attempts",
            "Check the source IP address",
            "Ask an analyst to verify the activity"
        ],
        "actions": [
            {
                "type": "block_ip",
                "target_type": "ip",
                "disruptive": True
            }
        ]
    },

    "PB-ACCOUNT": {
        "name": "Account Compromise",
        "applies_to": ["account_compromise"],
        "steps": [
            "Review identity and login alerts",
            "Check for unusual account activity",
            "Ask an analyst to verify the evidence"
        ],
        "actions": [
            {
                "type": "reset_credentials",
                "target_type": "user",
                "disruptive": True
            }
        ]
    },

    "PB-MALWARE": {
        "name": "Malware Detection",
        "applies_to": ["malware", "suspicious_process"],
        "steps": [
            "Review the suspicious process alert",
            "Check the affected host",
            "Ask an analyst to review the evidence"
        ],
        "actions": [
            {
                "type": "isolate_host",
                "target_type": "host",
                "disruptive": True
            }
        ]
    },

    "PB-NETATTACK": {
        "name": "Network Attack",
        "applies_to": ["network_attack"],
        "steps": [
            "Review network security alerts",
            "Check the source and destination",
            "Ask an analyst to review the evidence"
        ],
        "actions": [
            {
                "type": "block_ip",
                "target_type": "ip",
                "disruptive": True
            }
        ]
    },

    "PB-BENIGN": {
        "name": "Benign Activity",
        "applies_to": ["benign"],
        "steps": [
            "Record the normal activity",
            "Continue monitoring"
        ],
        "actions": [
            {
                "type": "monitor",
                "target_type": "event",
                "disruptive": False
            }
        ]
    }
}


def get_playbook(playbook_id):
    """Return a playbook using its ID."""
    return PLAYBOOKS.get(playbook_id)


def list_playbooks():
    """Return all available playbooks."""
    return list(PLAYBOOKS.values())


if __name__ == "__main__":
    for playbook_id, playbook in PLAYBOOKS.items():
        print(playbook_id, "-", playbook["name"])
