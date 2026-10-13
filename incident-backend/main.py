from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
from uuid import uuid4

app = FastAPI(
    title="Cybersecurity Incident Integration API",
    description="Unified service for incidents, intelligence, AI analysis and playbooks",
    version="1.0.0"
)

# Allow the frontend to call this backend during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simulated playbooks for the hackathon demonstration.
PLAYBOOKS = [
    {
        "id": "PB001",
        "name": "Suspicious Login Investigation",
        "description": "Investigate unfamiliar login activity.",
        "actions": [
            "Review login history",
            "Verify user identity",
            "Check associated IP address",
            "Escalate if suspicious activity is confirmed"
        ]
    },
    {
        "id": "PB002",
        "name": "Malware Alert Investigation",
        "description": "Investigate a suspected malware detection.",
        "actions": [
            "Review endpoint evidence",
            "Identify affected assets",
            "Check related alerts",
            "Recommend containment if confirmed"
        ]
    },
    {
        "id": "PB003",
        "name": "Unusual Data Access Investigation",
        "description": "Investigate unexpected access to sensitive data.",
        "actions": [
            "Review access logs",
            "Identify affected resources",
            "Verify whether access was authorized",
            "Escalate confirmed suspicious activity"
        ]
    }
]

# Simulated incident data.
# Replace or supplement this data with your teammates' module outputs.
INCIDENTS = {
    "INC001": {
        "id": "INC001",
        "title": "Suspicious Login Attempt",
        "description": "A login was detected from an unfamiliar location.",
        "source": "Identity Intelligence",
        "asset": "Employee Account",
        "risk_score": 85,
        "severity": "High",
        "status": "Open",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence": [
            {
                "type": "Authentication",
                "description": "Login from an unfamiliar location",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "type": "Identity",
                "description": "Additional identity verification recommended",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ],
        "ai_analysis": {
            "risk_score": 85,
            "severity": "High",
            "reason": "Simulated risk assessment based on unfamiliar login activity."
        },
        "playbook_ids": ["PB001"]
    },
    "INC002": {
        "id": "INC002",
        "title": "Possible Malware Detection",
        "description": "A simulated endpoint alert requires investigation.",
        "source": "Endpoint Intelligence",
        "asset": "Workstation-07",
        "risk_score": 72,
        "severity": "High",
        "status": "Investigating",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence": [
            {
                "type": "Endpoint",
                "description": "Simulated suspicious endpoint activity",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ],
        "ai_analysis": {
            "risk_score": 72,
            "severity": "High",
            "reason": "Simulated risk assessment for a possible malware alert."
        },
        "playbook_ids": ["PB002"]
    },
    "INC003": {
        "id": "INC003",
        "title": "Unusual Data Access",
        "description": "Unusual access to a sensitive resource was flagged.",
        "source": "Data Intelligence",
        "asset": "Sensitive Document Store",
        "risk_score": 45,
        "severity": "Medium",
        "status": "Open",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence": [
            {
                "type": "Access Log",
                "description": "Simulated unusual resource access",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ],
        "ai_analysis": {
            "risk_score": 45,
            "severity": "Medium",
            "reason": "Simulated risk assessment for unusual data access."
        },
        "playbook_ids": ["PB003"]
    }
}


# Request model for incoming intelligence alerts.
class AlertInput(BaseModel):
    title: str
    description: str
    source: str = "External Intelligence"
    asset: str = "Unknown Asset"
    risk_score: int = 50
    evidence: list[dict] = []


def get_severity(score: int) -> str:
    if score >= 80:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 30:
        return "Medium"
    return "Low"


def get_incident_or_404(incident_id: str) -> dict:
    incident = INCIDENTS.get(incident_id)
    if not incident:
        raise HTTPException(
            status_code=404,
            detail=f"Incident {incident_id} was not found"
        )
    return incident


@app.get("/")
def home():
    return {
        "message": "Cybersecurity Incident Integration API is running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "incident-backend",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/incidents")
def get_incidents(
    severity: Optional[str] = None,
    status: Optional[str] = None
):
    results = list(INCIDENTS.values())

    if severity:
        results = [
            item for item in results
            if item["severity"].lower() == severity.lower()
        ]

    if status:
        results = [
            item for item in results
            if item["status"].lower() == status.lower()
        ]

    return {
        "count": len(results),
        "incidents": results
    }


@app.get("/incidents/{incident_id}")
def get_incident(incident_id: str):
    return get_incident_or_404(incident_id)


@app.get("/incidents/{incident_id}/investigation")
def get_investigation(incident_id: str):
    incident = get_incident_or_404(incident_id)

    return {
        "incident_id": incident["id"],
        "title": incident["title"],
        "status": incident["status"],
        "asset": incident["asset"],
        "evidence": incident["evidence"],
        "ai_analysis": incident["ai_analysis"],
        "recommended_playbooks": [
            playbook for playbook in PLAYBOOKS
            if playbook["id"] in incident["playbook_ids"]
        ]
    }


@app.get("/incidents/{incident_id}/playbooks")
def get_incident_playbooks(incident_id: str):
    incident = get_incident_or_404(incident_id)

    recommendations = [
        playbook for playbook in PLAYBOOKS
        if playbook["id"] in incident["playbook_ids"]
    ]

    return {
        "incident_id": incident_id,
        "recommendations": recommendations
    }


@app.get("/playbooks")
def get_playbooks():
    return {
        "count": len(PLAYBOOKS),
        "playbooks": PLAYBOOKS
    }


@app.post("/incidents")
def create_incident(alert: AlertInput):
    if not 0 <= alert.risk_score <= 100:
        raise HTTPException(
            status_code=400,
            detail="risk_score must be between 0 and 100"
        )

    incident_id = "INC-" + uuid4().hex[:8].upper()
    severity = get_severity(alert.risk_score)

    # Basic rule-based playbook association.
    text = (alert.title + " " + alert.description).lower()

    if any(word in text for word in ["login", "identity", "authentication"]):
        playbook_ids = ["PB001"]
    elif any(word in text for word in ["malware", "endpoint", "virus"]):
        playbook_ids = ["PB002"]
    elif any(word in text for word in ["data", "access", "document"]):
        playbook_ids = ["PB003"]
    else:
        playbook_ids = []

    incident = {
        "id": incident_id,
        "title": alert.title,
        "description": alert.description,
        "source": alert.source,
        "asset": alert.asset,
        "risk_score": alert.risk_score,
        "severity": severity,
        "status": "Open",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence": alert.evidence,
        "ai_analysis": {
            "risk_score": alert.risk_score,
            "severity": severity,
            "reason": "Rule-based demonstration assessment. Connect the team's AI module for actual analysis."
        },
        "playbook_ids": playbook_ids
    }

    INCIDENTS[incident_id] = incident

    return {
        "message": "Incident created successfully",
        "incident": incident
    }


@app.post("/incidents/{incident_id}/analyze")
def analyze_incident(incident_id: str):
    incident = get_incident_or_404(incident_id)

    return {
        "incident_id": incident_id,
        "analysis": incident["ai_analysis"],
        "note": "This is simulated analysis. Connect the actual AI module to replace it."
    }


@app.get("/summary")
def get_summary():
    incidents = list(INCIDENTS.values())

    return {
        "total_incidents": len(incidents),
        "open_incidents": sum(
            1 for item in incidents if item["status"] == "Open"
        ),
        "high_risk_incidents": sum(
            1 for item in incidents if item["risk_score"] >= 60
        ),
        "playbooks_available": len(PLAYBOOKS)
    }