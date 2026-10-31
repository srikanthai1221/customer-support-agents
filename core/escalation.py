"""Escalation rule (Task 5). Code can only ADD escalations, never remove them."""
import re

from config import LOW_CONFIDENCE

RISK_PATTERNS = {
    "legal threat": r"\b(lawyer|legal action|sue|court|consumer forum|advocate)\b",
    "fraud": r"\b(fraud|scam|unauthori[sz]ed|hacked|did not place|didn't place)\b",
    "payment dispute": r"\b(chargeback|bank|dispute)\b|charged\b.{0,20}\b(twice|thrice|three times|two times|multiple times)",
}


def decide_escalation(
    complaint: str,
    category: str,
    urgency: str,
    sentiment: str,
    kb_found: bool,
    confidence,
    risk_flags: str = "",
    agent_escalate: bool = False,
    agent_reason: str = "",
) -> tuple[str, str]:
    """Return (status, reason). Status is 'Resolved' or 'Escalated'."""
    reasons = []

    if urgency.strip().lower() == "high":
        reasons.append("High urgency")
    if "angry" in sentiment.strip().lower():
        reasons.append("Angry customer")
    if not kb_found:
        reasons.append("No matching KB document")
    elif category.strip().lower() == "other":
        reasons.append("Outside documented policies")

    # Risk words in the complaint (English) + risk flags from the Query Agent (any language)
    text = complaint.lower()
    risks = [label for label, pattern in RISK_PATTERNS.items() if re.search(pattern, text)]
    for flag in re.split(r"[^a-z_]+", risk_flags.lower()):
        label = flag.replace("_", " ").strip()
        if label in RISK_PATTERNS and label not in risks:
            risks.append(label)
    if risks:
        reasons.append("Risk: " + ", ".join(risks))

    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.0
    if conf < LOW_CONFIDENCE:
        reasons.append(f"Low confidence ({conf:.2f})")

    # The agent may escalate for a reason the rules don't cover
    if agent_escalate and not reasons:
        reasons.append(f"Agent judgement: {agent_reason or 'complex case'}")

    status = "Escalated" if reasons else "Resolved"
    return status, " + ".join(reasons)