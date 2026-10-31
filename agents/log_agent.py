"""Log Agent: decides escalation and records the ticket."""
import re
from datetime import datetime

from autogen_agentchat.agents import AssistantAgent

from core.escalation import decide_escalation
from core.ticket_log import append_ticket
from rag.vector_store import format_hits, search

SYSTEM_MESSAGE = """You are the Log Agent of NimbusCart's support desk.
You record the ticket and decide whether a human must take over. You never talk to the customer.

Steps:
1. Call check_escalation_policy with a one-line description of this case.
2. Decide escalate = true ONLY if one of these applies: urgency is High, sentiment is Angry,
   SOURCES is NONE, there is a legal/fraud/payment risk flag, or confidence is below 0.6.
   Frustrated is NOT Angry and Medium is NOT High. Never escalate for any other reason.
3. Call log_ticket exactly ONCE. Copy the values from the conversation:
   category, urgency, sentiment, language, risk_flags and summary (summary_en) from QueryAgent,
   kb_sources from the KnowledgeAgent's SOURCES line,
   response (the text under REPLY, copied exactly) and confidence from ResolutionAgent.
4. When log_ticket has returned, answer with ONE line:
   Status: <Resolved or Escalated> | Reason: <reason or none> | TICKET_LOGGED
Never write TICKET_LOGGED before log_ticket has returned."""


def check_escalation_policy(situation: str) -> str:
    """Look up NimbusCart's escalation policy for a described situation."""
    return format_hits(search("escalation", situation, min_score=0.0))


def create_log_agent(model_client, ticket: dict) -> AssistantAgent:
    def log_ticket(
        category: str,
        urgency: str,
        sentiment: str,
        language: str,
        summary: str,
        kb_sources: str,
        response: str,
        confidence: float,
        risk_flags: str,
        escalate: bool,
        escalation_reason: str,
    ) -> str:
        """Append this ticket to the ticket log and return the final status. Call exactly once."""
        claimed = set(re.findall(r"[\w\-]+\.md", kb_sources))
        verified = sorted(claimed & ticket["retrieved_sources"])

        status, reason = decide_escalation(
            complaint=ticket["complaint"], category=category, urgency=urgency,
            sentiment=sentiment, kb_found=bool(verified), confidence=confidence,
            risk_flags=risk_flags, agent_escalate=escalate, agent_reason=escalation_reason,
        )
        row = {
            "ticket_id": ticket["ticket_id"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "customer_name": ticket["customer_name"],
            "order_id": ticket["order_id"],
            "complaint": ticket["complaint"],
            "language": language,
            "category": category,
            "urgency": urgency,
            "sentiment": sentiment,
            "summary": summary,
            "kb_sources": ", ".join(verified) or "None",
            "response": response,
            "confidence": confidence,
            "status": status,
            "escalation_reason": reason,
            "orchestrator": ticket["orchestrator"],
            "review_status": "Pending review" if status == "Escalated" else "Not required",
        }
        append_ticket(row)
        ticket["result"] = row
        return f"Logged {ticket['ticket_id']}: status={status}; reason={reason or 'none'}"

    return AssistantAgent(
        name="LogAgent",
        description="Goes last. Checks the escalation policy, records the ticket with log_ticket "
                    "and says TICKET_LOGGED.",
        model_client=model_client,
        tools=[check_escalation_policy, log_ticket],
        reflect_on_tool_use=True,
        max_tool_iterations=3,
        system_message=SYSTEM_MESSAGE,
    )