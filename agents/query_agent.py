"""Customer Query Agent: understands the complaint."""
from autogen_agentchat.agents import AssistantAgent

from rag.vector_store import format_hits, search

SYSTEM_MESSAGE = """You are the Customer Query Agent of NimbusCart's support desk.
Your ONLY job is to understand the complaint. You never reply to the customer and never quote policies.

Steps:
1. Call lookup_triage_guide ONCE with the complaint text to get the official categories
   and the urgency, sentiment and risk rules.
2. Classify the complaint using ONLY those categories and rules.
3. Detect the language the customer wrote in.

Return ONLY this JSON, with no other text:
{
  "category": "Shipping | Refund & Returns | Account & Login | Product | Billing & Payments | Other",
  "urgency": "Low | Medium | High",
  "sentiment": "Positive | Neutral | Frustrated | Angry",
  "risk_flags": ["legal_threat", "fraud", "payment_dispute"],
  "language": "English",
  "summary_en": "one line in English, at most 25 words",
  "search_query": "English keywords that restate the customer's actual problem, e.g. 'order delayed late delivery not arrived'"
}
Use an empty list for risk_flags when there are none."""


def lookup_triage_guide(complaint: str) -> str:
    """Search the internal triage guide for categories and urgency, sentiment and risk rules."""
    return format_hits(search("triage", complaint, min_score=0.0))


def create_query_agent(model_client) -> AssistantAgent:
    return AssistantAgent(
        name="QueryAgent",
        description="Goes first. Reads the raw complaint and returns category, urgency, "
                    "sentiment, risk flags, language and a one-line summary as JSON.",
        model_client=model_client,
        tools=[lookup_triage_guide],
        reflect_on_tool_use=True,
        system_message=SYSTEM_MESSAGE,
    )