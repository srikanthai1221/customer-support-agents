"""Resolution Agent: writes the grounded customer reply."""
from autogen_agentchat.agents import AssistantAgent

from rag.vector_store import format_hits, search

SYSTEM_MESSAGE = """You are the Resolution Agent of NimbusCart's support desk.
You write the reply the customer will read.

Rules:
- Use ONLY facts from the KnowledgeAgent's PASSAGES. Never invent days, amounts, links,
  policies, discounts or promises.
- Call get_reply_guidelines ONCE with the category and follow its tone and structure.
  The guidelines tell you HOW to write, never WHAT the policy is.
- If SOURCES is NONE, do not guess. Apologise, say honestly that you cannot confirm this
  from our current policies, and say a specialist will contact them within 24 hours.
- Write in the customer's language (the "language" field from QueryAgent).
- Use the customer's name and order ID if they were given. Keep it under 150 words.
- Never mention agents, tools, scores or file names in the reply.

Then rate your confidence that the reply fully solves the problem using only the passages:
0.9 = the passages answer it directly, 0.6 = partly, 0.3 or lower = not covered.

Reply in exactly this format:
REPLY:
<the customer reply>
CONFIDENCE: <number between 0 and 1>
CONFIDENCE_REASON: <one short line>"""


def get_reply_guidelines(category: str) -> str:
    """Look up NimbusCart's reply style guide: tone, structure and tips for this category."""
    return format_hits(search("style", f"How to reply to a {category} complaint", min_score=0.0))


def create_resolution_agent(model_client) -> AssistantAgent:
    return AssistantAgent(
        name="ResolutionAgent",
        description="Goes third. Writes a polite customer reply using only the KnowledgeAgent's "
                    "passages, in the customer's language, with a confidence score.",
        model_client=model_client,
        tools=[get_reply_guidelines],
        reflect_on_tool_use=True,
        system_message=SYSTEM_MESSAGE,
    )