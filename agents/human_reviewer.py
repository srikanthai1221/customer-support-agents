"""Human-in-the-loop: a support person reviews escalated replies."""
from autogen_agentchat.agents import UserProxyAgent


async def run_human_review(ticket_id: str, draft: str, decision: str,
                           final_text: str, reviewer: str) -> str:
    """Route the supervisor's decision from the UI through AutoGen's UserProxyAgent."""

    def ui_input(prompt: str) -> str:
        if decision == "Approve":
            return f"APPROVED by {reviewer}:\n{final_text}"
        return f"REJECTED by {reviewer}: {final_text}"

    human = UserProxyAgent(
        name="HumanSupport",
        description="A human support supervisor who approves, edits or rejects escalated replies.",
        input_func=ui_input,
    )
    result = await human.run(
        task=f"Ticket {ticket_id} was escalated. Draft reply:\n{draft}\n\nApprove, edit or reject it."
    )
    return result.messages[-1].content