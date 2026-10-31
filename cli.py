"""Run one complaint through the team in the terminal.

Usage:
    python cli.py "My order is late"
    python cli.py "My order is late" --selector
    python cli.py "I'm calling my lawyer!" --review
"""
import asyncio
import sys

from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.ui import Console

from core.ticket_log import update_review
from team import build_task, build_team, new_ticket


async def human_review(row: dict) -> None:
    """Human-in-the-loop in the terminal, using AutoGen's UserProxyAgent."""
    print("\n=== HUMAN REVIEW ===")
    print(f"Why escalated: {row['escalation_reason']}")
    print(f"Draft reply:\n{row['response']}\n")
    print("Type APPROVE to send the draft as it is, or type a corrected reply (one line).")

    human = UserProxyAgent("HumanSupport", input_func=input)
    result = await human.run(task="Review the escalated draft reply.")
    answer = result.messages[-1].content.strip()

    approved = answer.upper() == "APPROVE"
    update_review(
        row["ticket_id"],
        "Approved" if approved else "Edited",
        "CLI reviewer",
        row["response"] if approved else answer,
    )
    print(f"Review saved: {'Approved' if approved else 'Edited'}")


async def main(complaint: str, mode: str) -> None:
    ticket = new_ticket(complaint, customer_name="Test User", order_id="ORD-1001", orchestrator=mode)
    team, model_client = build_team(ticket)
    try:
        await Console(team.run_stream(task=build_task(ticket)))
    finally:
        await model_client.close()

    print("\n=== LOGGED ROW ===")
    print(ticket["result"] or "log_ticket was NOT called - check the LogAgent")

    row = ticket["result"]
    if "--review" in sys.argv and row and row["status"] == "Escalated":
        await human_review(row)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    complaint = args[0] if args else "Where is my order? It was supposed to arrive yesterday."
    mode = "selector" if "--selector" in sys.argv else "round_robin"
    asyncio.run(main(complaint, mode))