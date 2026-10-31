"""Run one complaint through the team in the terminal.

Usage:
    python cli.py "My order is late"
    python cli.py "My order is late" --selector
"""
import asyncio
import sys

from autogen_agentchat.ui import Console

from team import build_task, build_team, new_ticket


async def main(complaint: str, mode: str) -> None:
    ticket = new_ticket(complaint, customer_name="Test User", order_id="ORD-1001", orchestrator=mode)
    team, model_client = build_team(ticket)
    try:
        await Console(team.run_stream(task=build_task(ticket)))
    finally:
        await model_client.close()

    print("\n=== LOGGED ROW ===")
    print(ticket["result"] or "log_ticket was NOT called - check the LogAgent")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    complaint = args[0] if args else "Where is my order? It was supposed to arrive yesterday."
    mode = "selector" if "--selector" in sys.argv else "round_robin"
    asyncio.run(main(complaint, mode))