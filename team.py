"""Wires the four agents into one AutoGen team."""
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agents.knowledge_agent import create_knowledge_agent
from agents.log_agent import create_log_agent
from agents.query_agent import create_query_agent
from agents.resolution_agent import create_resolution_agent
from config import CHAT_MODEL, OPENAI_API_KEY
from core.ticket_log import next_ticket_id

SELECTOR_PROMPT = """You are the dispatcher of a customer-support team. Choose who speaks next.

{roles}

Conversation so far:
{history}

Every ticket must pass through each specialist exactly once, in this order:
QueryAgent (classifies) -> KnowledgeAgent (finds policy passages)
-> ResolutionAgent (writes the reply) -> LogAgent (records the ticket).
Look at what has already been done and pick the agent whose step comes next.
Select one agent from {participants}. Return only the agent name."""


def new_ticket(complaint: str, customer_name: str = "", order_id: str = "",
               orchestrator: str = "round_robin") -> dict:
    """Shared state for one complaint, passed into the agents' tools."""
    return {
        "ticket_id": next_ticket_id(),
        "complaint": complaint.strip(),
        "customer_name": customer_name.strip(),
        "order_id": order_id.strip(),
        "orchestrator": orchestrator,
        "retrieved_sources": set(),
        "result": None,
    }


def build_task(ticket: dict) -> str:
    return (
        f"New support ticket {ticket['ticket_id']}\n"
        f"Customer name: {ticket['customer_name'] or 'Not given'}\n"
        f"Order ID: {ticket['order_id'] or 'Not given'}\n"
        f"Complaint:\n\"\"\"{ticket['complaint']}\"\"\""
    )


def build_team(ticket: dict):
    """Return (team, model_client). Close the client when the run is finished."""
    model_client = OpenAIChatCompletionClient(
        model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0.2
    )
    agents = [
        create_query_agent(model_client),
        create_knowledge_agent(model_client, ticket),
        create_resolution_agent(model_client),
        create_log_agent(model_client, ticket),
    ]
    termination = (
        TextMentionTermination("TICKET_LOGGED", sources=["LogAgent"])
        | MaxMessageTermination(12)
    )

    if ticket["orchestrator"] == "selector":
        team = SelectorGroupChat(
            agents,
            model_client=model_client,
            termination_condition=termination,
            selector_prompt=SELECTOR_PROMPT,
            allow_repeated_speaker=False,
        )
    else:
        team = RoundRobinGroupChat(agents, termination_condition=termination)

    return team, model_client