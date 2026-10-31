"""Knowledge Base Agent: retrieves policy passages with FAISS."""
from autogen_agentchat.agents import AssistantAgent

from rag.vector_store import format_hits, search

SYSTEM_MESSAGE = """You are the Knowledge Base Agent of NimbusCart's support desk.
Your ONLY job is retrieval. You never write the customer reply and never add facts of your own.

Steps:
1. Read the QueryAgent's JSON and call search_kb with its search_query.
2. If the result is NO_MATCH, you may call search_kb ONE more time with a rephrased query.
3. Keep only the passages that really help with this complaint. Copy them word for word.

Reply in exactly this format:
SOURCES: <comma-separated file names, or NONE>
PASSAGES:
- [file_name.md] <passage copied word for word>

If nothing relevant was found, reply exactly:
SOURCES: NONE
PASSAGES: No matching document found."""


def create_knowledge_agent(model_client, ticket: dict) -> AssistantAgent:
    def search_kb(query: str) -> str:
        """Search NimbusCart customer policies (shipping, refunds, account, products, billing).
        Returns passages with their source file names, or NO_MATCH."""
        hits = search("policies", query)
        ticket["retrieved_sources"].update(h["source"] for h in hits)
        return format_hits(hits)

    return AssistantAgent(
        name="KnowledgeAgent",
        description="Goes second. Searches the company policy documents with search_kb and "
                    "returns only the relevant passages with their file names.",
        model_client=model_client,
        tools=[search_kb],
        reflect_on_tool_use=True,
        max_tool_iterations=2,
        system_message=SYSTEM_MESSAGE,
    )