"""NimbusCart AI Support Desk: Streamlit UI."""
import asyncio
import html
import io
import json

import altair as alt
import pandas as pd
import streamlit as st
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import BaseChatMessage, ToolCallRequestEvent

from agents.human_reviewer import run_human_review
from config import APP_PASSWORD, COLLECTIONS, INDEX_DIR
from core.ticket_log import load_log, update_review
from rag.vector_store import build_all
from team import build_task, build_team, new_ticket

st.set_page_config(page_title="NimbusCart AI Support Desk", page_icon="🎧", layout="wide")

# Optional password gate: protects your OpenAI key once the app is public
if APP_PASSWORD and not st.session_state.get("authenticated"):
    st.title("NimbusCart AI Support Desk")
    password = st.text_input("Enter the app password", type="password")
    if password == APP_PASSWORD:
        st.session_state.authenticated = True
        st.rerun()
    elif password:
        st.error("Wrong password, please try again.")
    st.stop()

AGENT_STEPS = {
    "QueryAgent": "1 · Customer Query Agent: understanding the complaint",
    "KnowledgeAgent": "2 · Knowledge Base Agent: searching company documents",
    "ResolutionAgent": "3 · Resolution Agent: writing the reply",
    "LogAgent": "4 · Log Agent: logging and escalation decision",
}

SAMPLES = {
    "Type my own": "",
    "1 · Late order": "Where is my order? It was supposed to arrive yesterday.",
    "2 · Return shoes": "I want to return the shoes I bought last week. How do I get a refund?",
    "3 · Forgot password": "I forgot my password and can't log in.",
    "4 · Broken blender": "The blender I received is broken. Can I get a replacement?",
    "5 · Charged 3 times": "You charged my card three times! Fix this now or I'm contacting my bank and a lawyer.",
    "6 · Corporate gift cards": "Do you sell gift cards for a company of 500 employees with custom branding?",
    "7 · Tamil: password": "நான் என் கடவுச்சொல்லை மறந்துவிட்டேன், உள்நுழைய முடியவில்லை. என்ன செய்வது?",
    "8 · Hindi: UPI refund": "मेरा रिफंड अभी तक नहीं आया है। मैंने UPI से पेमेंट किया था।",
    "9 · Hacked account": "Someone placed two orders from my account that I did not place. I think my account was hacked!",
    "10 · Happy + warranty": "Loving my new headphones, thank you! Just wondering how long the warranty is.",
}


@st.cache_resource(show_spinner="Building FAISS indexes (first run only)...")
def ensure_indexes() -> bool:
    if not all((INDEX_DIR / name / "index.faiss").exists() for name in COLLECTIONS):
        build_all()
    return True


def load_sample() -> None:
    st.session_state.complaint = SAMPLES[st.session_state.sample]


def badge(status: str) -> str:
    colour = "#1e8e3e" if status == "Resolved" else "#d93025"
    return (f"<span style='background:{colour};color:white;padding:4px 14px;"
            f"border-radius:14px;font-weight:600;font-size:0.9rem'>{status}</span>")


def show_agent_output(source: str, text: str) -> None:
    if source == "QueryAgent":
        cleaned = text.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            st.json(json.loads(cleaned))
            return
        except ValueError:
            pass
        st.markdown(text.replace("\n", "  \n"))


async def run_ticket(ticket: dict, timeline) -> list:
    """Run the team and draw each agent's output the moment it arrives."""
    steps = []
    team, model_client = build_team(ticket)
    try:
        async for item in team.run_stream(task=build_task(ticket)):
            if isinstance(item, TaskResult):
                timeline.caption(f"Run finished: {item.stop_reason}")
            elif isinstance(item, ToolCallRequestEvent):
                for call in item.content:
                    timeline.caption(f"🔧 {item.source} called `{call.name}`")
            elif isinstance(item, BaseChatMessage) and item.source in AGENT_STEPS:
                text = item.to_text()
                steps.append((item.source, text))
                with timeline.expander(AGENT_STEPS[item.source], expanded=True):
                    show_agent_output(item.source, text)
    finally:
        await model_client.close()
    return steps


def show_result(row: dict | None) -> None:
    if not row:
        st.error("The Log Agent did not record this ticket. Run it with cli.py to see why.")
        return
    st.markdown(f"### Final reply &nbsp; {badge(row['status'])}", unsafe_allow_html=True)
    box = "#e6f4ea" if row["status"] == "Resolved" else "#fce8e6"
    st.markdown(
        f"<div style='background:{box};padding:18px;border-radius:10px;white-space:pre-wrap'>"
        f"{html.escape(str(row['response']))}</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Category", row["category"])
    c2.metric("Urgency", row["urgency"])
    c3.metric("Sentiment", row["sentiment"])
    c4.metric("Confidence", row["confidence"])
    st.caption(f"Sources: {row['kb_sources']} · Language: {row['language']} · "
               f"Orchestrator: {row['orchestrator']}")
    if row["status"] == "Escalated":
        st.error(f"Escalated to human support: {row['escalation_reason']}. "
                 "The draft is waiting in the Human review tab.")


ensure_indexes()

STATUS_COLOURS = {"Resolved": "#1e8e3e", "Escalated": "#d93025"}
URGENCY_COLOURS = {"Low": "#1e8e3e", "Medium": "#f9ab00", "High": "#d93025"}


def bar_chart(counts, colour="#1a73e8", colours=None, order=None):
    """Horizontal bar chart with the count printed at the end of each bar."""
    data = counts.rename_axis("label").reset_index(name="tickets")
    if order:
        data = (data.set_index("label").reindex(order, fill_value=0)
                .rename_axis("label").reset_index())
    if colours:
        colour_enc = alt.Color("label:N", legend=None,
                               scale=alt.Scale(domain=list(colours), range=list(colours.values())))
    else:
        colour_enc = alt.value(colour)
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order or "-x", title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("tickets:Q", title=None, axis=alt.Axis(tickMinStep=1, format="d", grid=False)),
    )
    bars = base.mark_bar(size=22, cornerRadiusEnd=4).encode(color=colour_enc,
                                                            tooltip=["label", "tickets"])
    labels = base.mark_text(align="left", dx=4, fontWeight="bold").encode(text="tickets:Q")
    return (bars + labels).properties(height=40 * len(data) + 30, width="container")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Orchestration", ["Round-robin (fixed order)", "Selector (model picks next agent)"])
    orchestrator = "selector" if mode.startswith("Selector") else "round_robin"
    st.selectbox("Load a test complaint", list(SAMPLES), key="sample", on_change=load_sample)
    if st.button("Rebuild knowledge base index"):
        with st.spinner("Re-embedding documents..."):
            build_all()
        st.success("Indexes rebuilt.")
    st.caption("Flow: Query → Knowledge → Resolution → Log")

st.title("🎧 NimbusCart AI Support Desk")
st.caption("Four AutoGen agents · FAISS knowledge base · automatic escalation to humans")

tab_new, tab_log, tab_dash, tab_review = st.tabs(
    ["New ticket", "Ticket log", "Dashboard", "Human review"]
)

# ---------------- Tab 1: New ticket ----------------
with tab_new:
    with st.form("ticket_form"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Customer name (optional)")
        order_id = c2.text_input("Order ID (optional)")
        complaint = st.text_area("Customer complaint", key="complaint", height=120)
        submitted = st.form_submit_button("Submit ticket", type="primary")

    if submitted and not complaint.strip():
        st.warning("Please type a complaint first.")
    elif submitted:
        ticket = new_ticket(complaint, name, order_id, orchestrator)
        st.subheader(f"Ticket {ticket['ticket_id']}")
        timeline = st.container(border=True)
        with st.spinner("The agents are working on it..."):
            steps = asyncio.run(run_ticket(ticket, timeline))
        st.session_state.last = {"id": ticket["ticket_id"], "steps": steps, "row": ticket["result"]}
        show_result(ticket["result"])
    elif "last" in st.session_state:
        last = st.session_state.last
        st.subheader(f"Last ticket: {last['id']}")
        for source, text in last["steps"]:
            with st.expander(AGENT_STEPS[source]):
                show_agent_output(source, text)
        show_result(last["row"])

# ---------------- Tab 2: Ticket log ----------------
with tab_log:
    df = load_log()
    if df.empty:
        st.info("No tickets yet. Submit one in the New ticket tab.")
    else:
        f1, f2 = st.columns(2)
        status_filter = f1.multiselect("Filter by status", sorted(df["status"].unique()))
        category_filter = f2.multiselect("Filter by category", sorted(df["category"].unique()))
        view = df
        if status_filter:
            view = view[view["status"].isin(status_filter)]
        if category_filter:
            view = view[view["category"].isin(category_filter)]

        def colour_status(value: str) -> str:
            if value == "Resolved":
                return "color:#1e8e3e;font-weight:600"
            if value == "Escalated":
                return "color:#d93025;font-weight:600"
            return ""

        st.dataframe(view.iloc[::-1].style.map(colour_status, subset=["status"]), hide_index=True)

        d1, d2 = st.columns(2)
        d1.download_button("Download CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           "tickets.csv", "text/csv")
        excel = io.BytesIO()
        df.to_excel(excel, index=False, engine="openpyxl")
        d2.download_button("Download Excel", excel.getvalue(), "tickets.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------- Tab 3: Dashboard ----------------
# ---------------- Tab 3: Dashboard ----------------
with tab_dash:
    df = load_log()
    if df.empty:
        st.info("The dashboard fills in once tickets are logged.")
    else:
        total = len(df)
        escalated = int((df["status"] == "Escalated").sum())
        urgency = df["urgency"].map({"Low": 1, "Medium": 2, "High": 3}).mean()
        urgency_label = "Low" if urgency < 1.5 else "Medium" if urgency < 2.5 else "High"
        confidence = pd.to_numeric(df["confidence"], errors="coerce").mean()

        m1, m2, m3, m4, m5 = st.columns(5)
        for col, label, value in [
            (m1, "Total tickets", total),
            (m2, "Resolved", total - escalated),
            (m3, "Escalated", f"{escalated} ({escalated / total:.0%})"),
            (m4, "Average urgency", f"{urgency:.1f} · {urgency_label}"),
            (m5, "Average confidence", f"{confidence:.2f}"),
        ]:
            with col.container(border=True):
                st.metric(label, value)

        c1, c2 = st.columns(2)
        with c1.container(border=True):
            st.markdown("**Tickets per category**")
            st.altair_chart(bar_chart(df["category"].value_counts()))
        with c2.container(border=True):
            st.markdown("**Resolved vs escalated**")
            st.altair_chart(bar_chart(df["status"].value_counts(), colours=STATUS_COLOURS))

        c3, c4 = st.columns(2)
        with c3.container(border=True):
            st.markdown("**Urgency mix**")
            st.altair_chart(bar_chart(df["urgency"].value_counts(), colours=URGENCY_COLOURS,
                                      order=["Low", "Medium", "High"]))
        c4 = c4.container(border=True)
        c4.markdown("**Top escalation reasons**")
        reasons = (
            df.loc[df["status"] == "Escalated", "escalation_reason"]
            .str.split(r" \+ ").explode()
            .str.replace(r"\(.*\)", "", regex=True).str.strip()
        )
        if reasons.empty:
            c4.caption("No escalations yet.")
        else:
            c4.altair_chart(bar_chart(reasons.value_counts(), colour="#e8710a"))

# ---------------- Tab 4: Human review ----------------
with tab_review:
    if "review_msg" in st.session_state:
        st.success(st.session_state.pop("review_msg"))
    df = load_log()
    df = load_log()
    pending = df[(df["status"] == "Escalated") & (df["review_status"] == "Pending review")]
    if pending.empty:
        st.success("No escalated tickets are waiting for review.")
    else:
        ticket_id = st.selectbox("Escalated ticket", pending["ticket_id"].tolist()[::-1])
        row = pending[pending["ticket_id"] == ticket_id].iloc[0]
        st.markdown(f"**Complaint:** {row['complaint']}")
        st.markdown(f"**Why it was escalated:** {row['escalation_reason']}")
        with st.form("review_form"):
            reviewer = st.text_input("Reviewer name", value="Support Lead")
            final_text = st.text_area("Reply to send (edit it if needed), or the reason for rejecting",
                                      value=row["response"], height=220)
            b1, b2 = st.columns(2)
            approve = b1.form_submit_button("Approve and send", type="primary")
            reject = b2.form_submit_button("Reject")
        if approve or reject:
            decision = "Approve" if approve else "Reject"
            agent_said = asyncio.run(
                run_human_review(ticket_id, row["response"], decision, final_text, reviewer)
            )
            update_review(ticket_id, "Approved" if approve else "Rejected", reviewer,
                          final_text if approve else "")
            st.session_state.review_msg = (
                f"{ticket_id} {'approved' if approve else 'rejected'} by {reviewer}. "
                f"HumanSupport agent: {agent_said[:150]}"
            )
            st.rerun()