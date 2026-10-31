"""Ticket log: one CSV row per ticket."""
import csv

import pandas as pd

from config import LOG_DIR, LOG_FILE

COLUMNS = [
    "ticket_id", "timestamp", "customer_name", "order_id", "complaint",
    "language", "category", "urgency", "sentiment", "summary",
    "kb_sources", "response", "confidence", "status", "escalation_reason",
    "orchestrator", "review_status", "reviewed_by", "final_response",
]


def load_log() -> pd.DataFrame:
    if not LOG_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(LOG_FILE, dtype=str).fillna("")


def next_ticket_id() -> str:
    df = load_log()
    if df.empty:
        return "TKT-0001"
    last = df["ticket_id"].str.replace("TKT-", "", regex=False).astype(int).max()
    return f"TKT-{last + 1:04d}"


def append_ticket(row: dict) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    is_new = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in COLUMNS})


def update_review(ticket_id: str, review_status: str, reviewed_by: str, final_response: str) -> None:
    df = load_log()
    mask = df["ticket_id"] == ticket_id
    df.loc[mask, "review_status"] = review_status
    df.loc[mask, "reviewed_by"] = reviewed_by
    df.loc[mask, "final_response"] = final_response
    df.to_csv(LOG_FILE, index=False)