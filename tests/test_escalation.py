from core.escalation import decide_escalation


def test_plain_shipping_question_is_resolved():
    status, reason = decide_escalation("Where is my order?", "Shipping", "Medium", "Neutral", True, 0.9)
    assert status == "Resolved" and reason == ""


def test_triple_charge_with_lawyer_is_escalated():
    complaint = "You charged my card three times! Fix this now or I'm contacting my bank and a lawyer."
    status, reason = decide_escalation(complaint, "Billing & Payments", "High", "Angry", True, 0.7)
    assert status == "Escalated"
    assert "High urgency" in reason and "legal threat" in reason and "payment dispute" in reason


def test_no_kb_match_is_escalated():
    status, reason = decide_escalation("Gift cards for 500 staff?", "Other", "Low", "Neutral", False, 0.2)
    assert status == "Escalated" and "No matching KB document" in reason


def test_low_confidence_is_escalated():
    status, reason = decide_escalation("Odd question", "Product", "Low", "Neutral", True, 0.4)
    assert status == "Escalated" and "Low confidence" in reason


def test_word_issue_is_not_a_legal_threat():
    status, _ = decide_escalation("I have an issue logging in", "Account & Login", "Medium", "Neutral", True, 0.8)
    assert status == "Resolved"