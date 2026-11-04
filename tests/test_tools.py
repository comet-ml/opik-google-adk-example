"""Offline tests for the tools. No API keys needed: `uv run pytest`."""

from support_agent import tools


def test_lookup_order_normalizes_id():
    assert tools.lookup_order(" #a1001 ")["found"] is True
    assert tools.lookup_order("Z9999")["found"] is False


def test_refund_inside_window_is_issued():
    result = tools.issue_refund("A1001", "wrong size")
    assert result["refunded"] is True
    assert result["refund_id"] == "R-A1001"


def test_refund_outside_window_is_refused_in_code():
    assert tools.check_refund_eligibility("A1003")["eligible"] is False
    assert tools.issue_refund("A1003", "changed mind")["refunded"] is False


def test_refund_before_delivery_is_refused():
    assert tools.issue_refund("A1002", "too slow")["refunded"] is False


def test_help_center_finds_policy():
    titles = [a["title"] for a in tools.search_help_center("returns refund policy")["articles"]]
    assert "Returns and refunds" in titles


def test_order_id_with_hash_and_space():
    assert tools.lookup_order("# a1001")["found"] is True


def test_escalate_without_order():
    assert tools.escalate_to_human("wants a callback")["ticket_id"] == "T-GENERAL"


def test_refund_on_last_day_of_window_is_issued():
    assert tools.issue_refund("A1004", "changed mind")["refunded"] is True


def test_second_refund_is_refused():
    eligibility = tools.check_refund_eligibility("A1005")
    assert eligibility["eligible"] is False
    assert "already refunded" in eligibility["reason"]
    assert tools.issue_refund("A1005", "again")["refunded"] is False


def test_tools_accept_non_string_arguments():
    # ADK passes the model's arguments through unchecked, so a number or null
    # must not crash the turn.
    assert tools.lookup_order(1001)["found"] is False
    assert tools.lookup_order(None)["found"] is False
    assert tools.check_refund_eligibility(None)["eligible"] is False
    assert tools.search_help_center(["refund"])["articles"] == tools.search_help_center("refund")["articles"]
