"""Tools the support agents can call.

ADK turns each function into a tool, using its name, type hints and docstring
as the schema the model sees. Each one returns a dict so the result shows up
as structured input and output on its span in Opik.
"""

import re

from .data import HELP_ARTICLES, ORDERS, REFUND_WINDOW_DAYS


def normalize_order_id(order_id: str) -> str:
    """Customers type IDs as "#a1001" or " a1001 ". Store them as "A1001"."""
    # str() because the model sometimes sends a number (1001) or null, and
    # ADK passes tool arguments through without checking their types.
    return str(order_id or "").strip().upper().lstrip("#").strip()


def lookup_order(order_id: str) -> dict:
    """Look up an order by its ID, for example "A1001".

    Returns the item, total, shipping status and delivery details.
    """
    order = ORDERS.get(normalize_order_id(order_id))
    if order is None:
        return {"found": False, "order_id": str(order_id)}
    return {"found": True, "order_id": normalize_order_id(order_id), **order}


def search_help_center(query: str) -> dict:
    """Search the store's help center for policy and how-to articles.

    Use this for questions about shipping, returns, sizing or damaged items.
    """
    words = set(re.findall(r"[a-z]+", str(query).lower()))
    scored = []
    for article in HELP_ARTICLES:
        text = f"{article['title']} {article['body']}".lower()
        hits = sum(1 for word in words if len(word) > 3 and word in text)
        if hits:
            scored.append((hits, article))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return {"articles": [article for _, article in scored[:2]]}


def check_refund_eligibility(order_id: str) -> dict:
    """Check whether an order can be refunded under the 30-day policy."""
    order = lookup_order(order_id)
    if not order["found"]:
        return {"eligible": False, "reason": "order not found"}
    if order.get("refund_id"):
        return {"eligible": False, "reason": f"already refunded ({order['refund_id']})"}
    if order["status"] != "delivered":
        return {"eligible": False, "reason": "order has not been delivered yet"}
    days = order["days_since_delivery"]
    if days > REFUND_WINDOW_DAYS:
        return {
            "eligible": False,
            "reason": f"delivered {days} days ago, outside the {REFUND_WINDOW_DAYS}-day window",
        }
    return {"eligible": True, "amount_usd": order["total_usd"]}


def issue_refund(order_id: str, reason: str) -> dict:
    """Refund an order to the original payment method.

    Only call this after check_refund_eligibility says the order is eligible.
    """
    eligibility = check_refund_eligibility(order_id)
    if not eligibility["eligible"]:
        # The policy is enforced here, not only in the prompt.
        return {"refunded": False, "reason": eligibility["reason"]}
    order_id = normalize_order_id(order_id)
    return {
        "refunded": True,
        "refund_id": f"R-{order_id}",
        "amount_usd": eligibility["amount_usd"],
        "reason": reason,
    }


def escalate_to_human(summary: str, order_id: str = "") -> dict:
    """Hand the case to a human support specialist.

    Use this for damaged items, anything outside policy, or when the customer
    asks for a person.
    """
    return {
        "escalated": True,
        "ticket_id": f"T-{normalize_order_id(order_id) or 'GENERAL'}",
        "summary": summary,
        "response_time": "within 1 business day",
    }
