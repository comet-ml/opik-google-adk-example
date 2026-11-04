"""Canned store data, so the example runs without a database.

Dates are stored as "days since delivery" so the refund policy gives the same
answer whenever you run it.
"""

STORE_NAME = "Trailhead Outfitters"

REFUND_WINDOW_DAYS = 30

ORDERS: dict[str, dict] = {
    "A1001": {
        "item": "Ridgeline hiking boots, size 10",
        "total_usd": 149.00,
        "status": "delivered",
        "days_since_delivery": 6,
    },
    "A1002": {
        "item": "Summit 2-person tent",
        "total_usd": 289.00,
        "status": "in transit",
        "carrier": "UPS",
        "estimated_delivery": "in 2 business days",
    },
    "A1003": {
        "item": "Basecamp trekking poles",
        "total_usd": 79.00,
        "status": "delivered",
        "days_since_delivery": 45,
    },
    # Delivered exactly 30 days ago: the last day a refund is allowed.
    "A1004": {
        "item": "Trailblazer running shoes, size 9",
        "total_usd": 119.00,
        "status": "delivered",
        "days_since_delivery": 30,
    },
    # Already refunded once. A second refund would pay the customer twice.
    "A1005": {
        "item": "Alpine down jacket, medium",
        "total_usd": 229.00,
        "status": "delivered",
        "days_since_delivery": 12,
        "refund_id": "R-A1005",
    },
}

HELP_ARTICLES: list[dict] = [
    {
        "title": "Returns and refunds",
        "body": (
            "Unworn items can be refunded within 30 days of delivery. "
            "Refunds go back to the original payment method within 5 business days."
        ),
    },
    {
        "title": "Shipping times",
        "body": (
            "Standard shipping takes 3 to 5 business days. "
            "Every order gets a tracking number by email once it ships."
        ),
    },
    {
        "title": "Damaged or defective items",
        "body": (
            "If an item arrives damaged or fails under normal use, we replace it "
            "at any time. A support specialist confirms the claim first."
        ),
    },
    {
        "title": "Sizing and exchanges",
        "body": (
            "Boots run true to size. Exchanges for a different size are free "
            "within 30 days of delivery."
        ),
    },
]
