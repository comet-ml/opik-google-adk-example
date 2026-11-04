"""A two-agent customer-support team built with Google ADK and traced with Opik.

`support_agent` answers order and policy questions and hands refund requests
to `refunds_agent`. `build_agent` builds the team on one Gemini model and
attaches Opik: create a tracer, then attach it to the root agent, its
sub-agents and their tools. conversation.py adds one decorator so each
customer turn is a single trace.
"""

import os

from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.genai import types
from opik.integrations.adk import OpikTracer, track_adk_agent_recursive

from . import tools
from .data import STORE_NAME

# Gemini 3.8 Flash is the current GA Flash model. evaluate.py and compare.py
# can run the same agent on any other Gemini model with --model.
DEFAULT_MODEL = "gemini-3.8-flash"
MODEL = os.getenv("ADK_MODEL", DEFAULT_MODEL)
OPIK_PROJECT = os.getenv("OPIK_PROJECT_NAME", "google-adk-support-agent")

REFUNDS_INSTRUCTION = (
    f"You handle refunds for {STORE_NAME}. For every refund request, call "
    "check_refund_eligibility first. If the order is eligible, call "
    "issue_refund and give the customer the refund ID and amount. If it is "
    "not eligible, explain why in one sentence. If the item arrived damaged "
    "or the customer asks for a person, call escalate_to_human. When the "
    "customer moves on to a topic that isn't a refund, transfer back to "
    "support_agent."
)

SUPPORT_INSTRUCTION = (
    f"You are the support agent for {STORE_NAME}, an outdoor gear store. "
    "Use lookup_order for anything about a specific order and "
    "search_help_center for policy questions. Never guess order details or "
    "order IDs. Transfer refund requests to refunds_agent. If an item arrived "
    "damaged or broke in normal use, call escalate_to_human yourself, even if "
    "the customer asks for a refund: damaged items are replaced, not refunded. "
    "Keep answers to two or three sentences."
)


def build_agent(model: str = MODEL) -> LlmAgent:
    """Build the support team on one Gemini model, with Opik tracing attached."""
    # Retry rate limits and transient errors. Free Gemini keys have low
    # per-minute limits, and an evaluation run makes about a hundred calls.
    llm = Gemini(
        model=model,
        retry_options=types.HttpRetryOptions(attempts=5, initial_delay=2),
    )

    refunds_agent = LlmAgent(
        name="refunds_agent",
        model=llm,
        description="Handles refund requests: checks eligibility and issues refunds.",
        instruction=REFUNDS_INSTRUCTION,
        tools=[
            tools.lookup_order,
            tools.check_refund_eligibility,
            tools.issue_refund,
            tools.escalate_to_human,
        ],
    )

    support_agent = LlmAgent(
        name="support_agent",
        model=llm,
        description=f"Front-line customer support for {STORE_NAME}.",
        instruction=SUPPORT_INSTRUCTION,
        tools=[tools.lookup_order, tools.search_help_center, tools.escalate_to_human],
        sub_agents=[refunds_agent],
    )

    # Opik: one tracer for the whole agent tree. track_adk_agent_recursive
    # attaches its callbacks to support_agent, refunds_agent and every tool.
    opik_tracer = OpikTracer(
        project_name=OPIK_PROJECT,
        tags=["google-adk", "example"],
        metadata={"model": model},
    )
    track_adk_agent_recursive(support_agent, opik_tracer)
    return support_agent


# `adk web` and `adk run` look for a module-level `root_agent`.
root_agent = build_agent()
