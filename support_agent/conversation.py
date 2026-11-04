"""Run turns through the ADK runner and collect what happened.

Shared by main.py (a scripted conversation) and evaluate.py (one fresh
session per test case).
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import opik
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from opik import opik_context

from .agent import OPIK_PROJECT, root_agent

APP_NAME = "trailhead_support"


@dataclass
class TurnResult:
    reply: str = ""
    # Every tool call in order: {"name": ..., "args": {...}, "result": {...}}.
    # The arguments are what the policy and grounding metrics check. "result"
    # is the tool's response, or None if the tool never answered.
    tool_calls: list[dict] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    # Errors the model returned instead of an answer, such as a blocked
    # response or a malformed function call. They don't raise, so collect them.
    model_errors: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    # Where this turn sits in Opik, so evaluate.py can read its cost back.
    trace_id: str | None = None
    span_id: str | None = None

    @property
    def tools_called(self) -> list[str]:
        return [call["name"] for call in self.tool_calls]


def make_runner(agent: LlmAgent = root_agent) -> Runner:
    return Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=InMemorySessionService(),
    )


async def new_session(runner: Runner, user_id: str, session_id: str | None = None) -> str:
    session = await runner.session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    return session.id


@opik.track(
    name="customer_turn",
    project_name=OPIK_PROJECT,
    ignore_arguments=["runner", "user_id", "session_id", "as_thread"],
    capture_output=False,
    create_duplicate_root_span=False,
)
async def run_turn(
    runner: Runner, user_id: str, session_id: str, message: str, as_thread: bool = True
) -> TurnResult:
    """Send one user message and return the reply, tool calls and agents involved.

    The @opik.track wrapper makes the whole turn one Opik trace. Without it, a
    handoff from support_agent to refunds_agent shows up as two separate
    traces, because ADK runs the sub-agent after the first agent's run ends.
    """
    if as_thread:
        # Every turn in the same ADK session joins one Opik thread.
        opik_context.update_current_trace(thread_id=session_id, metadata={"user_id": user_id})
    result = TurnResult()
    trace, span = opik_context.get_current_trace_data(), opik_context.get_current_span_data()
    result.trace_id = trace.id if trace else None
    # Inside an evaluation, this turn is a span within the evaluation's trace.
    # On its own (main.py), it is the trace itself and there is no span.
    result.span_id = span.id if span else None

    started = time.perf_counter()
    content = types.Content(role="user", parts=[types.Part(text=message)])
    # Read the whole event stream. Breaking out early on the final response
    # leaves the trace unfinished in Opik.
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        if event.author and event.author != "user" and event.author not in result.agents:
            result.agents.append(event.author)
        for call in event.get_function_calls():
            result.tool_calls.append({"id": call.id, "name": call.name, "args": dict(call.args or {}), "result": None})
        for response in event.get_function_responses():
            for call in result.tool_calls:
                if call["id"] == response.id:
                    call["result"] = response.response
        if event.error_code:
            result.model_errors.append(f"{event.error_code}: {event.error_message or ''}".strip())
        if event.is_final_response() and event.content and event.content.parts:
            text_parts = [part.text for part in event.content.parts if part.text]
            if text_parts:
                result.reply = "".join(text_parts).strip()
    result.latency_s = time.perf_counter() - started

    opik_context.update_current_trace(
        output={
            "reply": result.reply,
            "agents": result.agents,
            "tool_calls": result.tool_calls,
            "model_errors": result.model_errors,
        }
    )
    return result


def check_credentials() -> list[str]:
    """Return what is missing before a run, so failures are clear up front."""
    missing = []
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        missing.append("GOOGLE_API_KEY (free key at https://aistudio.google.com/apikey)")
    # `opik configure` writes ~/.opik.config (or OPIK_CONFIG_PATH), which can
    # supply the workspace even when the API key comes from the environment.
    config_file = Path(os.getenv("OPIK_CONFIG_PATH", "~/.opik.config")).expanduser().exists()
    if os.getenv("OPIK_API_KEY") and not os.getenv("OPIK_WORKSPACE") and not config_file:
        missing.append("OPIK_WORKSPACE (your Opik workspace name, shown in the Opik URL)")
    has_opik = os.getenv("OPIK_API_KEY") or os.getenv("OPIK_URL_OVERRIDE") or config_file
    if not has_opik:
        missing.append(
            "Opik credentials: run `uv run opik configure`, or set OPIK_API_KEY and "
            "OPIK_WORKSPACE in .env (or OPIK_URL_OVERRIDE for self-hosted Opik)"
        )
    return missing
