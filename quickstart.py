"""The smallest complete example: one Google ADK agent, traced in Opik.

    uv run quickstart.py

The rest of the repo builds on this: two agents, one trace per turn,
threads, and an evaluation. See README.md.
"""

import asyncio
import os

import opik
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types
from opik.integrations.adk import OpikTracer, track_adk_agent_recursive

load_dotenv()
for name in ("OPIK_API_KEY", "OPIK_WORKSPACE"):  # blank .env lines would hide `opik configure`
    if os.environ.get(name) == "":
        del os.environ[name]


def get_order_status(order_id: str) -> dict:
    """Look up the shipping status of an order, for example "A1002"."""
    return {"order_id": order_id, "status": "in transit", "carrier": "UPS"}


agent = LlmAgent(
    name="order_agent",
    model="gemini-3.8-flash",
    instruction="Answer questions about orders. Use get_order_status for any order ID.",
    tools=[get_order_status],
)

# The Opik part: one tracer, attached to the agent and everything under it.
track_adk_agent_recursive(agent, OpikTracer(project_name="google-adk-quickstart"))


async def main() -> None:
    runner = InMemoryRunner(agent=agent, app_name="quickstart")
    session = await runner.session_service.create_session(app_name="quickstart", user_id="user-1")
    message = types.Content(role="user", parts=[types.Part(text="Where is my order A1002?")])
    # Read the whole event stream; breaking early leaves the trace unfinished.
    async for event in runner.run_async(user_id="user-1", session_id=session.id, new_message=message):
        if event.is_final_response() and event.content and event.content.parts:
            print(event.content.parts[0].text)
    await runner.close()
    opik.flush_tracker()  # send buffered traces before the script exits


if __name__ == "__main__":
    asyncio.run(main())
