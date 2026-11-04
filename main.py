"""Run a three-turn support conversation and trace it in Opik.

    uv run main.py

Each turn becomes one trace in the `google-adk-support-agent` project, and
all three land in the same Opik thread because they share an ADK session.
"""

import asyncio
import sys
import uuid

import opik

from support_agent.agent import MODEL, OPIK_PROJECT
from support_agent.conversation import (
    check_credentials,
    make_runner,
    new_session,
    run_turn,
)

CONVERSATION = [
    "Hi, where is my order A1002?",
    "Also, the boots from order A1001 don't fit. Can I get my money back?",
    "One more: the trekking poles from A1003 snapped on the first hike.",
]


async def main() -> None:
    runner = make_runner()
    user_id = "demo-user"
    session_id = await new_session(runner, user_id, session_id=f"demo-{uuid.uuid4().hex[:8]}")
    print(f"Trailhead Outfitters support, running on {MODEL}")

    try:
        for text in CONVERSATION:
            print(f"\nCustomer: {text}")
            turn = await run_turn(runner, user_id, session_id, text)
            print(f"Agent ({turn.agents[-1] if turn.agents else '?'}): {turn.reply}")
            print(f"  agents: {' -> '.join(turn.agents)}   tools: {', '.join(turn.tools_called) or 'none'}")
            print(f"  trace: {turn.trace_id}")
    finally:
        await runner.close()
        # Send any traces still buffered before the script exits.
        opik.flush_tracker()
    print(
        f"\nDone. In Opik, open the '{OPIK_PROJECT}' project: Traces shows one "
        f"trace per turn, Threads shows the conversation as session {session_id}."
    )


if __name__ == "__main__":
    missing = check_credentials()
    if missing:
        print("Missing before this can run:\n  - " + "\n  - ".join(missing))
        print("Copy .env.example to .env and fill it in.")
        sys.exit(1)
    asyncio.run(main())
