"""Offline test for run_turn: a fake runner replays a scripted ADK event stream."""

import asyncio
from types import SimpleNamespace

from support_agent.conversation import run_turn


class FakeEvent:
    """Just the parts of google.adk.events.Event that run_turn reads."""

    def __init__(self, author, calls=(), responses=(), text=None, final=False, error_code=None):
        self.author = author
        self.calls, self.responses = list(calls), list(responses)
        self.content = SimpleNamespace(parts=[SimpleNamespace(text=text)]) if text else None
        self.final = final
        self.error_code, self.error_message = error_code, "scripted error" if error_code else None

    def get_function_calls(self):
        return self.calls

    def get_function_responses(self):
        return self.responses

    def is_final_response(self):
        return self.final


class FakeRunner:
    def __init__(self, events):
        self.events = events

    async def run_async(self, user_id, session_id, new_message):
        for event in self.events:
            yield event


def fn_call(id, name, **args):
    return SimpleNamespace(id=id, name=name, args=args)


def test_run_turn_collects_a_handoff_with_tool_arguments_and_results():
    events = [
        FakeEvent("support_agent", calls=[fn_call("c1", "transfer_to_agent", agent_name="refunds_agent")]),
        FakeEvent("refunds_agent", calls=[fn_call("c2", "issue_refund", order_id="A1001", reason="fit")]),
        FakeEvent("refunds_agent", responses=[SimpleNamespace(id="c2", response={"refunded": True})]),
        FakeEvent("refunds_agent", text="Refunded $149.00 (R-A1001).", final=True),
    ]
    turn = asyncio.run(run_turn(FakeRunner(events), "u", "s", "refund A1001", as_thread=False))
    assert turn.agents == ["support_agent", "refunds_agent"]
    assert turn.tools_called == ["transfer_to_agent", "issue_refund"]
    assert turn.tool_calls[1]["args"] == {"order_id": "A1001", "reason": "fit"}
    assert turn.tool_calls[1]["result"] == {"refunded": True}
    assert turn.tool_calls[0]["result"] is None  # no response was scripted for it
    assert turn.reply == "Refunded $149.00 (R-A1001)."
    assert turn.model_errors == []


def test_run_turn_records_model_errors_instead_of_hiding_them():
    events = [FakeEvent("support_agent", error_code="MALFORMED_FUNCTION_CALL", final=True)]
    turn = asyncio.run(run_turn(FakeRunner(events), "u", "s", "hi", as_thread=False))
    assert turn.reply == ""
    assert turn.model_errors == ["MALFORMED_FUNCTION_CALL: scripted error"]
