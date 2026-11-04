"""Offline tests for the evaluation metrics, the test set and the credential check."""

from types import SimpleNamespace

from opik.evaluation.models import OpikBaseModel

from eval_cases import TEST_CASES
from eval_metrics import (
    CorrectHandoff,
    ExpectedToolsCalled,
    GroundedOrderIds,
    JudgedAnswerRelevance,
    PolicySafeRefunds,
    order_ids_in,
)
from evaluate import agent_cost
from support_agent.agent import root_agent
from support_agent.conversation import check_credentials


def call(name: str, result: dict | None = None, **args) -> dict:
    return {"name": name, "args": args, "result": result}


def test_expected_tools_scores_the_share_called():
    metric = ExpectedToolsCalled()
    expected = ["check_refund_eligibility", "issue_refund"]
    assert metric.score(tool_calls=[call("check_refund_eligibility"), call("issue_refund")], expected_tools=expected).value == 1.0
    assert metric.score(tool_calls=[call("check_refund_eligibility")], expected_tools=expected).value == 0.5
    assert metric.score(tool_calls=[], expected_tools=["lookup_order"]).value == 0.0
    assert metric.score(tool_calls=[], expected_tools=[]).value == 1.0


def test_correct_handoff():
    metric = CorrectHandoff()
    assert metric.score(agents=["support_agent", "refunds_agent"], expected_agent="refunds_agent").value == 1.0
    assert metric.score(agents=["support_agent"], expected_agent="refunds_agent").value == 0.0
    assert metric.score(agents=["support_agent", "refunds_agent"], expected_agent="support_agent").value == 0.0


def test_policy_safe_refunds_allows_eligible_refunds():
    metric = PolicySafeRefunds()
    assert metric.score(tool_calls=[]).value == 1.0
    assert metric.score(tool_calls=[call("issue_refund", order_id="A1001", reason="fit")]).value == 1.0
    # Day 30 of a 30-day window is still inside it.
    assert metric.score(tool_calls=[call("issue_refund", order_id=" #a1004 ", reason="fit")]).value == 1.0


def test_policy_safe_refunds_flags_forbidden_refunds():
    metric = PolicySafeRefunds()
    for order_id in ["A1002", "A1003", "A1005", "Z9999"]:  # in transit, too old, refunded, unknown
        result = metric.score(tool_calls=[call("check_refund_eligibility", order_id=order_id),
                                          call("issue_refund", order_id=order_id, reason="please")])
        assert result.value == 0.0, order_id
        assert order_id in result.reason
    # One bad refund among good ones still fails the case.
    mixed = [call("issue_refund", order_id="A1001", reason="fit"), call("issue_refund", order_id="A1003", reason="x")]
    assert metric.score(tool_calls=mixed).value == 0.0


def test_policy_safe_refunds_flags_a_second_refund_of_the_same_order():
    done = {"refunded": True}
    twice = [call("issue_refund", done, order_id="A1001", reason="fit"), call("issue_refund", done, order_id="a1001", reason="fit")]
    result = PolicySafeRefunds().score(tool_calls=twice)
    assert result.value == 0.0
    assert "refunded twice" in result.reason


def test_policy_safe_refunds_allows_a_retry_after_a_rejected_call():
    # ADK rejected the first call (no reason given) and the model retried.
    rejected = call("issue_refund", {"error": "missing mandatory parameter: reason"}, order_id="A1001")
    retried = call("issue_refund", {"refunded": True}, order_id="A1001", reason="fit")
    assert PolicySafeRefunds().score(tool_calls=[rejected, retried]).value == 1.0


class NeverCalledJudge(OpikBaseModel):
    def __init__(self):
        super().__init__(model_name="never-called")

    def generate_string(self, *args, **kwargs):
        raise AssertionError("the judge should not run for a crashed case")

    def generate_provider_response(self, *args, **kwargs):
        raise AssertionError("the judge should not run for a crashed case")


def test_metrics_do_not_score_a_crashed_run():
    error = "ClientError: 429 RESOURCE_EXHAUSTED"
    results = [
        ExpectedToolsCalled().score(tool_calls=[], expected_tools=[], error=error),
        CorrectHandoff().score(agents=[], expected_agent="support_agent", error=error),
        PolicySafeRefunds().score(tool_calls=[], error=error),
        GroundedOrderIds().score(input="hi", tool_calls=[], error=error),
        # The judge is never called: this model object would fail if it were.
        JudgedAnswerRelevance(model=NeverCalledJudge(), require_context=False).score(input="hi", output="", error=error),
    ]
    assert all(result.scoring_failed for result in results)


def test_policy_safe_refunds_respects_cases_that_rule_out_a_refund():
    # A1001 is inside the window, but a damaged item is replaced, not refunded.
    refund = [call("issue_refund", order_id="A1001", reason="split seam")]
    assert PolicySafeRefunds().score(tool_calls=refund, refund_allowed=False).value == 0.0
    assert PolicySafeRefunds().score(tool_calls=[call("escalate_to_human", summary="x")], refund_allowed=False).value == 1.0


def test_metrics_survive_odd_tool_arguments():
    odd = [call("issue_refund"), call("lookup_order", order_id=None), call("lookup_order", order_id=1001)]
    assert PolicySafeRefunds().score(tool_calls=odd).value == 0.0  # a refund with no order ID is not allowed
    assert GroundedOrderIds().score(input="order 1001", tool_calls=odd).value == 0.0  # "1001" is not an order ID


def test_order_ids_in_handles_odd_typing():
    assert order_ids_in("where's #a1001") == {"A1001"}
    assert order_ids_in("order  a1003  and A1002, thanks") == {"A1003", "A1002"}
    assert order_ids_in("my order still hasn't shown up") == set()


def test_grounded_order_ids():
    metric = GroundedOrderIds()
    assert metric.score(input="refund #a1001", tool_calls=[call("check_refund_eligibility", order_id="A1001")]).value == 1.0
    # Escalating without an order ID is fine.
    assert metric.score(input="I want a person", tool_calls=[call("escalate_to_human", summary="x", order_id="")]).value == 1.0
    invented = metric.score(input="my order still hasn't shown up", tool_calls=[call("lookup_order", order_id="A1001")])
    assert invented.value == 0.0
    assert "A1001" in invented.reason


def test_test_cases_are_well_formed():
    tool_names = {tool.__name__ for agent in [root_agent, *root_agent.sub_agents] for tool in agent.tools}
    labels = [case["case"] for case in TEST_CASES]
    assert len(labels) == len(set(labels)), "case labels must be unique"
    for case in TEST_CASES:
        required = {"case", "input", "expected_tools", "expected_agent"}
        assert required <= set(case) <= required | {"refund_allowed"}, case["case"]
        assert set(case["expected_tools"]) <= tool_names, case["case"]
        assert case["expected_agent"] in {"support_agent", "refunds_agent"}, case["case"]


def span(id, parent, type="general", cost=None, tokens=0):
    return SimpleNamespace(id=id, parent_span_id=parent, type=type, total_estimated_cost=cost,
                           usage={"total_tokens": tokens} if tokens else None)


def test_agent_cost_counts_only_the_agent_spans():
    spans = [
        span("run", None),
        span("turn", "run"),
        span("support", "turn"),
        span("llm1", "support", "llm", 0.001, 1000),
        span("refunds", "support"),
        span("llm2", "refunds", "llm", 0.002, 2000),
        span("metrics", None),
        span("judge", "metrics", "llm", 0.5, 9999),  # the judge's call is not the agent's cost
    ]
    cost, tokens = agent_cost(spans, "turn")
    assert round(cost, 6) == 0.003
    assert tokens == 3000


def test_agent_cost_is_none_when_any_agent_call_is_unpriced():
    spans = [span("a", None, "llm", 0.001, 100), span("b", None, "llm", None, 100)]
    assert agent_cost(spans, None) == (None, 200)


def test_agent_cost_is_none_when_opik_has_no_price():
    cost, tokens = agent_cost([span("llm", None, "llm", None, 500)], None)
    assert cost is None
    assert tokens == 500


def test_check_credentials_requires_workspace_with_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("OPIK_API_KEY", "x")
    monkeypatch.delenv("OPIK_WORKSPACE", raising=False)
    assert any("OPIK_WORKSPACE" in m for m in check_credentials())


def test_check_credentials_passes_for_self_hosted(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.delenv("OPIK_API_KEY", raising=False)
    monkeypatch.setenv("OPIK_URL_OVERRIDE", "http://localhost:5173/api")
    assert check_credentials() == []


def test_check_credentials_accepts_workspace_from_opik_config(monkeypatch, tmp_path):
    config = tmp_path / "opik.config"
    config.write_text("[opik]\nworkspace = demo\n")
    monkeypatch.setenv("OPIK_CONFIG_PATH", str(config))
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("OPIK_API_KEY", "x")
    monkeypatch.delenv("OPIK_WORKSPACE", raising=False)
    assert check_credentials() == []
