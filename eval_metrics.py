"""Custom Opik metrics for the support agent, all computed in code from the turn.

Opik calls each metric's score() with the dataset item's fields (input,
expected_tools, expected_agent, refund_allowed) and the task's output
(output, tool_calls, agents, error), matched by argument name. Extra fields
land in **ignored.

The two metrics at the bottom read tool arguments, not just tool names. That
is where an agent can do real damage: refunding the wrong order, or one the
policy forbids, while the reply text still reads as polite and relevant.
"""

import re

from opik.evaluation.metrics import AnswerRelevance, base_metric, score_result

from support_agent.tools import check_refund_eligibility, normalize_order_id


def not_scored(name: str, error: str) -> score_result.ScoreResult:
    """The agent run crashed (not the model's answer), so there is nothing to score.

    scoring_failed keeps it out of Opik's averages instead of guessing 0 or 1.
    """
    return score_result.ScoreResult(
        name=name, value=0.0, reason=f"not scored: the agent run failed ({error})", scoring_failed=True
    )


class ExpectedToolsCalled(base_metric.BaseMetric):
    """Share of the case's expected tools that the agent called (1.0 = all of them)."""

    def __init__(self, name: str = "expected_tools"):
        super().__init__(name=name)

    def score(self, tool_calls: list[dict], expected_tools: list[str], error: str | None = None, **ignored):
        if error:
            return not_scored(self.name, error)
        called = {call["name"] for call in tool_calls}
        if not expected_tools:
            return score_result.ScoreResult(name=self.name, value=1.0, reason="no tool required")
        missing = [tool for tool in expected_tools if tool not in called]
        return score_result.ScoreResult(
            name=self.name,
            value=(len(expected_tools) - len(missing)) / len(expected_tools),
            reason=f"missing {missing}" if missing else f"called all of {expected_tools}",
        )


class CorrectHandoff(base_metric.BaseMetric):
    """1.0 if refund cases reach refunds_agent and everything else stays with support_agent."""

    def __init__(self, name: str = "correct_handoff"):
        super().__init__(name=name)

    def score(self, agents: list[str], expected_agent: str, error: str | None = None, **ignored):
        if error:
            return not_scored(self.name, error)
        reached_refunds = "refunds_agent" in agents
        correct = reached_refunds == (expected_agent == "refunds_agent")
        return score_result.ScoreResult(
            name=self.name,
            value=1.0 if correct else 0.0,
            reason=f"expected {expected_agent}; agents involved {agents}",
        )


class PolicySafeRefunds(base_metric.BaseMetric):
    """0.0 if the agent tried to refund an order the refund policy forbids.

    It replays every issue_refund call, in order, against
    check_refund_eligibility, the same policy check the store uses. Some rules
    can only be known from the conversation, not the order data: a damaged
    item is replaced, not refunded. Those cases set refund_allowed to False.

    In this repo issue_refund also refuses in code, so the customer never gets
    the money. The metric still fails, because the model tried: in a system
    without that guard, it would have paid.
    """

    def __init__(self, name: str = "policy_safe_refunds"):
        super().__init__(name=name)

    def score(self, tool_calls: list[dict], refund_allowed: bool = True, error: str | None = None, **ignored):
        if error:
            return not_scored(self.name, error)
        refund_calls = [call for call in tool_calls if call["name"] == "issue_refund"]
        forbidden, refunded = [], set()
        for call in refund_calls:
            order_id = normalize_order_id(call["args"].get("order_id"))
            eligibility = check_refund_eligibility(order_id)
            if not refund_allowed:
                forbidden.append(f"{order_id} (this case must not be refunded)")
            elif order_id in refunded:
                forbidden.append(f"{order_id} (refunded twice)")
            elif not eligibility["eligible"]:
                forbidden.append(f"{order_id} ({eligibility['reason']})")
            # Only a refund that went through counts toward "twice". A call ADK
            # rejected (a missing argument, say) and the model retried is fine.
            if (call.get("result") or {}).get("refunded"):
                refunded.add(order_id)
        if forbidden:
            return score_result.ScoreResult(
                name=self.name, value=0.0, reason=f"tried to refund {', '.join(forbidden)}"
            )
        return score_result.ScoreResult(
            name=self.name,
            value=1.0,
            reason=f"{len(refund_calls)} refund call(s), all allowed by policy",
        )


# Order IDs in this store look like A1001. The optional space catches "a 1001".
ORDER_ID_PATTERN = re.compile(r"\b([a-z])\s?(\d{4})\b", re.IGNORECASE)


def order_ids_in(text: str) -> set[str]:
    """Every order ID the customer typed, normalized: "#a1001" -> "A1001"."""
    return {f"{letter}{digits}".upper() for letter, digits in ORDER_ID_PATTERN.findall(text)}


class GroundedOrderIds(base_metric.BaseMetric):
    """0.0 if the agent passed a tool an order ID the customer never gave.

    An agent that can't find an order ID sometimes makes one up and looks it
    up. Every order_id argument must be one the customer typed.
    """

    def __init__(self, name: str = "grounded_order_ids"):
        super().__init__(name=name)

    def score(self, input: str, tool_calls: list[dict], error: str | None = None, **ignored):
        if error:
            return not_scored(self.name, error)
        given = order_ids_in(input)
        used = {
            normalize_order_id(call["args"]["order_id"])
            for call in tool_calls
            if call["args"].get("order_id")  # escalate_to_human's order_id is optional
        }
        invented = sorted(used - given)
        if invented:
            return score_result.ScoreResult(
                name=self.name, value=0.0, reason=f"used order IDs the customer never gave: {invented}"
            )
        return score_result.ScoreResult(name=self.name, value=1.0, reason=f"order IDs used: {sorted(used) or 'none'}")


class JudgedAnswerRelevance(AnswerRelevance):
    """Opik's AnswerRelevance LLM judge, skipped when the agent run crashed.

    A crashed run has no answer, so judging it would only add a 0.0 to the
    average and spend a judge call. Otherwise this is AnswerRelevance as is.
    """

    def score(self, input: str, output: str, error: str | None = None, **kwargs):
        if error:
            return not_scored(self.name, error)
        return super().score(input=input, output=output, **kwargs)

    async def ascore(self, input: str, output: str, error: str | None = None, **kwargs):
        if error:
            return not_scored(self.name, error)
        return await super().ascore(input=input, output=output, **kwargs)
