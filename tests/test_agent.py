import json

import pytest

from agent.orchestrator import (
    AgentError,
    OpenRouterRequestError,
    ask,
    available_free_tool_models,
)
from agent.orchestrator import (
    test_connection as run_connection_check,
)


def tool_call(name, args, ident="call_1"):
    return {
        "id": ident,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def response(content=None, calls=None):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": calls or [],
                }
            }
        ]
    }


class FakeTransport:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def __call__(self, api_key, payload):
        self.requests.append((api_key, payload))
        item = next(self.responses)
        if isinstance(item, Exception):
            raise item
        return item


def test_missing_key_is_clear(simple_frame, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(AgentError, match="OPENROUTER_API_KEY"):
        ask("hello", *simple_frame, api_key="", transport=None)


def test_iterative_tool_cycle_and_memory(simple_frame):
    frame, types = simple_frame
    fake = FakeTransport(
        [
            response(calls=[tool_call("describe_dataset", {})]),
            response(content="There are 5 rows [ev-placeholder]."),
        ]
    )
    result = ask(
        "How many rows?",
        frame,
        types,
        transport=fake,
        api_key="test-key",
        memories=[{"kind": "preference", "key": "units", "value": "whole numbers"}],
        max_steps=4,
    )
    assert len(result["trace"]) == 1
    assert result["trace"][0]["result"]["rows"] == 5
    assert result["memories_used"] and len(fake.requests) == 2
    assert fake.requests[1][1]["messages"][-1]["role"] == "tool"
    evidence_id = result["trace"][0]["result"]["evidence_id"]
    assert evidence_id in result["answer"]
    assert fake.requests[0][1]["tools"]
    assert fake.requests[0][1]["model"] == "openrouter/free"


def test_tool_limit_enforced(simple_frame):
    frame, types = simple_frame
    fake = FakeTransport(
        [
            response(calls=[tool_call("describe_dataset", {}, "1")]),
            response(calls=[tool_call("describe_dataset", {}, "2")]),
        ]
    )
    result = ask(
        "inspect", frame, types, transport=fake, api_key="test-key", max_steps=2
    )
    assert result["limit_reached"] and len(result["trace"]) == 2


def test_malformed_tool_arguments_become_tool_error(simple_frame):
    frame, types = simple_frame
    fake = FakeTransport(
        [
            response(
                calls=[
                    {
                        "id": "bad-call",
                        "type": "function",
                        "function": {
                            "name": "describe_dataset",
                            "arguments": "not-json",
                        },
                    }
                ]
            ),
            response(calls=[tool_call("describe_dataset", {}, "good-call")]),
            response(
                content="The tool arguments were invalid; table inspected [ev-placeholder]."
            ),
        ]
    )
    result = ask("Inspect the table", frame, types, transport=fake, api_key="test-key")
    assert len(result["trace"]) == 2
    assert not result["trace"][0]["result"]["ok"]
    assert any(
        "preceding analysis tool call failed" in (message.get("content") or "")
        for message in fake.requests[1][1]["messages"]
    )


def test_provider_failure_does_not_leak_exception_text(simple_frame):
    def broken(_api_key, _payload):
        raise RuntimeError("secret key hidden")

    with pytest.raises(AgentError, match="OpenRouter request failed") as exc:
        ask("question", *simple_frame, api_key="test-key", transport=broken)
    assert "secret key hidden" not in str(exc.value)


def test_503_retries_and_succeeds(simple_frame):
    class Busy:
        status = 503

    frame, types = simple_frame
    fake = FakeTransport(
        [
            OpenRouterRequestError(503),
            response(calls=[tool_call("describe_dataset", {})]),
            response(content="It is 5 rows [ev-placeholder]."),
        ]
    )
    result = ask("How many rows?", frame, types, transport=fake, api_key="test-key")
    assert len(fake.requests) == 3
    assert "5 rows" in result["answer"]
    assert all(payload["model"] == "openrouter/free" for _, payload in fake.requests)


def test_unauthorized_response_is_actionable(simple_frame):
    def unauthorized(_api_key, _payload):
        raise OpenRouterRequestError(401)

    with pytest.raises(AgentError, match="OpenRouter rejected the API key"):
        ask("question", *simple_frame, api_key="bad-key", transport=unauthorized)


def test_insufficient_balance_is_actionable(simple_frame):
    def no_balance(_api_key, _payload):
        raise OpenRouterRequestError(402)

    with pytest.raises(AgentError, match="could not route"):
        ask("question", *simple_frame, api_key="test-key", transport=no_balance)


def test_model_list_keeps_only_free_tool_calling_models(monkeypatch):
    from agent import orchestrator

    payload = {
        "data": [
            {
                "id": "author/free-tools",
                "name": "Free Tools",
                "supported_parameters": ["tools"],
                "pricing": {"prompt": "0", "completion": "0"},
            },
            {
                "id": "author/paid-tools",
                "supported_parameters": ["tools"],
                "pricing": {"prompt": "0.1", "completion": "0.1"},
            },
            {
                "id": "author/free-no-tools",
                "supported_parameters": [],
                "pricing": {"prompt": "0", "completion": "0"},
            },
        ]
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return json.dumps(payload).encode()

    monkeypatch.setattr(
        orchestrator.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    assert available_free_tool_models() == [
        {"id": "author/free-tools", "name": "Free Tools"}
    ]


def test_wrong_provider_response_is_actionable(simple_frame):
    def empty(_api_key, _payload):
        return {"choices": []}

    with pytest.raises(AgentError, match="returned no answer"):
        ask("question", *simple_frame, api_key="test-key", transport=empty)


def test_connection_accepts_a_provider_text_reply():
    fake = FakeTransport([response(content="Connected")])
    message = run_connection_check(
        "openrouter/free", api_key="test-key", transport=fake
    )
    assert message.startswith("Connected to OpenRouter")
    assert message == "Connected to OpenRouter; openrouter/free responded."
    assert fake.requests[0][1]["max_tokens"] >= 128
