from pathlib import Path

import pytest

from agent.orchestrator import ask
from memory.store import MemoryStore, parse_plain_definition


def test_scoped_upsert_retrieve_update_delete(tmp_path):
    db = tmp_path / "m.sqlite3"
    store = MemoryStore(str(db), "test")
    a = store.upsert(
        {
            "kind": "metric_definition",
            "key": "net sales",
            "value": "exclude returned rows",
        },
        "abc",
    )
    assert (
        store.relevant("net sales total", "abc")[0]["value"] == "exclude returned rows"
    )
    assert store.relevant("net sales", "other") == []
    assert len(store.list_all()) == 1
    b = store.upsert(
        {
            "kind": "metric_definition",
            "key": "net sales",
            "value": "exclude returns and tax",
        },
        "abc",
    )
    assert b["updated"] and len(store.list("abc")) == 1
    reopened = MemoryStore(str(db), "test")
    assert reopened.list("abc")[0]["value"] == "exclude returns and tax"
    assert reopened.delete(a["id"]) and not reopened.list("abc")


def test_dataset_definition_overrides_workspace_conflict_after_reopen(tmp_path):
    db = str(tmp_path / "conflicts.sqlite3")
    initial = MemoryStore(db, "test")
    initial.upsert(
        {
            "kind": "metric_definition",
            "key": "net sales",
            "value": "Include all rows.",
        }
    )
    first = initial.upsert(
        {
            "kind": "metric_definition",
            "key": "net sales",
            "value": "Exclude cancelled rows.",
        },
        "dataset-a",
    )
    updated = initial.upsert(
        {
            "kind": "metric_definition",
            "key": "net sales",
            "value": "Exclude cancelled and refunded rows.",
        },
        "dataset-a",
    )
    assert updated["updated"] and updated["id"] == first["id"]

    reopened = MemoryStore(db, "test")
    selected = reopened.relevant("Calculate net sales", "dataset-a")
    other_dataset = reopened.relevant("Calculate net sales", "dataset-b")
    assert len(selected) == 1
    assert selected[0]["value"] == "Exclude cancelled and refunded rows."
    assert len(other_dataset) == 1
    assert other_dataset[0]["value"] == "Include all rows."


def test_workspace_memory_and_clear(tmp_path):
    a = MemoryStore(str(tmp_path / "m.sqlite3"), "a")
    b = MemoryStore(str(tmp_path / "m.sqlite3"), "b")
    a.upsert({"kind": "preference", "key": "frequency", "value": "monthly"})
    updated = a.upsert({"kind": "preference", "key": "frequency", "value": "weekly"})
    assert updated["updated"] and len(a.list()) == 1
    assert len(a.list()) == 1 and b.list() == []
    assert a.clear() == 1 and a.list() == []


def test_plain_definition_parser_preserves_relations():
    exclusion = parse_plain_definition("active customers excludes cancelled accounts")
    assert exclusion == {
        "kind": "metric_definition",
        "key": "active customers",
        "value": "excludes cancelled accounts",
    }
    numeric = parse_plain_definition("highly rated = IMDB_Rating >= 8.5")
    assert numeric["key"] == "highly rated"
    assert numeric["value"] == "IMDB_Rating >= 8.5"
    assert (
        parse_plain_definition("always use monthly summaries")["kind"] == "preference"
    )
    with pytest.raises(ValueError, match="Use a sentence"):
        parse_plain_definition("please remember something")


def test_saved_definition_can_change_later_calculation(tmp_path):
    from dataset.loader import load_dataset

    data = Path(__file__).parents[1] / "data/demo_sales.csv"
    frame, info = load_dataset(data.name, data.read_bytes())
    memory = MemoryStore(str(tmp_path / "memory.sqlite3"), "test")
    memory.upsert(
        {
            "kind": "metric_definition",
            "key": "retained revenue",
            "value": "exclude rows where returned is true",
        },
        info["fingerprint"],
    )

    class FakeTransport:
        def __init__(self):
            self.calls = iter([self.response(True), self.response(False)])

        @staticmethod
        def response(first):
            if first:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "filter_rows",
                                "arguments": (
                                    '{"conditions":[{"column":"returned","operator":"eq",'
                                    '"value":false}],"count_only":true}'
                                ),
                            },
                        }
                    ],
                }
            else:
                message = {
                    "role": "assistant",
                    "content": "Mean retained revenue computed from non-returned rows [ev-123].",
                    "tool_calls": [],
                }
            return {"choices": [{"message": message}]}

        def __call__(self, _api_key, _payload):
            return next(self.calls)

    result = ask(
        "What is retained revenue?",
        frame,
        info["types"],
        api_key="test-key",
        transport=FakeTransport(),
        memories=memory.relevant("retained revenue", info["fingerprint"]),
    )
    tool = result["trace"][0]["result"]
    assert result["memories_used"] and tool["count_only"] is True
    assert tool["matched_rows"] == 33
