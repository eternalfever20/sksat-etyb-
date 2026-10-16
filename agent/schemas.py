"""Function schemas exposed to compatible chat-completion APIs."""

from tools.analysis import tool_definitions


def function_tools() -> list[dict]:
    return tool_definitions()
