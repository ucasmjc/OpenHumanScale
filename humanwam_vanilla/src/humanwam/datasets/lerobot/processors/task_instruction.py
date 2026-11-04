# Modified by Open-HumanScale contributors: removed unused release runtime code.
"""One text contract for training samples and offline cache validation."""

from collections.abc import Mapping


def task_instruction(data: Mapping) -> str:
    """Preserve the complete task, including literal @ signs and whitespace."""
    task = data.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Dataset sample requires a non-empty string `task` instruction")
    return task
