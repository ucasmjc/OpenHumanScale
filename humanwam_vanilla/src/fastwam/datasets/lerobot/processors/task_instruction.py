"""One text contract for training samples and offline cache validation."""

from collections.abc import Mapping


def task_instruction(data: Mapping) -> str:
    """Preserve the complete task, including literal @ signs and whitespace."""
    task = data.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Dataset sample requires a non-empty string `task` instruction")
    return task


def segment_task_instruction(data: Mapping) -> str:
    """Match the manifest loader's whitespace normalization, preserving literal @."""
    task = task_instruction(data)
    return task_instruction({"task": task.strip()})
