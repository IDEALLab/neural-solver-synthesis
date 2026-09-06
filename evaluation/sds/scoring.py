"""Independent SDS objective calculations used by the evaluation harness."""

from __future__ import annotations

from typing import Protocol


class ScorableInstance(Protocol):
    w: list[float]
    W: dict[tuple[int, int], float]


def calculate_true_score(
    instance: ScorableInstance, selected_ids: list[int]
) -> float:
    """Score a binary selection, counting every selected variable once."""
    selected = set(selected_ids)
    objective = sum(instance.w[index] for index in selected)
    objective += sum(
        weight
        for (left, right), weight in instance.W.items()
        if left in selected and right in selected
    )
    return objective
