from types import SimpleNamespace

from evaluation.sds.scoring import calculate_true_score


def test_duplicate_variable_ids_do_not_inflate_objective() -> None:
    instance = SimpleNamespace(
        w=[10.0, 20.0, 30.0],
        W={(0, 1): 5.0, (1, 2): 7.0},
    )

    assert calculate_true_score(instance, [0, 0, 1, 1]) == 35.0


def test_empty_selection_has_zero_objective() -> None:
    instance = SimpleNamespace(w=[10.0], W={})

    assert calculate_true_score(instance, []) == 0.0
