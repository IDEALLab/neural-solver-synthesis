import math

from open_r1.simulators.sds_simulator import SDSSimulator


def test_topk_positive_interaction_normalization_changes_reward_on_skewed_terms():
    simulator_default = SDSSimulator()
    simulator_topk = SDSSimulator(
        config={"normalization_variant": "topk_positive_interactions"}
    )

    requirements = {
        "n_variables": 4,
        "weights": [10.0, 9.0, 8.0, -5.0],
        "cardinality_bounds": [0, 3],
        "interactions": {
            "0,1": 100.0,
            "0,2": 1.0,
            "1,2": 1.0,
            "0,3": 1.0,
            "1,3": 1.0,
            "2,3": 1.0,
        },
    }
    results = {
        "score": 110.0,
        "constraint_violations": 0.0,
        "feasible": True,
    }

    default_reward = simulator_default._calculate_reward(results, requirements)
    topk_reward = simulator_topk._calculate_reward(results, requirements)

    assert math.isclose(default_reward, 1.0)
    assert math.isclose(topk_reward, 110.0 / 129.0, rel_tol=1e-9)
    assert topk_reward < default_reward


def test_topk_positive_interaction_normalization_matches_average_for_uniform_terms():
    simulator_default = SDSSimulator()
    simulator_topk = SDSSimulator(
        config={"normalization_variant": "topk_positive_interactions"}
    )

    requirements = {
        "n_variables": 4,
        "weights": [3.0, 2.0, 1.0, 0.0],
        "cardinality_bounds": [0, 3],
        "interactions": {
            "0,1": 2.0,
            "0,2": 2.0,
            "1,2": 2.0,
        },
    }
    results = {
        "score": 6.0,
        "constraint_violations": 0.0,
        "feasible": True,
    }

    default_reward = simulator_default._calculate_reward(results, requirements)
    topk_reward = simulator_topk._calculate_reward(results, requirements)

    assert math.isclose(default_reward, topk_reward, rel_tol=1e-9)
