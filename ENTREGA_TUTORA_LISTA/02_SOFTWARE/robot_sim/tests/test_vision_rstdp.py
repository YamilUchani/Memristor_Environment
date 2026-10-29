import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_vision import make_patterns
from run_vision_rstdp import make_agent, run_classifier


def test_vision_rstdp_eligibility_preserves_bipolar_input_polarity():
    patterns = make_patterns()
    agent = make_agent(
        seed=3,
        eta=0.2,
        initial_x=0.9,
        nv_profile="strukov_ideal.json",
        weights_file=None,
    )

    agent.act(patterns[0].ravel())
    step = agent.last_step()
    probs = np.asarray(step["probs"], dtype=float)
    selected_col = int(step["col"])
    posterior = (
        (1.0 - step["exploration_epsilon"]) * probs[selected_col]
        / step["behavior_probs"][selected_col]
    )
    action_score = -posterior * probs
    action_score[selected_col] += posterior
    centered = step["I_col"] - np.mean(step["I_col"])
    sigma = np.std(step["I_col"])
    dlogp_dcurrent = action_score / (sigma + 1e-8)
    if sigma > 1e-15:
        dlogp_dcurrent -= (
            centered * np.dot(action_score, centered)
            / (agent.brain.M * sigma * (sigma + 1e-8) ** 2)
        )
    minus_dcurrent_dX = (
        step["V_rows"][:, None]
        * (agent.brain._cell_r_off - agent.brain._cell_r_on)
        * agent.brain.G**2
        * agent.steps_per_action
    )
    expected = (
        minus_dcurrent_dX
        * dlogp_dcurrent[None, :]
        * np.where(dlogp_dcurrent >= 0.0, agent.a_plus, agent.a_minus)[None, :]
    )

    assert np.allclose(agent.eligibility, expected)


def test_vision_rstdp_trains_balanced_image_trials_and_records_metrics():
    brain, log = run_classifier(
        make_patterns(),
        episodes=1024,
        eta=0.2,
        seed=3,
        initial_x=0.9,
        verbose=False,
    )

    assert log["episodes_completed"] == 1024
    assert np.array_equal(
        np.bincount(log["class_true"], minlength=8), np.full(8, 128)
    )
    assert len(log["greedy_correct"]) == 1024
    assert len(log["sampled_correct"]) == 1024
    assert len(log["delta_x_abs_mean"]) == 1024
    assert log["final_accuracy_greedy"] >= 0.9
    assert np.isfinite(brain.X).all()
    assert brain.X.shape == (64, 8)
    assert log["weights_source"] is None
