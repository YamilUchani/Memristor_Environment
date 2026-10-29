import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_car_rstdp
from car_env import ACTION_NAMES, CarEnv, N_ACTIONS
from neurobot.rstdp_cartpole_agent import NeuromorphicAgent as RSTDPAgent


def test_car_rstdp_agent_uses_auto2d_and_eight_action_crossbar():
    agent = run_car_rstdp.make_agent(seed=32, initial_x=0.1)

    assert agent.encoder.encoding == "auto2d"
    assert agent.brain.X.shape == (24, 8)
    assert agent.n_actions == N_ACTIONS == len(ACTION_NAMES)


def test_car_rstdp_greedy_evaluation_preserves_weights_and_action_rng():
    seed = 41
    agent = run_car_rstdp.make_agent(seed=seed)
    env = CarEnv(seed=seed)
    agent.act_deterministic(env.reset(seed=seed))
    weights_before = agent.brain.X.copy()
    rng_before = json.dumps(agent.decoder.rng.bit_generator.state, sort_keys=True)
    sample_before = agent.decoder.sample

    result = run_car_rstdp._greedy_evaluate(
        agent,
        [run_car_rstdp._training_map_seed(seed)] * 2,
        same_map=True,
    )

    np.testing.assert_array_equal(agent.brain.X, weights_before)
    assert json.dumps(
        agent.decoder.rng.bit_generator.state, sort_keys=True
    ) == rng_before
    assert agent.decoder.sample is sample_before
    assert result["episodes"] == 2
    assert result["same_map"]
    assert result["map_seeds"] == [
        run_car_rstdp._training_map_seed(seed)
    ] * 2
    assert result["successes"] + result["collisions"] + result["timeouts"] == 2
    assert result["mean_distance"] >= 0.0
    assert 0.0 <= result["mean_path_efficiency"] <= 1.0
    assert all(
        "unproductive_turns" in outcome
        for outcome in result["outcomes"]
    )


def test_car_rstdp_load_rejects_non_auto2d_encoder(tmp_path, monkeypatch):
    monkeypatch.setattr(run_car_rstdp, "OUTPUT_DIR", tmp_path)
    weights_path = tmp_path / "other_encoder.npy"
    np.save(weights_path, np.full((24, 8), 0.5))
    Path(f"{weights_path}.meta.json").write_text(
        json.dumps({"encoder_mode": "random"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Auto 2D R-STDP requiere"):
        run_car_rstdp.make_agent(seed=2, weights_file=weights_path.name)


def test_car_rstdp_training_records_learning_and_turn_metrics(monkeypatch):
    class _FakeAgent:
        def __init__(self):
            self.brain = type(
                "Brain", (), {"X": np.full((24, 8), 0.5)}
            )()
            self.baseline = 0.0
            self.decoder = type("Decoder", (), {"sample": True})()
            self.learned_episodes = 0

        def act(self, _observation):
            return 0

        def observe_reward(self, _reward):
            pass

        def finish_episode(self, _episode_return):
            self.learned_episodes += 1
            return {
                "reward_modulated_eligibility_abs_mean": 0.1,
                "actual_delta_x_abs_mean": 0.01,
                "actual_delta_x_abs_max": 0.02,
                "actual_delta_g_abs_mean": 0.03,
                "update_applied": True,
                "pre_events": 3,
                "post_spikes": 2,
                "selected_post_events": 1,
            }

        def end_episode(self):
            pass

        def reset_episode_state(self):
            pass

    monkeypatch.setattr(
        run_car_rstdp,
        "_greedy_evaluate",
        lambda *_args, **_kwargs: {
            "success_rate": 0.25,
            "mean_unproductive_turns": 0.5,
            "outcomes": [],
        },
    )
    agent = _FakeAgent()
    log = run_car_rstdp.run_training(
        agent,
        CarEnv(seed=4),
        episodes=3,
        seed=4,
        verbose=False,
        policy_eval_interval=2,
    )

    assert log["episodes_completed"] == 3
    assert agent.learned_episodes == 3
    assert log["pre_events"] == [3, 3, 3]
    assert log["post_spikes"] == [2, 2, 2]
    assert log["delta_x_abs_mean"] == [0.01, 0.01, 0.01]
    assert log["delta_g_abs_mean"] == [0.03, 0.03, 0.03]
    assert len(log["unproductive_turns"]) == 3
    assert [item["episode"] for item in log["policy_evaluations"]] == [2, 3]
    assert log["policy_evaluations"][0]["success_rate"] == 0.25
    assert log["held_out_test_evaluation"]["success_rate"] == 0.25
