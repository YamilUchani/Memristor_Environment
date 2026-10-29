import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_car import (
    _adapt_temperature,
    _training_finished,
    _validate_car_weights,
    run_headless,
)
from run_car import (
    GreedyPolicyTracker,
    M,
    N,
    POLICY_TEST_SEED_OFFSET,
    _evaluation_map_seeds,
    _parse_positive_integer,
    _training_map_seed,
)
from car_env import (
    CarEnv,
    N_ACTIONS,
    OMEGA_MAX,
    REWARD_STEERING_SMOOTHNESS,
)
from neurobot.reinforce_agent import NeuromorphicAgent as ReinforceAgent


class _FakeBrain:
    def __init__(self):
        self.X = np.full((2, 2), 0.5)

    def set_weights(self, weights):
        self.X = np.asarray(weights).copy()


class _FakeAgent:
    def __init__(self):
        self.brain = _FakeBrain()
        self.decoder = type(
            "Decoder",
            (),
            {
                "temperature": 1.0,
                "set_temperature": lambda decoder, value: setattr(
                    decoder, "temperature", max(value, 0.001)
                ),
            },
        )()
        self.learned_episodes = 0
        self.max_weight_update = 0.01

    def act(self, _observation):
        return 0

    def act_deterministic(self, _observation):
        return 0

    def reset_episode_state(self):
        pass

    def learn_from_rewards(self, _rewards):
        self.learned_episodes += 1

    def end_episode(self):
        pass

    def current_temperature(self):
        return self.decoder.temperature


class _OneStepEnv:
    def __init__(self):
        self.trail = [[0.0, 0.0], [1.0, 1.0]]
        self.steps = 0
        self.reset_seeds = []

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.steps = 0
        return np.zeros(2)

    def step(self, _action):
        self.steps += 1
        return np.zeros(2), 1.0, True, {
            "reason": "goal",
            "dist": 0.0,
            "steps": self.steps,
            "collision_type": None,
        }


class _FakePolicyTracker:
    def __init__(self, solve_at=None):
        self.solve_at = solve_at
        self.history = []
        self.best_success = 1.0
        self.best_mean_steps = 1.0
        self.validation_regressed = False
        self.solved = False
        self.current_update_limit = 0.01

    def evaluate(self, _agent, episode):
        self.solved = self.solve_at is not None and episode >= self.solve_at
        result = {
            "episode": episode,
            "success_rate": 1.0,
            "best_success_rate": 1.0,
            "solved": self.solved,
        }
        self.history.append(result)
        return result


def test_gui_episode_limit_is_a_hard_upper_bound():
    assert not _training_finished(9, 10)
    assert _training_finished(10, 10)


@pytest.mark.parametrize(("value", "expected"), [("1", 1), ("3000", 3000)])
def test_episode_configuration_accepts_positive_integers(value, expected):
    assert _parse_positive_integer(value, "episodes") == expected


@pytest.mark.parametrize("value", ["", "0", "-2", "2.5", "abc"])
def test_episode_configuration_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="entero positivo"):
        _parse_positive_integer(value, "episodes")


def test_headless_training_continues_through_sustained_validation_success():
    agent = _FakeAgent()
    tracker = _FakePolicyTracker(solve_at=1)
    env = _OneStepEnv()

    log = run_headless(
        env,
        agent,
        episodes=5,
        seed=3,
        verbose=False,
        policy_tracker=tracker,
        eval_interval=1,
    )

    assert log["episodes_requested"] == 5
    assert log["episodes_completed"] == 5
    assert agent.learned_episodes == 5
    assert log["policy_solved"]
    assert log["termination_reason"] == ["goal"] * 5
    assert log["episode_steps"] == [1] * 5
    assert log["collision_type"] == [None] * 5
    assert log["training_map_seed"] == _training_map_seed(3)
    assert log["training_map_mode"] == "fixed_map_per_run"
    assert log["training_map_seeds"] == [_training_map_seed(3)] * 5
    assert log["path_length"] == pytest.approx([np.sqrt(2.0)] * 5)
    assert log["path_efficiency"] == pytest.approx([1.0] * 5)
    assert log["steering_reversals"] == [0] * 5
    assert env.reset_seeds == [_training_map_seed(3), None, None, None, None]


def test_headless_training_never_exceeds_requested_episode_count():
    tracker = _FakePolicyTracker()
    env = _OneStepEnv()
    log = run_headless(
        env,
        _FakeAgent(),
        episodes=3,
        seed=3,
        verbose=False,
        policy_tracker=tracker,
        eval_interval=1,
    )

    assert log["episodes_requested"] == 3
    assert log["episodes_completed"] == 3
    assert len(log["policy_evaluations"]) == 3
    assert len(log["termination_reason"]) == 3
    assert len(log["episode_steps"]) == 3
    assert log["training_map_seed"] == _training_map_seed(3)
    assert log["training_map_seeds"] == [_training_map_seed(3)] * 3
    assert env.reset_seeds == [_training_map_seed(3), None, None]


def test_headless_training_can_disable_validation_temperature_adaptation(
    monkeypatch,
):
    import run_car

    def fail_if_adapted(*_args):
        pytest.fail("Validation should not adjust exploration in fixed mode.")

    monkeypatch.setattr(run_car, "_adapt_temperature", fail_if_adapted)
    tracker = _FakePolicyTracker()
    log = run_headless(
        _OneStepEnv(),
        _FakeAgent(),
        episodes=2,
        seed=3,
        verbose=False,
        policy_tracker=tracker,
        eval_interval=1,
        adapt_exploration=False,
    )

    assert log["episodes_completed"] == 2
    assert len(log["policy_evaluations"]) == 2


def test_headless_training_continues_and_records_validation_regression():
    class _RegressingTracker(_FakePolicyTracker):
        def evaluate(self, _agent, episode):
            success_rate = 0.95 if episode == 1 else 0.85
            self.validation_regressed = episode > 1
            self.solved = False
            result = {
                "episode": episode,
                "success_rate": success_rate,
                "best_success_rate": max(
                    [item["best_success_rate"] for item in self.history]
                    + [success_rate]
                ),
                "solved": self.solved,
            }
            self.history.append(result)
            return result

    agent = _FakeAgent()
    tracker = _RegressingTracker()
    log = run_headless(
        _OneStepEnv(),
        agent,
        episodes=5,
        seed=3,
        verbose=False,
        policy_tracker=tracker,
        eval_interval=1,
    )

    assert log["episodes_completed"] == 5
    assert agent.learned_episodes == 5
    assert tracker.validation_regressed


def test_tracker_records_regression_without_restoring_old_weights(monkeypatch):
    import run_car

    monkeypatch.setattr(run_car, "POLICY_SOLVED_SUCCESS", 0.9)
    outcomes = [19, 16]

    class _ValidationEnv:
        created = 0

        def __init__(self, seed):
            self.index = type(self).created
            type(self).created += 1
            self.steps = 0
            self.episode = -1
            self.trail = [[0.0, 0.0], [1.0, 1.0]]

        def reset(self, seed=None):
            self.episode += 1
            return np.zeros(2)

        def step(self, _action):
            self.steps += 1
            success = self.episode < outcomes[self.index]
            return np.zeros(2), 0.0, True, {
                "reason": "goal" if success else "collision",
            }

    monkeypatch.setattr(run_car, "CarEnv", _ValidationEnv)
    tracker = GreedyPolicyTracker(seed=3, eval_episodes=20)
    agent = _FakeAgent()

    first = tracker.evaluate(agent, episode=100)
    agent.brain.set_weights(np.full((2, 2), 0.25))
    second = tracker.evaluate(agent, episode=200)

    assert first["success_rate"] == 0.95
    assert first["success_confirmations"] == 1
    assert second["success_rate"] == 0.8
    assert second["validation_regressed"]
    assert tracker.best_episode == 100
    np.testing.assert_array_equal(
        agent.brain.X,
        np.full((2, 2), 0.25),
    )
    assert tracker.map_seeds == [_training_map_seed(3)] * 20


def test_reinforce_update_respects_adaptive_weight_limit():
    env = CarEnv(seed=9)
    agent = ReinforceAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        N=N,
        M=M,
        V_read=0.05,
        eta=0.01,
        encoder_mode="random",
        seed=9,
        initial_X=0.1,
        temp_init=1.5,
        temp_min=0.01,
    )
    agent._episode_scores = [np.ones((N, M))]
    agent.max_weight_update = 0.0005

    update = agent.learn_from_rewards([1.0])["dX"]

    assert np.max(np.abs(update)) == pytest.approx(0.0005)


def test_car_environment_map_changes_reproducibly_with_reset_seed():
    env = CarEnv(seed=7)

    env.reset(seed=101)
    first_map = list(env.obstacles)
    first_heading = env.theta
    env.reset(seed=102)
    second_map = list(env.obstacles)
    assert second_map != first_map

    env.reset(seed=101)
    assert env.obstacles == first_map
    assert env.theta == first_heading


def test_car_environment_exposes_eight_distinct_control_actions():
    env = CarEnv(seed=12)
    assert env.n_actions == N_ACTIONS == 8
    expected_turn = {
        0: 0.0,
        1: 0.0,
        2: OMEGA_MAX,
        3: -OMEGA_MAX,
        4: 2.0 * OMEGA_MAX,
        5: -2.0 * OMEGA_MAX,
        6: OMEGA_MAX,
        7: -OMEGA_MAX,
    }

    for action, expected_delta in expected_turn.items():
        env.reset(seed=123)
        theta_before = env.theta
        _, _, _, _ = env.step(action)
        delta = (env.theta - theta_before + np.pi) % (2 * np.pi) - np.pi
        assert delta == pytest.approx(expected_delta)

    with pytest.raises(ValueError, match="integer"):
        env.step(8)


def test_car_observes_speed_and_directional_wall_or_obstacle_clearance():
    env = CarEnv(seed=18)
    observation = env.reset(seed=18)
    assert observation.shape == (11,)
    assert observation[10] == 0.0

    env.x, env.y, env.theta = 1.0, 5.0, 0.0
    env.obstacles = [(4.0, 5.0)]
    sensors, _ = env._sensors()
    assert sensors[0] == pytest.approx((4.0 - 0.4 - 1.0) / 3.0)

    env.obstacles = []
    env.x = 9.0
    sensors, _ = env._sensors()
    assert sensors[0] == pytest.approx(1.0 / 3.0)

    env.reset(seed=18)
    sensors, _, _, _ = env.step(0)
    assert sensors[10] > 0.0


def test_front_proximity_to_wall_has_a_measurable_reward_penalty():
    env = CarEnv(seed=19)
    env.reset(seed=19)
    env.obstacles = []
    env.x, env.y, env.theta = 8.8, 5.0, 0.0
    _, _, _, info = env.step(0)

    assert info["front_hazard"] > 0.0
    assert info["front_proximity_penalty"] == pytest.approx(
        1.5 * info["front_hazard"] ** 2
    )


def test_unproductive_full_turn_is_penalized_but_tracked_separately():
    env = CarEnv(seed=20)
    env.reset(seed=20)
    env.obstacles = []
    env.x, env.y, env.theta = 5.0, 5.0, 0.0
    env.prev_dist = env._dist_goal()
    env._turn_progress_anchor_dist = env._dist_goal()

    rewards = []
    infos = []
    for _ in range(int(np.ceil(2.0 * np.pi / OMEGA_MAX))):
        _, reward, done, info = env.step(2)
        rewards.append(reward)
        infos.append(info)
        assert not done

    assert infos[-1]["unproductive_turns"] == 1
    assert infos[-1]["step_unproductive_turn_penalty"] == pytest.approx(2.0)
    assert infos[-1]["turning_angle"] >= 2.0 * np.pi
    assert rewards[-1] < rewards[-2]


def test_unproductive_turn_detection_includes_reversing_steering():
    env = CarEnv(seed=21)
    env.reset(seed=21)
    env.obstacles = []
    env.x, env.y = 5.0, 5.0
    goal_heading = np.arctan2(8.5 - env.y, 8.5 - env.x)
    env.theta = goal_heading + np.pi / 2.0
    env.prev_dist = env._dist_goal()
    env._turn_progress_anchor_dist = env._dist_goal()

    infos = []
    for step in range(50):
        _, _, done, info = env.step(2 if step % 2 == 0 else 3)
        infos.append(info)
        if done:
            break

    assert infos[-1]["turning_angle"] >= 2.0 * np.pi
    assert infos[-1]["unproductive_turns"] >= 1
    assert infos[-1]["unproductive_turn_penalty"] >= 0.0
    assert any(item["step_unproductive_turn_penalty"] > 0.0 for item in infos)


def test_auto2d_encoder_preserves_ray_goal_and_speed_features():
    from neurobot.encoder import SensorEncoder

    encoder = SensorEncoder(
        obs_dim=11,
        N=24,
        V_read=0.05,
        encoding="auto2d",
        seed=3,
    )
    observation = np.asarray([0.9] * 8 + [0.0, 0.5, 0.0])
    slow_features = encoder.encode(observation)
    fast_observation = observation.copy()
    fast_observation[10] = 1.0
    fast_features = encoder.encode(fast_observation)

    assert slow_features.shape == (24,)
    assert not np.array_equal(slow_features[21:23], fast_features[21:23])
    assert slow_features[16:19].max() == pytest.approx(0.05)
    assert slow_features[23] == pytest.approx(0.05)


def test_temperature_tracks_greedy_validation_not_noisy_training_success():
    class _TemperatureAgent:
        temp_init = 1.5
        temp_min = 0.3

        def __init__(self):
            self.decoder = type(
                "Decoder",
                (),
                {
                    "temperature": 1.0,
                    "set_temperature": lambda decoder, value: setattr(
                        decoder, "temperature", max(value, 0.001)
                    ),
                },
            )()

        def current_temperature(self):
            return self.decoder.temperature

    agent = _TemperatureAgent()
    _adapt_temperature(agent, [0.1, 0.1, 0.2])
    assert agent.current_temperature() == pytest.approx(0.95)

    agent = _TemperatureAgent()
    _adapt_temperature(agent, [0.2, 0.2, 0.2])
    _adapt_temperature(agent, [0.2, 0.2, 0.2, 0.2])
    assert agent.current_temperature() == pytest.approx(1.05)


def test_steering_changes_are_penalized_and_reported(monkeypatch):
    import car_env

    smooth_env = CarEnv(seed=15)
    baseline_env = CarEnv(seed=15)
    smooth_env.reset(seed=15)
    baseline_env.reset(seed=15)
    first_action = smooth_env.step(2)
    baseline_env.step(2)

    second_action = smooth_env.step(3)
    monkeypatch.setattr(car_env, "REWARD_STEERING_SMOOTHNESS", 0.0)
    baseline_second_action = baseline_env.step(3)

    info = second_action[3]
    assert info["steering_reversal"]
    assert info["steering_reversals"] == 1
    assert info["steering_change"] == pytest.approx(-2.0)
    assert info["steering_smoothness_penalty"] == pytest.approx(
        REWARD_STEERING_SMOOTHNESS * 4.0
    )
    assert second_action[1] == pytest.approx(
        baseline_second_action[1] - info["steering_smoothness_penalty"]
    )
    assert info["path_length"] > 0.0

    smooth_env.reset(seed=15)
    assert smooth_env.steering_reversals == 0
    assert smooth_env.steering_smoothness_penalty == 0.0
    assert smooth_env.path_length == 0.0


def test_car_reinforce_crossbar_has_one_output_per_action(tmp_path):
    env = CarEnv(seed=9)
    agent = ReinforceAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        N=N,
        M=M,
        seed=9,
    )

    assert M == N_ACTIONS
    assert agent.brain.X.shape == (N, N_ACTIONS)
    assert agent.decoder.n_actions == N_ACTIONS
    agent.decoder.sample = False
    assert agent.decoder.decode(np.arange(N_ACTIONS)) == N_ACTIONS - 1
    observation = env.reset(seed=9)
    action = agent.act_deterministic(observation)
    assert 0 <= action < N_ACTIONS
    assert len(env.step(action)) == 4

    incompatible_path = tmp_path / "four_action_crossbar.npy"
    np.save(incompatible_path, np.full((N, 4), 0.5))
    Path(f"{incompatible_path}.meta.json").write_text(json.dumps({
        "N": N,
        "M": 4,
        "convention": "X = 1 - x",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="eight actions"):
        _validate_car_weights(incompatible_path)
    with pytest.raises(ValueError, match="dimensions"):
        agent.brain.load_weights(str(incompatible_path))


def test_training_and_greedy_evaluation_use_disjoint_map_seeds():
    train_seeds = {_training_map_seed(5)}
    eval_seeds = set(_evaluation_map_seeds(20))
    test_seeds = set(
        _evaluation_map_seeds(20, seed_offset=POLICY_TEST_SEED_OFFSET)
    )

    assert len(eval_seeds) == 20
    assert train_seeds.isdisjoint(eval_seeds)
    assert train_seeds.isdisjoint(test_seeds)
    assert eval_seeds.isdisjoint(test_seeds)
    assert eval_seeds == set(_evaluation_map_seeds(20))
    assert train_seeds.isdisjoint({_training_map_seed(6)})


def test_training_reuses_one_map_but_varies_heading_reproducibly():
    env = CarEnv(seed=7)
    map_seed = _training_map_seed(7)
    headings = []
    maps = []
    for episode in range(4):
        if episode == 0:
            env.reset(seed=map_seed)
        else:
            env.reset()
        headings.append(env.theta)
        maps.append(list(env.obstacles))

    assert all(obstacles == maps[0] for obstacles in maps)
    assert len(set(headings)) == len(headings)


def test_greedy_evaluation_is_isolated_and_does_not_change_weights(monkeypatch):
    import run_car

    monkeypatch.setattr(run_car, "POLICY_SOLVED_SUCCESS", 0.0)
    seed = 7
    env = CarEnv(seed=seed)
    agent = ReinforceAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        N=N,
        M=M,
        V_read=0.05,
        eta=0.01,
        encoder_mode="random",
        seed=seed,
        initial_X=0.9,
        temp_init=1.5,
        temp_min=0.01,
    )
    initial_weights = agent.brain.X.copy()
    observation = env.reset(seed=seed)
    agent.act_deterministic(observation)
    assert agent.brain._mna_factorization is not None
    cached_factorization = agent.brain._mna_factorization
    initial_cell_states = np.asarray([
        [cell.x for cell in row] for row in agent.brain.cells
    ])
    rng_state = copy.deepcopy(agent.decoder.rng.bit_generator.state)
    tracker = GreedyPolicyTracker(seed, eval_episodes=2)

    evaluation = tracker.evaluate(agent, episode=100)

    np.testing.assert_array_equal(agent.brain.X, initial_weights)
    assert agent.brain._mna_factorization is cached_factorization
    assert agent.decoder.rng.bit_generator.state == rng_state
    assert (
        evaluation["successful_episodes"]
        + evaluation["collisions"]
        + evaluation["timeouts"]
        == evaluation["episodes"]
    )
    assert evaluation["best_episode"] == 100
    assert evaluation["map_seeds"] == [_training_map_seed(seed)] * 2
    assert not evaluation["solved"]
    assert evaluation["success_confirmations"] == 1
    second_evaluation = tracker.evaluate(agent, episode=200)
    assert not second_evaluation["solved"]
    assert second_evaluation["success_confirmations"] == 2
    third_evaluation = tracker.evaluate(agent, episode=300)
    assert third_evaluation["solved"]
    assert not third_evaluation["validation_regressed"]
    assert third_evaluation["success_confirmations"] == 3
    test_tracker = GreedyPolicyTracker(
        seed,
        eval_episodes=2,
        map_seed_offset=POLICY_TEST_SEED_OFFSET,
    )
    assert set(test_tracker.map_seeds).isdisjoint(evaluation["map_seeds"])
    np.testing.assert_array_equal(agent.brain.X, initial_weights)
    np.testing.assert_array_equal(
        np.asarray([[cell.x for cell in row] for row in agent.brain.cells]),
        initial_cell_states,
    )


def test_evaluation_only_never_learns_and_uses_training_map():
    agent = _FakeAgent()
    tracker = _FakePolicyTracker()
    env = _OneStepEnv()

    log = run_headless(
        env,
        agent,
        episodes=4,
        seed=6,
        verbose=False,
        policy_tracker=tracker,
        eval_interval=10,
        evaluation_only=True,
    )

    assert agent.learned_episodes == 0
    assert log["evaluation_only"]
    assert log["training_map_mode"] == "fixed_map_for_greedy_evaluation"
    assert log["training_map_seeds"] == [_training_map_seed(6)] * 4
    assert env.reset_seeds == [_training_map_seed(6), None, None, None]
