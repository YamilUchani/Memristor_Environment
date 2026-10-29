import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_applications import (
    make_disjoint_noisy_vision_sets,
    make_noisy_vision_set,
)
from neurobot.agent import NeuromorphicAgent
from neurobot.reinforce_agent import NeuromorphicAgent as ReinforceAgent
from neurobot.rstdp_cartpole_agent import NeuromorphicAgent as RSTDPAgent
from run_vision import make_patterns


def test_noisy_vision_sets_are_balanced_and_reproducible():
    patterns = make_patterns()
    images_a, labels_a = make_noisy_vision_set(
        patterns, seed=123, examples_per_class=7
    )
    images_b, labels_b = make_noisy_vision_set(
        patterns, seed=123, examples_per_class=7
    )

    assert images_a.shape == (56, 8, 8)
    assert labels_a.shape == (56,)
    assert np.array_equal(images_a, images_b)
    assert np.array_equal(labels_a, labels_b)
    assert np.array_equal(np.bincount(labels_a), np.full(8, 7))


def test_noisy_vision_sets_from_disjoint_seeds_differ():
    patterns = make_patterns()
    validation, _, test, _, test_seed = make_disjoint_noisy_vision_sets(
        patterns,
        validation_seed=10,
        test_seed=20,
        validation_examples_per_class=20,
        test_examples_per_class=20,
    )

    validation_examples = {image.tobytes() for image in validation}
    test_examples = {image.tobytes() for image in test}
    training_examples = {pattern.astype(float).tobytes() for pattern in patterns}
    assert validation_examples.isdisjoint(test_examples)
    assert validation_examples.isdisjoint(training_examples)
    assert test_examples.isdisjoint(training_examples)
    assert test_seed == 20


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_noisy_vision_rejects_invalid_flip_probabilities(probability):
    with pytest.raises(ValueError, match="pixel_flip_probability"):
        make_noisy_vision_set(make_patterns(), seed=0, pixel_flip_probability=probability)


def test_episode_reset_clears_transient_state_without_changing_weights():
    class Brain:
        def __init__(self):
            self.X = np.full((2, 2), 0.4)
            self.reset_called = False

        def reset(self, keep_weights):
            assert keep_weights is True
            self.reset_called = True

    agent = object.__new__(NeuromorphicAgent)
    agent.brain = Brain()
    agent._ep_step = 8
    agent._E_accum = np.ones((2, 2))
    agent._episode_scores = []
    agent._last = {"action": 1}
    initial_weights = agent.brain.X.copy()

    agent.reset_episode_state()

    assert agent.brain.reset_called
    assert agent._ep_step == 0
    assert not agent._E_accum.any()
    assert agent._last == {}
    assert np.array_equal(agent.brain.X, initial_weights)


def _make_minimal_reinforce_agent(track_policy_gradients):
    class Encoder:
        V_read = 1.0
        encoding = "random"

        def encode(self, _obs):
            return np.array([1.0, 0.0])

        def last_pre_spikes(self):
            return np.array([True, False])

    class Brain:
        N = 2
        M = 2

        def step(self, *_args, **_kwargs):
            return np.array([True, False]), np.array([0.4, 0.1]), np.array([0]), 0.0

    class Decoder:
        def __init__(self):
            self.col = 0
            self.rng = np.random.default_rng(0)

        def decode(self, *_args, **_kwargs):
            self.col = 0
            return 0

        def last_col(self):
            return self.col

        def last_probs(self):
            return np.array([0.75, 0.25])

        def last_behavior_probs(self):
            return np.array([0.75, 0.25])

    agent = object.__new__(ReinforceAgent)
    agent.encoder = Encoder()
    agent.brain = Brain()
    agent.decoder = Decoder()
    agent.track_policy_gradients = track_policy_gradients
    agent.steps_per_action = 1
    agent.dt = 0.01
    agent.k_wta = 1
    agent.wta_temperature = 0.0
    agent.epsilon = 0.0
    agent.gamma = 0.99
    agent._E_accum = np.zeros((2, 2))
    agent._episode_scores = []
    agent._ep_step = 0
    agent._last = {}
    return agent


def test_reinforce_agent_can_skip_unused_policy_gradient_tracking():
    agent = _make_minimal_reinforce_agent(track_policy_gradients=False)

    agent.act(np.zeros(4))

    assert agent._episode_scores == []
    assert not agent._E_accum.any()
    assert agent.last_step()["score"] is None


def test_reinforce_agent_tracks_policy_gradients_by_default():
    agent = _make_minimal_reinforce_agent(track_policy_gradients=True)

    agent.act(np.zeros(4))

    assert len(agent._episode_scores) == 1
    assert np.any(agent._E_accum)
    assert agent.last_step()["score"] is not None


def test_rstdp_credits_decoded_action_independently_of_wta_winners():
    class Brain:
        N = 2
        M = 2

        def __init__(self):
            self.substep = 0

        def step(self, *_args, **_kwargs):
            outcomes = (
                (np.array([True, False]), np.array([1])),
                (np.array([False, True]), np.array([0])),
            )
            spikes, winners = outcomes[self.substep]
            self.substep += 1
            return spikes, np.array([0.4, 0.1]), winners, 0.0

    agent = object.__new__(RSTDPAgent)
    base = _make_minimal_reinforce_agent(track_policy_gradients=False)
    agent.__dict__.update(base.__dict__)
    agent.brain = Brain()
    agent.steps_per_action = 2
    agent.a_plus = 0.01
    agent.a_minus = 0.01
    agent.eligibility = np.zeros((2, 2))
    agent._eligibility_history = []
    agent._episode_event_counts = {
        "pre_events": 0,
        "post_spikes": 0,
        "wta_post_spikes": 0,
        "selected_post_events": 0,
        "selected_action_wta_spikes": 0,
        "wta_posts_not_selected": 0,
    }

    agent.act(np.zeros(4))

    assert agent.last_step()["spikes"].all()
    assert agent.last_step()["wta_winners"].all()
    assert agent.last_step()["rstdp_post_credit"].tolist() == [True, False]
    assert agent.eligibility[0, 0] > 0.0
    assert agent.eligibility[0, 1] < 0.0
    assert agent._episode_event_counts["selected_post_events"] == 1
    assert agent._episode_event_counts["selected_action_wta_spikes"] == 0


def _make_minimal_rstdp_agent_for_episode_update():
    class Brain:
        def __init__(self):
            self.X = np.full((2, 2), 0.5)
            self.G = np.ones((2, 2))
            self.update_called = False

        def apply_learning_update(self, delta_x):
            self.update_called = True
            self.X = np.clip(self.X + delta_x, 0.0, 1.0)
            return delta_x

        def reset(self, keep_weights):
            assert keep_weights is True

    agent = object.__new__(RSTDPAgent)
    agent.brain = Brain()
    agent.baseline = 100.0
    agent.reward_history = deque([100.0], maxlen=200)
    agent._episode_rewards = [1.0]
    agent._eligibility_history = [np.ones((2, 2))]
    agent.gamma = 0.99
    agent._return_baseline_by_step = {0: 0.0}
    agent.reward_modulated_eligibility = np.zeros((2, 2))
    agent.eligibility = np.ones((2, 2))
    agent.eta = 0.2
    agent.max_abs_delta_x = 0.02
    agent.return_baseline_alpha = 0.05
    agent._episode_event_counts = {
        "pre_events": 1,
        "post_spikes": 0,
        "wta_post_spikes": 0,
        "selected_post_events": 1,
        "selected_action_wta_spikes": 0,
        "wta_posts_not_selected": 0,
    }
    agent._E_accum = np.zeros((2, 2))
    agent._episode_scores = []
    agent._ep_step = 1
    agent._last = {}
    agent.stats = {"dX_abs_mean": [], "signal": [], "R_ep": []}
    return agent


def test_rstdp_freezes_updates_at_the_solved_return_threshold():
    agent = _make_minimal_rstdp_agent_for_episode_update()
    initial_x = agent.brain.X.copy()

    metrics = agent.finish_episode(500.0, freeze_update_above=430.0)

    assert metrics["update_frozen"]
    assert not metrics["update_applied"]
    assert not agent.brain.update_called
    assert np.array_equal(agent.brain.X, initial_x)


def test_rstdp_keeps_learning_below_the_solved_return_threshold():
    agent = _make_minimal_rstdp_agent_for_episode_update()

    metrics = agent.finish_episode(200.0, freeze_update_above=430.0)

    assert not metrics["update_frozen"]
    assert metrics["update_applied"]
    assert agent.brain.update_called


def test_rstdp_one_step_baseline_uses_unscaled_reward_advantage():
    agent = _make_minimal_rstdp_agent_for_episode_update()
    agent._eligibility_history = [np.full((2, 2), 1e-3)]

    metrics = agent.finish_episode(1.0, baseline_override=0.2)

    assert metrics["baseline_override"] == 0.2
    assert metrics["requested_delta_x_abs_mean"] < agent.max_abs_delta_x
