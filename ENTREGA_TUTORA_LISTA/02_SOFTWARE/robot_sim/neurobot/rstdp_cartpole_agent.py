"""Agente CartPole con plasticidad pre/post modulada por retorno futuro.

Reutiliza el encoder, crossbar, memristor volátil y LIF del agente neuromórfico.
La elegibilidad local asocia las entradas activas con la acción elegida y sus
alternativas competitivas. Se conserva una traza por decisión; al cerrar el
episodio, el retorno futuro descontado asigna crédito y programa el crossbar.
"""
from collections import deque

import numpy as np

from neurobot.reinforce_agent import NeuromorphicAgent as BaseAgent


class NeuromorphicAgent(BaseAgent):
    def __init__(
        self,
        *args,
        a_plus: float = 0.01,
        a_minus: float = 0.01,
        reward_baseline_window: int = 200,
        return_baseline_alpha: float = 0.05,
        max_abs_delta_x: float = 0.02,
        **kwargs,
    ):
        # Match the shared agent's current-based action decoder. WTA remains
        # part of the neural dynamics, but must not gate the policy's actions.
        kwargs.setdefault("k_wta", 3)
        kwargs.setdefault("wta_temperature", 1.0)
        kwargs.setdefault("track_policy_gradients", False)
        super().__init__(*args, **kwargs)

        self.a_plus = float(a_plus)
        self.a_minus = float(a_minus)
        self.return_baseline_alpha = float(return_baseline_alpha)
        self.max_abs_delta_x = float(max_abs_delta_x)
        if min(self.return_baseline_alpha,
               self.max_abs_delta_x) <= 0.0:
            raise ValueError("R-STDP baseline and update limits must be positive")
        if self.return_baseline_alpha > 1.0:
            raise ValueError("return_baseline_alpha must be <= 1")
        if self.a_plus < 0.0 or self.a_minus < 0.0:
            raise ValueError("STDP amplitudes must be nonnegative")

        self.eligibility = np.zeros((self.brain.N, self.brain.M), dtype=float)
        self.reward_modulated_eligibility = np.zeros_like(self.eligibility)
        self._eligibility_history: list[np.ndarray] = []
        self.reward_history = deque(maxlen=int(reward_baseline_window))
        self._episode_rewards: list[float] = []
        self.baseline = 0.0
        self.last_learning_metrics = {
            "advantage": 0.0,
            "eligibility_abs_mean": 0.0,
            "eligibility_abs_max": 0.0,
            "requested_delta_x_abs_mean": 0.0,
            "actual_delta_x_abs_mean": 0.0,
            "actual_delta_x_abs_max": 0.0,
            "pre_events": 0,
            "post_spikes": 0,
            "wta_post_spikes": 0,
            "selected_post_events": 0,
            "selected_action_wta_spikes": 0,
            "wta_posts_not_selected": 0,
        }
        self._episode_event_counts = {
            "pre_events": 0,
            "post_spikes": 0,
            "wta_post_spikes": 0,
            "selected_post_events": 0,
            "selected_action_wta_spikes": 0,
            "wta_posts_not_selected": 0,
        }

    def _select_action(self, I_col, exploration_epsilon, wta_winners):
        return self.decoder.decode(
            I_col,
            exploration_epsilon=exploration_epsilon,
        )

    def act(self, obs, exploration_epsilon: float = 0.0):
        action = super().act(obs, exploration_epsilon=exploration_epsilon)
        record = self.last_step()
        pre = np.asarray(record["pre"], dtype=bool)
        spikes_by_substep = np.asarray(record["spikes_by_substep"], dtype=bool)
        winners_by_substep = np.asarray(
            record["wta_winners_by_substep"], dtype=bool
        )
        wta_events = spikes_by_substep & winners_by_substep
        wta_spikes = np.any(wta_events, axis=0)
        selected_post = np.zeros(self.brain.M, dtype=bool)
        selected_col = int(record["col"])
        # The decoded action is the postsynaptic decision event. Center it by
        # the policy probabilities so selected and competing actions receive
        # opposite eligibility, as in a reward-modulated policy update.
        selected_post[selected_col] = True
        post_event = selected_post.astype(float) - np.asarray(
            record["probs"], dtype=float
        )

        # Keep a compact local pre/post eligibility for this decision. Centering
        # the postsynaptic event gives unselected actions negative credit instead
        # of reinforcing only the sampled action regardless of its alternatives.
        self.eligibility.fill(0.0)
        if self.encoder.encoding == "vision_bipolar":
            current = np.asarray(record["I_col"], dtype=float)
            centered = current - float(np.mean(current))
            sigma = float(np.std(current))
            scale = sigma + 1e-8
            probs = np.asarray(record["probs"], dtype=float)
            posterior = (
                (1.0 - float(record["exploration_epsilon"])) * probs[selected_col]
                / max(float(record["behavior_probs"][selected_col]), 1e-15)
            )
            action_score = -posterior * probs
            action_score[selected_col] += posterior
            dlogp_dcurrent = action_score / scale
            if sigma > 1e-15:
                dlogp_dcurrent -= (
                    centered * float(np.dot(action_score, centered))
                    / (self.brain.M * sigma * scale * scale)
                )
            delta_r = self.brain._cell_r_off - self.brain._cell_r_on
            minus_dcurrent_dX = (
                np.asarray(record["V_rows"], dtype=float)[:, None]
                * delta_r
                * self.brain.G**2
                * max(1, self.steps_per_action)
            )
            post_amplitudes = np.where(
                dlogp_dcurrent >= 0.0, self.a_plus, self.a_minus
            )
            self.eligibility += (
                minus_dcurrent_dX
                * dlogp_dcurrent[None, :]
                * post_amplitudes[None, :]
            )
        else:
            self.eligibility += self.a_plus * np.outer(
                pre, np.maximum(post_event, 0.0)
            )
            self.eligibility += self.a_minus * np.outer(
                pre, np.minimum(post_event, 0.0)
            )
        self._eligibility_history.append(self.eligibility.copy())

        self._episode_event_counts["pre_events"] += int(np.count_nonzero(pre))
        self._episode_event_counts["post_spikes"] += int(
            np.count_nonzero(spikes_by_substep)
        )
        self._episode_event_counts["wta_post_spikes"] += int(
            np.count_nonzero(wta_events)
        )
        self._episode_event_counts["selected_post_events"] += int(np.count_nonzero(selected_post))
        self._episode_event_counts["selected_action_wta_spikes"] += int(
            np.count_nonzero(wta_events[:, selected_col])
        )
        self._episode_event_counts["wta_posts_not_selected"] += int(
            np.count_nonzero(wta_events & ~selected_post[None, :])
        )
        record["rstdp_post_credit"] = selected_post
        return action

    def observe_reward(self, step_reward: float) -> float:
        """Store the reward aligned with the latest action/eligibility trace."""
        reward = float(step_reward)
        self._episode_rewards.append(reward)
        return reward

    def finish_episode(
        self,
        episode_return: float,
        freeze_update_above: float | None = None,
        baseline_override: float | None = None,
    ) -> dict:
        """Apply reward-modulated eligibility, optionally freezing solved episodes."""
        episode_return = float(episode_return)
        baseline_before = float(self.baseline) if self.reward_history else episode_return
        episode_advantage = episode_return - baseline_before

        rewards = np.asarray(self._episode_rewards, dtype=float)
        if len(self._eligibility_history) != rewards.size:
            raise RuntimeError(
                "R-STDP credit alignment error: expected one eligibility snapshot "
                f"per reward, got {len(self._eligibility_history)} traces for {rewards.size} rewards"
            )
        returns = np.empty(rewards.size, dtype=float)
        future = 0.0
        for t in range(rewards.size - 1, -1, -1):
            future = rewards[t] + self.gamma * future
            returns[t] = future
        if baseline_override is None:
            baselines = np.asarray([
                self._return_baseline_by_step.get(t, 0.0)
                for t in range(rewards.size)
            ], dtype=float)
        else:
            baselines = np.full(rewards.size, float(baseline_override))
        advantages = returns - baselines
        advantage_scale = (
            max(float(np.std(advantages)), 1e-8)
            if rewards.size > 1 else 1.0
        )
        normalized_advantages = advantages / advantage_scale

        self.reward_modulated_eligibility.fill(0.0)
        for eligibility_t, advantage_t in zip(
            self._eligibility_history, normalized_advantages
        ):
            self.reward_modulated_eligibility += eligibility_t * advantage_t
        if rewards.size:
            self.reward_modulated_eligibility /= np.sqrt(rewards.size)

        delta_x = -self.eta * self.reward_modulated_eligibility
        max_delta = float(np.max(np.abs(delta_x))) if delta_x.size else 0.0
        if max_delta > self.max_abs_delta_x:
            delta_x *= self.max_abs_delta_x / max_delta
        update_frozen = (
            freeze_update_above is not None
            and episode_return >= float(freeze_update_above)
        )
        if update_frozen:
            delta_x.fill(0.0)

        x_before = self.brain.X.copy()
        g_before = self.brain.G.copy()
        eligibility_mean = float(self.reward_modulated_eligibility.mean())
        positive_e = int(np.count_nonzero(self.reward_modulated_eligibility > 0.0))
        negative_e = int(np.count_nonzero(self.reward_modulated_eligibility < 0.0))
        if np.any(delta_x):
            self.brain.apply_learning_update(delta_x)
        actual_delta = self.brain.X - x_before
        actual_delta_g = self.brain.G - g_before

        # Update the temporal return baseline only after this episode's update,
        # so the trajectory is compared with prior experience.
        for t, value in enumerate(returns):
            previous = self._return_baseline_by_step.get(t)
            self._return_baseline_by_step[t] = (
                float(value) if previous is None
                else (1.0 - self.return_baseline_alpha) * previous
                + self.return_baseline_alpha * float(value)
            )
        self.reward_history.append(episode_return)
        self.baseline = float(np.mean(self.reward_history))
        self.last_learning_metrics = {
            "episode_return_advantage_diagnostic_only": episode_advantage,
            "baseline_before": baseline_before,
            "baseline_after": self.baseline,
            "baseline_override": baseline_override,
            "eligibility_abs_mean": float(np.abs(self.eligibility).mean()),
            "eligibility_abs_max": float(np.abs(self.eligibility).max(initial=0.0)),
            "reward_modulated_eligibility_abs_mean": float(
                np.abs(self.reward_modulated_eligibility).mean()
            ),
            "reward_modulated_eligibility_abs_max": float(
                np.abs(self.reward_modulated_eligibility).max(initial=0.0)
            ),
            "eligibility_signed_mean": eligibility_mean,
            "eligibility_positive_cells": positive_e,
            "eligibility_negative_cells": negative_e,
            "requested_delta_x_abs_mean": float(np.abs(delta_x).mean()),
            "actual_delta_x_abs_mean": float(np.abs(actual_delta).mean()),
            "actual_delta_x_abs_max": float(np.abs(actual_delta).max(initial=0.0)),
            "actual_delta_x_signed_mean_by_column": actual_delta.mean(axis=0).tolist(),
            "actual_delta_x_abs_mean_by_column": np.abs(actual_delta).mean(axis=0).tolist(),
            "reward_modulated_eligibility_abs_mean_by_column": (
                np.abs(self.reward_modulated_eligibility).mean(axis=0).tolist()
            ),
            "G_delta_signed_mean_by_column": actual_delta_g.mean(axis=0).tolist(),
            "update_applied": bool(np.any(delta_x)),
            "update_frozen": update_frozen,
            "freeze_update_above": freeze_update_above,
            "G_mean_before": float(g_before.mean()),
            "G_mean_after": float(self.brain.G.mean()),
            "actual_delta_g_signed_mean": float(actual_delta_g.mean()),
            "actual_delta_g_abs_mean": float(np.abs(actual_delta_g).mean()),
            "delta_x_positive_cells": int(np.count_nonzero(actual_delta > 0.0)),
            "delta_x_negative_cells": int(np.count_nonzero(actual_delta < 0.0)),
            "x_cells_at_zero": int(np.count_nonzero(self.brain.X <= 1e-12)),
            "x_cells_at_one": int(np.count_nonzero(self.brain.X >= 1.0 - 1e-12)),
            "pre_events": int(self._episode_event_counts["pre_events"]),
            "post_spikes": int(self._episode_event_counts["post_spikes"]),
            "wta_post_spikes": int(self._episode_event_counts["wta_post_spikes"]),
            "selected_post_events": int(self._episode_event_counts["selected_post_events"]),
            "selected_action_wta_spikes": int(
                self._episode_event_counts["selected_action_wta_spikes"]
            ),
            "wta_posts_not_selected": int(self._episode_event_counts["wta_posts_not_selected"]),
            "discounted_return_mean": float(returns.mean()) if returns.size else 0.0,
            "return_advantage_mean": float(advantages.mean()) if advantages.size else 0.0,
            "return_advantage_abs_mean": float(np.abs(advantages).mean()) if advantages.size else 0.0,
            "return_advantage_positive_steps": int(np.count_nonzero(advantages > 0.0)),
            "return_advantage_negative_steps": int(np.count_nonzero(advantages < 0.0)),
            "return_advantage_min": float(advantages.min()) if advantages.size else 0.0,
            "return_advantage_max": float(advantages.max()) if advantages.size else 0.0,
            "positive_reward_steps": int(sum(r > 0.0 for r in self._episode_rewards)),
            "negative_reward_steps": int(sum(r < 0.0 for r in self._episode_rewards)),
            "reward_modulated_eligibility_positive_cells": positive_e,
            "reward_modulated_eligibility_negative_cells": negative_e,
        }
        self.stats["dX_abs_mean"].append(self.last_learning_metrics["actual_delta_x_abs_mean"])
        self.stats["signal"].append(float(advantages.mean()) if advantages.size else 0.0)
        self.stats["R_ep"].append(episode_return)
        self.reset_episode_state()
        return dict(self.last_learning_metrics)

    def reset_episode_state(self):
        super().reset_episode_state()
        self.eligibility.fill(0.0)
        self.reward_modulated_eligibility.fill(0.0)
        self._eligibility_history.clear()
        self._episode_rewards.clear()
        for key in self._episode_event_counts:
            self._episode_event_counts[key] = 0
