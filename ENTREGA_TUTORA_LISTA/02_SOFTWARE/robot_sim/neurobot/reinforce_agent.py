"""Agente REINFORCE que usa el crossbar como matriz de pesos.

La política combina una distribución softmax, π, con exploración uniforme:
    q(a|s) = (1 - ε) π(a|s) + ε / M.

Para la acción elegida, el score de la política es:
    score[i,j] = pre[i] · P(rama softmax | a) · (1[j = a] - π[j])

Hay dos métodos de actualización. `learn` usa un retorno por episodio y una
elegibilidad acumulada con decaimiento γ. `learn_from_rewards` usa los retornos
descontados de cada paso y un baseline exponencial por posición temporal. Ambos
son variantes de REINFORCE; ninguno implementa STDP entre disparos.
"""
from collections import deque
from typing import Optional

import numpy as np

from neurobot.encoder import SensorEncoder
from neurobot.decoder import ActionDecoder
from neurobot.crossbar_brain import CrossbarBrain


class NeuromorphicAgent:
    def __init__(self,
                 obs_dim: int,
                 n_actions: int,
                 N: int = 8,
                 M: int = 4,
                 V_read: float = 0.05,
                 dt: float = 1e-4,
                 k_wta: int = 1,
                 wta_temperature: float = 0.0,
                 steps_per_action: int = 1,
                 seed: int = 0,
                 encoder_mode: str = "random",
                 eta: float = 1e-3,
                 temp_init: float = 1.0,
                 temp_min: float = 0.3,
                 temp_decay: float = 0.999,
                 baseline_window: int = 50,
                 epsilon: float = 0.0,
                 gamma: float = 0.99,
                 json_memristor_nv: str = "last_session.json",
                 json_memristor_v: str = "memristor_volatile.json",
                 json_lif: str = "lif_config.json",
                 initial_X: Optional[float] = None,
                 d2d_enabled: bool = False,
                 c2c_enabled: bool = False,
                 d2d_sigma: float = 0.08,
                 c2c_sigma: float = 0.05,
                 c2c_theta: float = 1.0,
                 c2c_dt: float = 0.02,
                 record_history: bool = False,
                 crossbar_seed: Optional[int] = None,
                 encoder_seed: Optional[int] = None,
                 action_seed: Optional[int] = None,
                 track_policy_gradients: bool = True):
        
        crossbar_seed = seed if crossbar_seed is None else int(crossbar_seed)
        encoder_seed = seed if encoder_seed is None else int(encoder_seed)
        action_seed = seed if action_seed is None else int(action_seed)
        self.seed_config = {
            "crossbar": crossbar_seed,
            "encoder": encoder_seed,
            "actions": action_seed,
        }
        self._rng = np.random.default_rng(action_seed)

        # El crossbar almacena los pesos sinápticos no volátiles.
        self.brain = CrossbarBrain(N=N, M=M, seed=crossbar_seed, scheme="V2",
                                   json_memristor_nv=json_memristor_nv,
                                   json_memristor_v=json_memristor_v,
                                   json_lif=json_lif,
                                   initial_X=initial_X,
                                   d2d_enabled=d2d_enabled,
                                   c2c_enabled=c2c_enabled,
                                   d2d_sigma=d2d_sigma,
                                   c2c_sigma=c2c_sigma,
                                   c2c_theta=c2c_theta,
                                   c2c_dt=c2c_dt,
                                   record_history=record_history)
        
        self.encoder = SensorEncoder(obs_dim=obs_dim, N=N,
                                     V_read=V_read, seed=encoder_seed,
                                     encoding=encoder_mode)
        self.decoder = ActionDecoder(n_actions=n_actions, M=M,
                                     sample=True,
                                     temperature=temp_init,
                                     rng=self._rng)

        self.n_actions = int(n_actions)
        self.dt = float(dt)
        self.k_wta = int(k_wta)
        self.wta_temperature = max(float(wta_temperature), 0.0)
        self.steps_per_action = int(steps_per_action)

        self.eta = float(eta)
        self.temp_init = float(temp_init)
        self.temp_min = float(temp_min)
        self.temp_decay = float(temp_decay)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.track_policy_gradients = bool(track_policy_gradients)

        self.reward_history = deque(maxlen=int(baseline_window))
        self.baseline = 0.0
        self.episode_count = 0
        self._ep_step = 0
        self._E_accum = np.zeros((N, M))
        self._episode_scores = []
        self._return_baseline_by_step = {}

        # Diagnóstico
        self.stats = {
            "dX_abs_mean": [],
            "signal": [],
            "R_ep": [],
        }

        self._last = {}

    def act(self, obs, exploration_epsilon: float = 0.0):
        """Procesa una observación y conserva el score de la acción elegida."""
        V_rows = self.encoder.encode(obs)
        pre = self.encoder.last_pre_spikes()

        # Acumula la respuesta del crossbar cuando se usan varios pasos por acción.
        spikes_acc = np.zeros(self.brain.M, dtype=bool)
        wta_winners_acc = np.zeros(self.brain.M, dtype=bool)
        # Keep individual neural substeps and their action-level unions.
        spikes_by_substep = np.zeros(
            (self.steps_per_action, self.brain.M), dtype=bool
        )
        wta_winners_by_substep = np.zeros_like(spikes_by_substep)
        I_acc = np.zeros(self.brain.M)
        sneak_acc = 0.0
        winners = np.array([], dtype=int)

        for substep in range(self.steps_per_action):
            spikes, I_col, winners, sneak = self.brain.step(
                V_rows, dt=self.dt, k_wta=self.k_wta,
                wta_temperature=self.wta_temperature,
                rng=self.decoder.rng,
            )
            spikes_by_substep[substep] = spikes
            spikes_acc |= spikes
            if winners.size:
                wta_winners_acc[winners] = True
                wta_winners_by_substep[substep, winners] = True
            I_acc += I_col
            sneak_acc += sneak

        exploration_epsilon = 1.0 - (
            (1.0 - float(exploration_epsilon)) * (1.0 - self.epsilon)
        )
        action = self._select_action(
            I_acc, exploration_epsilon, wta_winners_acc
        )
        col = self.decoder.last_col()
        probs = self.decoder.last_probs()
        behavior_probs = self.decoder.last_behavior_probs()

        score = None
        if self.track_policy_gradients:
            # Score de la política de comportamiento para la acción seleccionada.
            indicator = np.zeros(self.brain.M)
            indicator[col] = 1.0
            # Si la acción salió de la rama aleatoria, su gradiente de softmax es
            # cero; el factor siguiente calcula la probabilidad posterior de la
            # rama softmax dado que se observó esta acción.
            posterior_policy = (
                (1.0 - exploration_epsilon) * probs[col]
                / max(float(behavior_probs[col]), 1e-15)
            )
            delta = posterior_policy * (indicator - probs)
            # Keep feature magnitude in REINFORCE's eligibility so opposite and
            # stronger sensor contexts do not collapse to the same binary mask.
            policy_features = np.clip(
                V_rows / max(self.encoder.V_read, 1e-12), -1.0, 1.0
            )
            score = np.outer(policy_features, delta)
            self._episode_scores.append(score)

            # La elegibilidad más antigua pierde peso antes de acumular este paso.
            self._E_accum *= self.gamma
            if self.encoder.encoding == "vision_bipolar":
                eligibility_features = np.clip(
                    V_rows / max(self.encoder.V_read, 1e-12), -1.0, 1.0
                )
            else:
                eligibility_features = pre.astype(float)
            self._E_accum += np.outer(eligibility_features, delta)
        self._ep_step += 1
        
        self._last = {
            "V_rows": V_rows,
            "pre": pre,
            "spikes": spikes_acc,
            "spikes_by_substep": spikes_by_substep,
            "winners": winners,
            "wta_winners": wta_winners_acc,
            "wta_winners_by_substep": wta_winners_by_substep,
            "I_col": I_acc,
            "sneak": sneak_acc / self.steps_per_action,
            "action": action,
            "col": col,
            "probs": probs,
            "behavior_probs": behavior_probs,
            "exploration_epsilon": exploration_epsilon,
            "score": score,
        }
        return action

    def _select_action(self, I_col, exploration_epsilon, wta_winners):
        """Selector por defecto de REINFORCE: softmax sobre todas las columnas."""
        return self.decoder.decode(
            I_col, exploration_epsilon=exploration_epsilon
        )

    def learn(self, R_ep: float, freeze_update_above: Optional[float] = None,
              baseline_override: Optional[float] = None):
        """Actualiza los pesos usando el retorno total y la elegibilidad del episodio."""
        R_ep = float(R_ep)

        # El baseline se calcula con episodios anteriores.
        if baseline_override is None:
            signal = R_ep - self.baseline if self.reward_history else 0.0
        else:
            # A baseline conditioned on the observed input is valid for
            # REINFORCE and lowers variance in this repeated-pattern task.
            signal = R_ep - float(baseline_override)
        self.reward_history.append(R_ep)
        self.baseline = float(np.mean(self.reward_history))
        self._last_signal = signal

        # Actualización episódica con la elegibilidad acumulada.
        dX = -self.eta * self._E_accum * signal
        if freeze_update_above is not None and R_ep >= freeze_update_above:
            dX.fill(0.0)
        
        # El escalado global limita el tamaño del paso sin cambiar su dirección.
        max_grad = float(np.max(np.abs(dX)))
        if max_grad > 0.05:
            dX = dX * (0.05 / max_grad)
        
        if np.any(dX):
            dX = self.brain.apply_learning_update(dX)
        
        # Registrar magnitudes de la actualización y retorno.
        self.stats["dX_abs_mean"].append(float(np.abs(dX).mean()))
        self.stats["signal"].append(float(signal))
        self.stats["R_ep"].append(R_ep)

    def learn_from_rewards(self, rewards):
        """Actualiza los pesos con retornos descontados y baseline por paso.

        `act` guarda el score ∇log q(a_t|s_t) de cada decisión. El baseline
        asociado a cada posición temporal se actualiza al terminar el episodio.
        """
        rewards = np.asarray(rewards, dtype=float).reshape(-1)
        if rewards.size == 0:
            raise ValueError("rewards must contain at least one environment step")
        if len(self._episode_scores) != rewards.size:
            raise ValueError(
                "one policy score is required for every environment action: "
                f"got {len(self._episode_scores)} scores for {rewards.size} rewards"
            )

        returns = np.empty(rewards.size, dtype=float)
        future = 0.0
        for step in range(rewards.size - 1, -1, -1):
            future = rewards[step] + self.gamma * future
            returns[step] = future

        advantages = np.asarray([
            returns[step] - self._return_baseline_by_step.get(step, 0.0)
            for step in range(rewards.size)
        ])
        advantage_scale = max(float(np.std(advantages)), 1e-8)
        normalized_advantages = advantages / advantage_scale
        grad = np.zeros_like(self._E_accum)
        for score, advantage in zip(self._episode_scores, normalized_advantages):
            grad += score * advantage
        grad /= np.sqrt(rewards.size)

        # Normalizar por longitud del episodio y limitar el tamaño del paso.
        dX = -self.eta * grad
        max_weight_update = float(
            getattr(self, "max_weight_update", 0.01)
        )
        if max_weight_update <= 0.0:
            raise ValueError("max_weight_update must be positive")
        max_grad = float(np.max(np.abs(dX)))
        if max_grad > max_weight_update:
            dX *= max_weight_update / max_grad
        if np.any(dX):
            dX = self.brain.apply_learning_update(dX)

        # Actualizar el baseline después de calcular las ventajas para que el
        # episodio actual no se use como referencia de sí mismo.
        alpha = 0.05
        for step, value in enumerate(returns):
            previous = self._return_baseline_by_step.get(step)
            self._return_baseline_by_step[step] = (
                float(value) if previous is None
                else (1.0 - alpha) * previous + alpha * float(value)
            )

        total_return = float(rewards.sum())
        self.reward_history.append(total_return)
        self.baseline = float(np.mean(self.reward_history))
        self._last_signal = float(np.mean(advantages))
        self.stats["dX_abs_mean"].append(float(np.abs(dX).mean()))
        self.stats["signal"].append(self._last_signal)
        self.stats["R_ep"].append(total_return)
        return {"returns": returns, "advantages": advantages, "dX": dX}

    def end_episode(self):
        """Se llama tras learn(). Decae temperatura y reinicia el estado neuronal."""
        self.episode_count += 1
        
        # Una señal muy negativa recupera ligeramente la temperatura de exploración.
        T_curr = self.decoder.temperature
        if (hasattr(self, '_last_signal') and self._last_signal < -30
                and T_curr < 0.10):
            T = min(0.10, T_curr + 0.005)
        else:
            T = max(self.temp_min, T_curr * self.temp_decay)
            
        self.decoder.set_temperature(T)
        self.reset_episode_state()

    def reset_episode_state(self):
        """Reinicia la actividad y elegibilidad sin modificar los pesos."""
        self._ep_step = 0
        self.brain.reset(keep_weights=True)
        self._E_accum.fill(0.0)
        self._episode_scores.clear()
        self._last = {}

    def last_step(self):
        return self._last

    def current_temperature(self):
        return self.decoder.temperature

    def act_deterministic(self, obs):
        """Choose the greedy action for evaluation without sampling."""
        sample = self.decoder.sample
        self.decoder.sample = False
        try:
            return self.act(obs)
        finally:
            self.decoder.sample = sample
