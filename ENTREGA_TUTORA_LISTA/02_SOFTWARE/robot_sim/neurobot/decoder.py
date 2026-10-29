"""
Ganadores del WTA → acción discreta, con muestreo softmax.
La política es sobre COLUMNAS (M), no sobre acciones.
La acción es un mapeo determinista col → acción.

Expone la distribución usada para muestrear una columna; no actualiza pesos.
"""
from typing import Optional

import numpy as np


class ActionDecoder:
    def __init__(self,
                 n_actions: int,
                 M: int,
                 sample: bool = True,
                 temperature: float = 1.0,
                 rng: Optional[np.random.Generator] = None):
        self.n_actions = int(n_actions)
        self.M = int(M)
        self.sample = bool(sample)
        self.temperature = float(temperature)
        self.rng = rng if rng is not None else np.random.default_rng()

        # col j → acción: reparto uniforme
        self.col_to_action = np.array(
            [j % self.n_actions for j in range(self.M)], dtype=int
        )

        self._last_probs = np.full(self.M, 1.0 / self.M)
        self._last_behavior_probs = self._last_probs.copy()
        self._last_col = 0
        self._last_action = 0

    # ------------------------------------------------------------------
    def decode(self, I_col: np.ndarray, exploration_epsilon: float = 0.0,
               allowed_columns: Optional[np.ndarray] = None) -> int:
        """
        Softmax sobre I_col normalizada internamente.
        Los logits son adimensionales y de orden ~1 antes de dividir por T.
        """
        I_col = np.asarray(I_col, dtype=float)
        if I_col.shape != (self.M,):
            raise ValueError(f"I_col debe tener forma ({self.M},), recibido {I_col.shape}")
        if allowed_columns is None:
            allowed = np.ones(self.M, dtype=bool)
        else:
            allowed = np.asarray(allowed_columns, dtype=bool)
            if allowed.shape != (self.M,):
                raise ValueError(
                    f"allowed_columns debe tener forma ({self.M},), recibido {allowed.shape}"
                )
            if not np.any(allowed):
                allowed = np.ones(self.M, dtype=bool)

        # Normalización autoescalante (Z-score) para aumentar el contraste de la política
        std = float(np.std(I_col)) + 1e-8
        logits = (I_col - np.mean(I_col)) / std

        # Temperatura adimensional sobre logits
        T = max(self.temperature, 1e-8)
        logits = logits / T
        logits[~allowed] = -np.inf
        logits -= np.max(logits[allowed])

        p = np.zeros(self.M, dtype=float)
        p[allowed] = np.exp(logits[allowed])
        s = p.sum()
        p = p / s if s > 0 else np.full(self.M, 1.0 / self.M)

        self._last_probs = p.copy()
        epsilon = float(np.clip(exploration_epsilon, 0.0, 1.0))
        behavior = (1.0 - epsilon) * p
        behavior[allowed] += epsilon / int(np.count_nonzero(allowed))
        self._last_behavior_probs = behavior

        if self.sample:
            col = int(self.rng.choice(self.M, p=behavior))
        else:
            col = int(np.argmax(behavior))

        self._last_col = col
        self._last_action = int(self.col_to_action[col])
        return self._last_action

    # ------------------------------------------------------------------
    def last_probs(self) -> np.ndarray:
        """π_j = probabilidad de cada columna en el último decode()."""
        return self._last_probs.copy()

    def last_col(self) -> int:
        return self._last_col

    def last_behavior_probs(self) -> np.ndarray:
        """Probability distribution that actually selected the last column."""
        return self._last_behavior_probs.copy()

    def set_temperature(self, T: float):
        self.temperature = max(float(T), 1e-3)
