"""
Ganadores del WTA → acción discreta, con muestreo softmax.
La política es sobre COLUMNAS (M), no sobre acciones.
La acción es un mapeo determinista col → acción.

Expone last_probs() para que el agente construya la elegibilidad REINFORCE.
"""
import numpy as np


class ActionDecoder:
    def __init__(self,
                 n_actions: int,
                 M: int,
                 sample: bool = True,
                 temperature: float = 1.0):
        self.n_actions = int(n_actions)
        self.M = int(M)
        self.sample = bool(sample)
        self.temperature = float(temperature)

        # col j → acción: reparto uniforme
        self.col_to_action = np.array(
            [j % self.n_actions for j in range(self.M)], dtype=int
        )

        self._last_probs = np.full(self.M, 1.0 / self.M)
        self._last_col = 0
        self._last_action = 0

    # ------------------------------------------------------------------
    def decode(self, I_col: np.ndarray) -> int:
        """
        Softmax sobre I_col normalizada internamente.
        Los logits son adimensionales y de orden ~1 antes de dividir por T.
        """
        I_col = np.asarray(I_col, dtype=float)

        # Normalización autoescalante (Z-score) para aumentar el contraste de la política
        std = float(np.std(I_col)) + 1e-8
        logits = (I_col - np.mean(I_col)) / std

        # Temperatura adimensional sobre logits
        T = max(self.temperature, 1e-8)
        logits = logits / T
        logits -= logits.max()

        p = np.exp(logits)
        s = p.sum()
        p = p / s if s > 0 else np.full(self.M, 1.0 / self.M)

        self._last_probs = p

        if self.sample:
            col = int(np.random.choice(self.M, p=p))
        else:
            col = int(np.argmax(p))

        self._last_col = col
        self._last_action = int(self.col_to_action[col])
        return self._last_action

    # ------------------------------------------------------------------
    def last_probs(self) -> np.ndarray:
        """π_j = probabilidad de cada columna en el último decode()."""
        return self._last_probs.copy()

    def last_col(self) -> int:
        return self._last_col

    def set_temperature(self, T: float):
        self.temperature = max(float(T), 1e-3)
