"""
Codificación sensor → voltajes de fila del crossbar.
Dos modos: 'rate' (proporcional) y 'spikes' (umbralizado).
"""
import numpy as np


class SensorEncoder:
    """
    Traduce observación o ∈ ℝ^d a V_rows ∈ ℝ^N acotado a [-V_read, +V_read].
    La matriz de proyección W_in aprende por separado (si se desea);
    por defecto es fija y aleatoria pequeña.
    """

    def __init__(self,
                 obs_dim: int,
                 N: int,
                 V_read: float = 0.1,
                 pre_topk: float = 0.4,
                 pre_threshold: float = 0.3,
                 mode: str = "rate",
                 seed: int = 0):
        if mode not in ("rate", "spikes"):
            raise ValueError("mode debe ser 'rate' o 'spikes'")

        self.obs_dim = int(obs_dim)
        self.N = int(N)
        self.V_read = float(V_read)
        self.pre_topk = float(pre_topk)
        self.pre_threshold = float(pre_threshold)
        self.mode = mode

        rng = np.random.default_rng(seed)
        # Proyección pequeña: no saturar la LIF al inicio
        self.W_in = rng.normal(0.0, 1.0 / np.sqrt(obs_dim), (obs_dim, N))

        # Última codificación (para STDP: qué filas estuvieron "activas")
        self._last_V = np.zeros(N)
        self._last_pre = np.zeros(N, dtype=bool)

    # -----------------------------------------------------------------
    def encode(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=float).ravel()
        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"obs dim {obs.shape[0]} != {self.obs_dim}")

        z = self.W_in.T @ obs                 # (N,)
        V = self.V_read / (1.0 + np.exp(-4.0 * z))

        # Top-k por magnitud → siempre k filas activas, distintas por contexto
        k = max(1, int(round(self.pre_topk * self.N)))
        idx = np.argsort(-np.abs(V))[:k]
        pre = np.zeros(self.N, dtype=bool)
        pre[idx] = True

        self._last_V = V
        self._last_pre = pre
        return V

    # -----------------------------------------------------------------
    def last_pre_spikes(self) -> np.ndarray:
        """Máscara booleana de filas activas (para STDP en el brain)."""
        return self._last_pre.copy()

    def last_V(self) -> np.ndarray:
        return self._last_V.copy()
