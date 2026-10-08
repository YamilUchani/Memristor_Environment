"""
Tareas triviales para verificar la maquinaria de aprendizaje,
sin dependencia de Gym.
"""
import numpy as np


class BanditTask:
    """
    Bandit de 2 brazos sin contexto.
    p = probabilidad de recompensa +1 por brazo.
    Cada step es un episodio completo (done=True siempre).
    """
    def __init__(self, p=(0.2, 0.8), n_actions=2, seed=0):
        self.p = np.asarray(p, dtype=float)
        self.n_actions = int(n_actions)
        self.rng = np.random.default_rng(seed)
        self.obs_dim = 2

    def reset(self):
        return np.array([1.0, 0.0])   # constante, para no anular el encoder

    def step(self, action):
        action = int(action)
        r = 1.0 if self.rng.random() < self.p[action] else -1.0
        return self.reset(), float(r), True, {"action": action}


class ContextualBanditTask:
    """
    Bandit contextual de 1 paso.
    Estado s ∈ {0, 1}. Acción correcta = s.
    Recompensa +1 si acierta, -1 si falla.
    """
    def __init__(self, n_actions=2, seed=0):
        self.n_actions = int(n_actions)
        self.rng = np.random.default_rng(seed)
        self.obs_dim = 2
        self._s = 0

    def reset(self):
        self._s = int(self.rng.integers(0, 2))
        # obs = [1.0, s]  → el encoder recibe una señal discriminativa
        return np.array([1.0, float(self._s)])

    def step(self, action):
        correct = int(action == self._s)
        r = 1.0 if correct else -1.0
        return self.reset(), float(r), True, {"correct": correct, "s": self._s}
