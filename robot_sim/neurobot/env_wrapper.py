"""
Wrapper mínimo sobre CartPole (Gymnasium con fallback a Gym).
Expone: reset() -> obs, step(action) -> (obs, r, done, truncated, info)
"""
import numpy as np

try:
    import gymnasium as gym
    _GYM = "gymnasium"
except ImportError:
    import gym
    _GYM = "gym"


class CartPoleWrapper:
    def __init__(self, normalize: bool = True, max_steps: int = 500, seed: int = 0, death_penalty: float = 10.0):
        self.env = gym.make("CartPole-v1")
        
        # --- MEJORA FÍSICA DEL PÉNDULO ---
        # Hacemos el péndulo más pesado, más largo y el paso de simulación más fino
        self.env.unwrapped.masscart = 1.5       # Carrito más pesado (antes 1.0)
        self.env.unwrapped.masspole = 0.5       # Poste más pesado (antes 0.1)
        self.env.unwrapped.length = 0.8         # Poste más largo (antes 0.5)
        self.env.unwrapped.tau = 0.015          # Simulación más fina/suave (antes 0.02)
        
        # Recalcular variables dependientes internas de CartPole
        self.env.unwrapped.total_mass = self.env.unwrapped.masspole + self.env.unwrapped.masscart
        self.env.unwrapped.polemass_length = self.env.unwrapped.masspole * self.env.unwrapped.length
        
        self.normalize = normalize
        self.max_steps = max_steps
        self.death_penalty = death_penalty
        self.obs_dim = self.env.observation_space.shape[0]   # = 4
        self.n_actions = 12  # 12 columnas: 6 Izquierda, 6 Derecha (Excelente resolución y rápida convergencia)
        self._step_count = 0
        self._seed = seed

        # 6 Niveles de Fuerza por dirección: espectro de 1.0 N a 30.0 N
        magnitudes = list(np.linspace(1.0, 30.0, 6))
        self._forces = magnitudes + magnitudes  # Indices 0-5 (Izq), 6-11 (Der)

        # Media/desv aproximadas de CartPole para normalizar
        self._mean = np.array([0.0, 0.0, 0.0, 0.0])
        self._std  = np.array([2.5, 2.5, 0.2, 2.5])
        self._current_force = 0.0  # Inercia del motor: iniciada aquí

    def reset(self):
        out = self.env.reset(seed=self._seed)
        self._seed = None  # Sólo inicializar la semilla la primera vez
        obs = out[0] if isinstance(out, tuple) else out
        self._step_count = 0
        self._current_force = 0.0  # El motor se detiene al resetear
        return self._obs(obs)

    def step(self, action: int):
        # Mapeo de accion a direccion y fuerza deseada (Target)
        action = int(action)
        if action < 0 or action >= self.n_actions:
            action = int(np.clip(action, 0, self.n_actions - 1))
            
        # Determinar la fuerza objetivo y su dirección matemática real
        # Columnas 0..5 = Izquierda (negativo), 6..11 = Derecha (positivo)
        half = self.n_actions // 2
        mag = self._forces[action]
        target_force = mag if action >= half else -mag
        
        # INERCIA DEL ACTUADOR (Motor Slew Rate)
        # 50% momentum del motor anterior + 50% de la nueva orden para mejor respuesta.
        self._current_force = 0.5 * self._current_force + 0.5 * target_force
        
        # Derivar dirección Y magnitud del valor continuo con signo.
        # CartPole usa: force = force_mag * (1 si action==1, -1 si action==0)
        # Por eso usamos abs() para force_mag y extraemos la dirección del signo.
        gym_action = 1 if self._current_force >= 0.0 else 0
        self.env.unwrapped.force_mag = abs(self._current_force)
        out = self.env.step(gym_action)
        if len(out) == 5:
            obs, r, term, trunc, info = out
            done = term or trunc
            if term:
                r = -self.death_penalty
            elif trunc:
                r = 0.0
            else:
                # Penalizar el desvío del centro (x = obs[0]) para evitar que se deslice hacia un lado
                r = 1.0 - 0.5 * (abs(obs[0]) / 2.4)
        else:
            obs, r, done, info = out
            if done:
                r = -self.death_penalty
            else:
                r = 1.0 - 0.5 * (abs(obs[0]) / 2.4)
                
        self._step_count += 1
        if self._step_count >= self.max_steps:
            done = True
        if isinstance(info, dict):
            info = dict(info)
        else:
            info = {}
        info["obs_raw"] = np.asarray(obs, dtype=float).copy()

        return self._obs(obs), float(r), bool(done), info

    def _obs(self, obs):
        obs = np.asarray(obs, dtype=float)
        if self.normalize:
            obs = (obs - self._mean) / self._std
        return np.clip(obs, -3.0, 3.0)

    def close(self):
        self.env.close()


BACKEND = _GYM
