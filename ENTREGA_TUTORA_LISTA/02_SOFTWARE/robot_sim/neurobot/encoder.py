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
                 encoding: str = "random",
                 seed: int = 0):
        if mode not in ("rate", "spikes"):
            raise ValueError("mode debe ser 'rate' o 'spikes'")

        self.obs_dim = int(obs_dim)
        self.N = int(N)
        self.V_read = float(V_read)
        self.pre_topk = float(pre_topk)
        self.pre_threshold = float(pre_threshold)
        self.mode = mode
        if encoding not in ("random", "cartpole_rbf", "cartpole_population",
                            "cartpole_rate_population", "cartpole_opponent_rate",
                            "cartpole_opponent_binary", "obstacle_grid", "auto2d", "identity",
                            "vision_bipolar"):
            raise ValueError(
                "encoding debe ser 'random', 'cartpole_rbf', 'cartpole_population', "
                "'cartpole_rate_population', 'cartpole_opponent_rate', "
                "'cartpole_opponent_binary', "
                "'obstacle_grid', 'auto2d', 'identity' o 'vision_bipolar'"
            )
        if encoding in ("cartpole_rbf", "cartpole_population",
                        "cartpole_rate_population", "cartpole_opponent_rate") and self.obs_dim != 4:
            raise ValueError(f"{encoding} requiere observaciones de 4 dimensiones")
        if encoding in ("cartpole_population", "cartpole_rate_population",
                        "cartpole_opponent_rate") and self.N != 24:
            raise ValueError(f"{encoding} requiere 24 filas: 6 por variable")
        if encoding == "cartpole_opponent_binary" and self.N != 8:
            raise ValueError("cartpole_opponent_binary requiere 8 filas: ON/OFF para 4 sensores")
        if encoding == "obstacle_grid" and (self.obs_dim != 11 or self.N != 16):
            raise ValueError("obstacle_grid requiere obs_dim=11 y N=16")
        if encoding == "auto2d" and (self.obs_dim != 11 or self.N != 24):
            raise ValueError("auto2d requiere 11 sensores y 24 filas")
        if encoding in ("identity", "vision_bipolar") and self.obs_dim != self.N:
            raise ValueError(f"{encoding} requiere obs_dim == N")
        self.encoding = encoding

        rng = np.random.default_rng(seed)
        # Proyección pequeña: no saturar la LIF al inicio
        self.W_in = rng.normal(0.0, 1.0 / np.sqrt(obs_dim), (obs_dim, N))
        # Receptive fields locales y reproducibles: evita que el signo y el
        # contexto de CartPole se pierdan en una proyección aleatoria global.
        if self.encoding == "cartpole_rbf":
            scales = np.array([0.9, 1.0, 0.65, 1.0])
            self._rbf_centers = np.clip(
                rng.normal(0.0, scales[:, None], size=(4, self.N)), -2.5, 2.5
            )
            self._rbf_widths = np.array([0.85, 0.9, 0.55, 0.9])
        if self.encoding in ("cartpole_population", "cartpole_rate_population"):
            # Six ordered, overlapping receptive fields per normalized state
            # variable. Unlike cartpole_rbf, each neuron tunes to one variable
            # rather than a random conjunction of all four variables.
            ranges = np.array([
                [-1.0, 1.0],    # cart position (|x| <= 2.4 m => |x_norm| <= 0.96)
                [-2.0, 2.0],    # cart velocity, normalized
                [-1.05, 1.05],  # pole angle (termination near |theta| = 0.209 rad)
                [-2.5, 2.5],    # pole angular velocity, normalized
            ])
            centers = np.linspace(0.0, 1.0, 6)
            self._population_centers = (
                ranges[:, :1] + centers[None, :] * (ranges[:, 1:] - ranges[:, :1])
            )
            spacing = (ranges[:, 1] - ranges[:, 0]) / 5.0
            self._population_widths = (
                (0.90 if self.encoding == "cartpole_rate_population" else 0.70)
                * spacing
            )
        if self.encoding == "cartpole_opponent_rate":
            # Separate ON/OFF rate channels give every state variable an
            # explicit sign and magnitude representation (3 levels per sign).
            self._opponent_scales = np.array([1.0, 2.0, 1.05, 2.5])
            self._opponent_thresholds = np.array([0.0, 0.33, 0.66])
            self._opponent_baseline = 0.05
        if self.encoding == "cartpole_opponent_binary":
            self._opponent_baseline = 0.05
        if self.encoding == "obstacle_grid":
            # 9 rows preserve the signed local patch, 3+3 rows encode the
            # normalized grid coordinates, and the last row is a bias.
            self._grid_position_centers = np.linspace(0.0, 1.0, 3)
            self._grid_position_width = 0.38

        # Última codificación (para STDP: qué filas estuvieron "activas")
        self._last_V = np.zeros(N)
        self._last_pre = np.zeros(N, dtype=bool)

    # -----------------------------------------------------------------
    def encode(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=float).ravel()
        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"obs dim {obs.shape[0]} != {self.obs_dim}")

        if self.encoding == "cartpole_rbf":
            delta = (obs[:, None] - self._rbf_centers) / self._rbf_widths[:, None]
            activity = np.exp(-0.5 * np.sum(delta * delta, axis=0))
            V = self.V_read * activity
            pre = activity >= max(self.pre_threshold, 0.25)
            if not np.any(pre):
                pre[int(np.argmax(activity))] = True
            self._last_V = V
            self._last_pre = pre
            return V

        if self.encoding in ("cartpole_population", "cartpole_rate_population"):
            # Six 1-D Gaussian tuning curves per state component. Grouped rows
            # keep the crossbar input interpretable: x, x_dot, theta, theta_dot.
            values = np.clip(obs, -3.0, 3.0)
            delta = (
                values[:, None] - self._population_centers
            ) / self._population_widths[:, None]
            activity = np.exp(-0.5 * delta * delta)
            if self.encoding == "cartpole_rate_population":
                # Neurons keep a low spontaneous firing baseline; state
                # selectivity is represented by the modulated rate above it.
                activity = 0.10 + 0.90 * activity
            activity = activity.reshape(self.N)
            V = self.V_read * activity
            pre = activity >= max(self.pre_threshold, 0.25)
            self._last_V = V
            self._last_pre = pre
            return V

        if self.encoding == "cartpole_opponent_rate":
            values = np.clip(obs / self._opponent_scales, -1.0, 1.0)
            activity = np.full(self.N, self._opponent_baseline, dtype=float)
            for dim, value in enumerate(values):
                start = 6 * dim
                magnitude = abs(float(value))
                levels = np.clip(
                    (magnitude - self._opponent_thresholds)
                    / (1.0 - self._opponent_thresholds),
                    0.0, 1.0,
                )
                channel_start = start if value >= 0.0 else start + 3
                activity[channel_start:channel_start + 3] = (
                    self._opponent_baseline
                    + (1.0 - self._opponent_baseline) * levels
                )
            V = self.V_read * activity
            self._last_V = V
            self._last_pre = activity >= max(self.pre_threshold, 0.25)
            return V

        if self.encoding == "cartpole_opponent_binary":
            # Eight fixed rows preserve sensor identity and sign: ON/OFF for
            # x, x_dot, pole angle, and pole angular velocity. This is the
            # smallest positive-voltage encoding that keeps all four signed
            # CartPole variables separately visible to STDP.
            values = np.clip(obs, -1.0, 1.0)
            activity = np.full(self.N, self._opponent_baseline, dtype=float)
            for sensor, value in enumerate(values):
                polarity_row = 2 * sensor + (0 if value >= 0.0 else 1)
                activity[polarity_row] = (
                    self._opponent_baseline
                    + (1.0 - self._opponent_baseline) * abs(float(value))
                )
            pre = activity >= max(self.pre_threshold, 0.25)
            if not np.any(pre):
                sensor = int(np.argmax(np.abs(values)))
                polarity_row = 2 * sensor + (0 if values[sensor] >= 0.0 else 1)
                activity[polarity_row] = 0.30
                pre[polarity_row] = True
            V = self.V_read * activity
            self._last_V = V
            self._last_pre = pre
            return V

        if self.encoding == "obstacle_grid":
            values = np.clip(obs, -1.0, 1.0)
            activity = np.empty(self.N, dtype=float)
            # Observation[0:9]: 3x3 patch (-1 obstacle, 0 free, +1 goal).
            activity[:9] = values[:9]
            # Observation[9:11]: normalized row/column, each with 3 fields.
            for dim, start in ((9, 9), (10, 12)):
                delta = (values[dim] - self._grid_position_centers) / self._grid_position_width
                activity[start:start + 3] = np.exp(-0.5 * delta * delta)
            activity[15] = 1.0  # constant bias feature
            V = self.V_read * activity
            self._last_pre = np.abs(activity) >= max(self.pre_threshold, 0.25)
            self._last_V = V
            return V

        if self.encoding == "auto2d":
            # Preserve the meaning of each Auto2D sensor instead of mixing
            # all rays through a random projection. Each ray gets a danger
            # and a free-space channel; remaining rows encode goal bearing,
            # distance, speed, and a bias feature.
            obs = np.clip(obs, -1.0, 1.0)
            rays = np.clip(obs[:8], 0.0, 1.0)
            angle = float(np.clip(obs[8], -1.0, 1.0))
            goal_dist = float(np.clip(obs[9], 0.0, 1.0))
            speed = float(np.clip(obs[10], 0.0, 1.0))
            centers = np.array([-0.67, 0.0, 0.67])
            angle_features = np.exp(-0.5 * ((angle - centers) / 0.35) ** 2)
            distance_centers = np.array([0.25, 0.75])
            distance_features = np.exp(
                -0.5 * ((goal_dist - distance_centers) / 0.30) ** 2
            )
            speed_centers = np.array([0.20, 0.80])
            speed_features = np.exp(
                -0.5 * ((speed - speed_centers) / 0.30) ** 2
            )
            activity = np.concatenate((1.0 - rays, rays,
                                       angle_features, distance_features,
                                       speed_features,
                                       [1.0]))
            V = self.V_read * activity
            k = max(1, int(round(self.pre_topk * self.N)))
            idx = np.argsort(-activity)[:k]
            pre = np.zeros(self.N, dtype=bool)
            pre[idx] = True
            self._last_V = V
            self._last_pre = pre
            return V

        if self.encoding == "identity":
            # Static vision: preserve the one-pixel-to-one-crossbar-row map.
            # A random projection plus top-k binarization discards useful
            # pixel structure for this small supervised pattern task.
            activity = np.clip(obs, 0.0, 1.0)
            V = self.V_read * activity
            self._last_pre = activity > 0.5
            self._last_V = V
            return V

        if self.encoding == "vision_bipolar":
            # Encode both pixel states: black=+Vread, white=-Vread. The
            # bipolar crossbar input supplies explicit negative evidence for
            # absent pixels, which is important for overlapping patterns.
            activity = np.clip(obs, 0.0, 1.0)
            signed_activity = 2.0 * activity - 1.0
            V = self.V_read * signed_activity
            self._last_pre = activity > 0.5
            self._last_V = V
            return V

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
