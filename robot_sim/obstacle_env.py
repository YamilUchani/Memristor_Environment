"""Gridworld 2D con obstaculos para el agente R-STDP (seccion 3.3.7).

Grid 10x10, ~15% de obstaculos fijos (layout deterministico por seed),
inicio abajo-izquierda (9,0), meta arriba-derecha (0,9).
Reward sparse: +1 al llegar a la meta, -1 por colision, -0.01 por paso.
Observacion: parche 3x3 (9 bits: -1 obstaculo / 0 libre / +1 meta) +
posicion normalizada (2) -> 11 dims.
Acciones: 0=arriba, 1=abajo, 2=izquierda, 3=derecha.
"""
from collections import deque

import numpy as np

GRID = 10
OBS_RATIO = 0.15
MAX_STEPS = 100

_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]          # arriba, abajo, izq, der
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
            (0, 1), (1, -1), (1, 0), (1, 1)]          # parche 3x3


class ObstacleGridworld:
    """Entorno determinista por seed: mismo layout si se reinicia el estado."""

    def __init__(self, size=GRID, obs_ratio=OBS_RATIO, seed=0,
                 max_steps=MAX_STEPS, collision_penalty=-0.5,
                 shaping_w=0.6, goal_reward=1.0):
        self.size = size
        self.max_steps = max_steps
        self.collision_penalty = float(collision_penalty)
        self.shaping_w = float(shaping_w)
        self.goal_reward = float(goal_reward)
        self.start = (size - 1, 0)      # abajo-izquierda (fila, col)
        self.goal = (0, size - 1)       # arriba-derecha
        self.obs_ratio = obs_ratio
        self.obstacles = self._make_obstacles(seed)
        self.obs_dim = 11
        self.n_actions = 4
        self.agent = self.start
        self.steps = 0
        self.path = [self.start]

    # --------------------------------------------------------------
    # Layout
    # --------------------------------------------------------------
    def _make_obstacles(self, seed):
        rng = np.random.default_rng(seed)
        n_obs = int(round(self.size * self.size * self.obs_ratio))
        for _ in range(300):
            cells = set()
            while len(cells) < n_obs:
                r = int(rng.integers(self.size))
                c = int(rng.integers(self.size))
                q = (r, c)
                if q != self.start and q != self.goal:
                    cells.add(q)
            if self._is_solvable(cells):
                return cells
        raise RuntimeError("No se pudo generar un layout con camino")

    def _is_solvable(self, obstacles):
        if self.start in obstacles or self.goal in obstacles:
            return False
        seen = {self.start}
        q = deque([self.start])
        while q:
            r, c = q.popleft()
            if (r, c) == self.goal:
                return True
            for dr, dc in _MOVES:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.size and 0 <= nc < self.size
                        and (nr, nc) not in obstacles and (nr, nc) not in seen):
                    seen.add((nr, nc))
                    q.append((nr, nc))
        return False

    # --------------------------------------------------------------
    # Ciclo episodico
    # --------------------------------------------------------------
    def reset(self):
        self.agent = self.start
        self.steps = 0
        self.path = [self.agent]
        return self._obs()

    def step(self, action):
        self.steps += 1
        r, c = self.agent
        dr, dc = _MOVES[int(action)]
        nr, nc = r + dr, c + dc
        dist_prev = self._manhattan(r, c)

        hit = (nr < 0 or nr >= self.size or nc < 0 or nc >= self.size
               or (nr, nc) in self.obstacles)
        if hit:
            reward, done = self.collision_penalty, False   # colision: se queda
            dist_new = dist_prev
        else:
            self.agent = (nr, nc)
            self.path.append(self.agent)
            if self.agent == self.goal:
                reward, done = self.goal_reward, True
                dist_new = 0.0
            else:
                reward, done = -0.01, False
                dist_new = self._manhattan(nr, nc)

        # Reward shaping por funcion potencial (seguro: preserva optimalidad);
        # da gradiente local hacia la meta sin cambiar la politica optima.
        reward += self.shaping_w * (dist_prev - dist_new)

        if self.steps >= self.max_steps:
            done = True
        return self._obs(), float(reward), done, {}

    def _manhattan(self, r, c):
        return float(abs(r - self.goal[0]) + abs(c - self.goal[1]))

    def _obs(self):
        r, c = self.agent
        patch = np.zeros(9, dtype=float)
        for i, (dr, dc) in enumerate(_OFFSETS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if (nr, nc) == self.goal:
                    patch[i] = 1.0
                elif (nr, nc) in self.obstacles:
                    patch[i] = -1.0
            else:
                patch[i] = -1.0                    # fuera de bordes = obstaculo
        pos = np.array([r / (self.size - 1), c / (self.size - 1)])
        return np.concatenate([patch, pos])

    # --------------------------------------------------------------
    # Render (para figuras)
    # --------------------------------------------------------------
    def draw(self, ax, path=None, title=""):
        grid = np.zeros((self.size, self.size))
        for (r, c) in self.obstacles:
            grid[r, c] = 0.5
        ax.imshow(grid, cmap="gray", vmin=-0.1, vmax=1.1)
        ax.add_patch(plt.Rectangle((self.goal[1] - 0.5, self.goal[0] - 0.5),
                                   1, 1, facecolor="tab:green",
                                   edgecolor="black", lw=1.0))
        ax.add_patch(plt.Rectangle((self.start[1] - 0.5, self.start[0] - 0.5),
                                   1, 1, facecolor="tab:blue",
                                   edgecolor="black", lw=1.0))
        if path:
            ys = np.array([p[0] for p in path], dtype=float)
            xs = np.array([p[1] for p in path], dtype=float)
            ax.plot(xs, ys, color="tab:red", lw=2.2)
            ax.plot(xs[0], ys[0], "o", color="tab:blue", ms=8)
            ax.plot(xs[-1], ys[-1], "*", color="tab:red", ms=16)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)


import matplotlib.pyplot as plt  # noqa: E402  (import tardio evita dep GUI)