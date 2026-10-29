"""Interactive paired REINFORCE/R-STDP runs across Robot Sim environments.

Each run stops once its latest rolling window reaches the requested success
rate. Success means goal reached (Car/Obstacle), greedy classification correct
(Vision), or CartPole surviving to its 500-step time limit (Robot).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_car
import run_car_rstdp
import run_obstacle
import run_obstacle_rstdp
import run_vision
import run_vision_rstdp
from car_env import CarEnv
from obstacle_env import ObstacleGridworld
from neurobot.env_wrapper import CartPoleWrapper
from neurobot.reinforce_agent import NeuromorphicAgent as ReinforceAgent
from neurobot.rstdp_cartpole_agent import NeuromorphicAgent as RSTDPAgent


# Run in the presentation order: CartPole, vision, obstacle, then car.
DOMAINS = ("robot", "vision", "obstacle", "car")
ALGORITHMS = ("reinforce", "rstdp")
INITIAL_X_VALUES = (0.1, 0.9)
DEFAULT_SEEDS = tuple(range(10))
DEFAULT_MAX_EPISODES = 10_000
DEFAULT_WINDOW = 300
DEFAULT_TARGET_SUCCESS = 0.90
DEFAULT_PLOT_EVERY = 10
OUTPUT_ROOT = ROOT / "outputs"


class RollingSuccessStopper:
    """Stop when at least target successes occur in the last window episodes."""

    def __init__(self, window: int, target: float):
        self.window = int(window)
        self.target = float(target)
        self.values: deque[int] = deque(maxlen=self.window)
        self.solved_episode: int | None = None

    def record(self, success: bool, episode_number: int) -> bool:
        self.values.append(int(bool(success)))
        if (
            len(self.values) == self.window
            and float(np.mean(self.values)) > self.target
        ):
            self.solved_episode = int(episode_number)
        return self.solved_episode is None

    @property
    def rate(self) -> float | None:
        if not self.values:
            return None
        return float(np.mean(self.values))


class LiveTrainingPlot:
    """Four-panel live dashboard, with run_robot-style controls."""

    def __init__(self, window: int, target: float, refresh_every: int = 10):
        self.window = int(window)
        self.target = float(target)
        self.refresh_every = max(1, int(refresh_every))
        self.success: list[int] = []
        self.rolling: list[float] = []
        self.episode_rewards: list[float] = []
        self.obstacle_rewards: list[float] = []
        self.car_distances: list[float] = []
        self.vision_probabilities = None
        self.domain = ""
        self.paused = False
        self.stop_requested = False
        self.speed = 1.0
        self.plot_episode = 0
        self.robot_angles: list[float] = []
        self.robot_vels: list[float] = []
        self.robot_actions: list[int] = []
        self.last_robot_observation = None
        self.fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.92,
                                 bottom=0.17, hspace=0.38, wspace=0.28)
        self.fig.canvas.manager.set_window_title("Neuromorphic Robot Sim — Comparativa")
        self.ax_env, self.ax_X = axes[0]
        self.ax_success, self.ax_policy = axes[1]
        self.success_line, = self.ax_success.plot([], [])
        self.rolling_line, = self.ax_success.plot([], [])
        self._configure_learning_axis("")
        self.ax_X.set_title("Matriz X (memristores)")
        self.ax_X.set_xlabel("Columna (acción)")
        self.ax_X.set_ylabel("Fila (sensor)")
        self.image_X = None
        self.colorbar = None
        self.ax_policy.set_title("Política / diagnóstico")
        self.ax_policy.set_xlabel("Ángulo / acción")
        self.ax_policy.set_ylabel("Probabilidad / valor")
        self.ax_policy.grid(alpha=0.25)
        self.diagnostic_line = None
        self.probability_bars = []
        self.policy_lines = [
            self.ax_policy.plot([], [], "o-", color="tab:purple", ms=3,
                                label="Quieto (v~0)")[0],
            self.ax_policy.plot([], [], "^--", color="tab:red", ms=3,
                                label="Cayendo der. (v>0)")[0],
            self.ax_policy.plot([], [], "v--", color="tab:blue", ms=3,
                                label="Cayendo izq. (v<0)")[0],
        ]
        self.ax_policy.set_xlim(-0.22, 0.22)
        self.ax_policy.set_ylim(0, 1)
        self.ax_policy.legend(fontsize=7, loc="lower right")
        self.status = self.fig.text(0.02, 0.975, "Preparando corrida…",
                                    ha="left", va="top", family="monospace", fontsize=9)
        self.speed_status = self.fig.text(0.76, 0.975, "Velocidad x1",
                                          ha="right", va="top", family="monospace", fontsize=9)
        self.btn_pause = Button(self.fig.add_axes([0.23, 0.045, 0.16, 0.065]), "Pausar")
        self.btn_faster = Button(self.fig.add_axes([0.41, 0.045, 0.16, 0.065]), "Acelerar +")
        self.btn_slower = Button(self.fig.add_axes([0.59, 0.045, 0.16, 0.065]), "Desacelerar -")
        self.btn_stop = Button(self.fig.add_axes([0.77, 0.045, 0.16, 0.065]), "Detener lote")
        self.btn_pause.on_clicked(self.toggle_pause)
        self.btn_faster.on_clicked(self.speed_up)
        self.btn_slower.on_clicked(self.slow_down)
        self.btn_stop.on_clicked(self.stop)
        self.closed = False
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        plt.ion()
        plt.show(block=False)
        self.fig.canvas.draw_idle()
        plt.pause(0.05)

    def _on_close(self, _event):
        self.closed = True

    def begin(self, label: str, domain: str):
        self.domain = domain
        self.plot_episode = 0
        self.last_robot_observation = None
        self.success.clear()
        self.rolling.clear()
        self.episode_rewards.clear()
        self.obstacle_rewards.clear()
        self.car_distances.clear()
        self.vision_probabilities = None
        self.robot_angles.clear()
        self.robot_vels.clear()
        self.robot_actions.clear()
        self.success_line.set_data([], [])
        self.rolling_line.set_data([], [])
        # A new run starts at episode 1; do not retain the previous run's x-axis.
        self._configure_learning_axis(domain)
        self.ax_env.clear()
        self.ax_env.set_title(f"Entorno — {domain}")
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        self.ax_X.clear()
        self.ax_X.set_title("Matriz X (memristores)")
        self.image_X = None
        self.ax_policy.clear()
        self.ax_policy.set_title("Política / diagnóstico")
        self.ax_policy.set_xlabel("Ángulo / acción")
        self.ax_policy.set_ylabel("Probabilidad / valor")
        self.ax_policy.grid(alpha=0.25)
        self.policy_lines = [
            self.ax_policy.plot([], [], "o-", color="tab:purple", ms=3,
                                label="Quieto (v~0)")[0],
            self.ax_policy.plot([], [], "^--", color="tab:red", ms=3,
                                label="Cayendo der. (v>0)")[0],
            self.ax_policy.plot([], [], "v--", color="tab:blue", ms=3,
                                label="Cayendo izq. (v<0)")[0],
        ]
        self.ax_policy.set_xlim(-0.22, 0.22)
        self.ax_policy.set_ylim(0, 1)
        self._configure_diagnostic_axis(domain)
        self.status.set_text(label)
        if not self.closed and plt.fignum_exists(self.fig.number):
            self.fig.canvas.draw_idle()
            plt.pause(0.05)

    def _configure_learning_axis(self, domain: str):
        """Use the learning metric shown by that environment's own run window."""
        self.ax_success.clear()
        self.ax_success.set_xlabel("Episodio")
        self.ax_success.set_xlim(1, 10)
        if domain == "robot":
            # Match run_robot.py: episode return is the primary learning curve.
            self.ax_success.set_title("Curva de aprendizaje")
            self.ax_success.set_ylabel("Recompensa")
            self.ax_success.set_ylim(0, 520)
            self.ax_success.axhline(22, color="gray", ls="--", lw=1,
                                    label="aleatorio (~22)")
            self.ax_success.axhline(430, color="tab:orange", ls="-.", lw=1,
                                    label="freeze X (430)")
            self.ax_success.axhline(500, color="green", ls=":", lw=1,
                                    label="óptimo (~500)")
            self.success_line, = self.ax_success.plot(
                [], [], color="tab:blue", lw=1.2, label="Recompensa por episodio"
            )
            self.rolling_line, = self.ax_success.plot(
                [], [], color="tab:purple", lw=1.8, label="Media móvil (20)"
            )
            self.ax_success.legend(fontsize=7, loc="upper left")
        else:
            metric_window = 100 if domain == "vision" else 50
            self.ax_success.set_title("Curva de aprendizaje")
            self.ax_success.set_ylabel(
                f"Accuracy greedy (media {metric_window})" if domain == "vision"
                else f"Éxito (media móvil {metric_window})"
            )
            self.ax_success.set_ylim(-0.05, 1.05)
            if domain == "vision":
                self.ax_success.axhline(1 / 8, ls="--", color="gray",
                                        label="azar (1/8)")
            self.success_line, = self.ax_success.plot(
                [], [], color="tab:gray", lw=0.7, alpha=0.45,
                label="Correcto / éxito por episodio",
            )
            self.rolling_line, = self.ax_success.plot(
                [], [], color="tab:blue", lw=1.6,
                label=f"Media móvil ({metric_window})",
            )
            self.ax_success.legend(fontsize=7, loc="lower right")
        self.ax_success.grid(alpha=0.25)

    def _configure_diagnostic_axis(self, domain: str):
        """Show the same companion plot used by that domain's regular runner."""
        self.ax_policy.clear()
        self.diagnostic_line = None
        self.probability_bars = []
        if domain == "robot":
            self.ax_policy.set_title("Política aprendida (Multidimensional)")
            self.ax_policy.set_xlabel("Ángulo del poste (rad)")
            self.ax_policy.set_ylabel("P(Derecha | ángulo, vel)")
            self.ax_policy.axhline(0.5, color="gray", ls="--", lw=1)
            self.policy_lines = [
                self.ax_policy.plot([], [], "o-", color="tab:purple", ms=3,
                                    label="Quieto (v~0)")[0],
                self.ax_policy.plot([], [], "^--", color="tab:red", ms=3,
                                    label="Cayendo der. (v>0)")[0],
                self.ax_policy.plot([], [], "v--", color="tab:blue", ms=3,
                                    label="Cayendo izq. (v<0)")[0],
            ]
            self.ax_policy.set_xlim(-0.22, 0.22)
            self.ax_policy.set_ylim(0, 1)
            self.ax_policy.legend(fontsize=7, loc="lower right")
        elif domain == "vision":
            self.ax_policy.set_title("Softmax (último paso)")
            self.ax_policy.set_xlabel("Clase")
            self.ax_policy.set_ylabel("P(clase | entrada)")
            self.ax_policy.set_xticks(np.arange(8))
            self.ax_policy.set_ylim(0, 1)
            self.probability_bars = self.ax_policy.bar(
                np.arange(8), np.zeros(8), color="tab:blue"
            )
        elif domain == "obstacle":
            self.ax_policy.set_title("Recompensa por episodio")
            self.ax_policy.set_xlabel("Episodio")
            self.ax_policy.set_ylabel("Recompensa")
            self.diagnostic_line, = self.ax_policy.plot(
                [], [], color="tab:purple", lw=1.2
            )
        elif domain == "car":
            self.ax_policy.set_title("Distancia a meta")
            self.ax_policy.set_xlabel("Episodio")
            self.ax_policy.set_ylabel("Distancia final a meta (media 50)")
            self.diagnostic_line, = self.ax_policy.plot(
                [], [], color="tab:purple", lw=1.2
            )
        else:
            self.ax_policy.set_title("Diagnóstico")
        self.ax_policy.grid(alpha=0.25)

    def _draw_domain_diagnostic(self, vision=None):
        if self.domain == "robot":
            self._draw_policy()
        elif self.domain == "vision" and vision is not None:
            probabilities = np.asarray(vision[3], dtype=float)
            if probabilities.size == len(self.probability_bars):
                for bar, value in zip(self.probability_bars, probabilities):
                    bar.set_height(float(value))
        elif self.domain == "obstacle" and self.diagnostic_line is not None:
            x = np.arange(1, len(self.obstacle_rewards) + 1)
            self.diagnostic_line.set_data(x, self.obstacle_rewards)
            self.ax_policy.set_xlim(0, max(10, len(self.obstacle_rewards)))
            self.ax_policy.relim()
            self.ax_policy.autoscale_view(scalex=False, scaley=True)
        elif self.domain == "car" and self.diagnostic_line is not None:
            values = np.asarray(self.car_distances, dtype=float)
            if values.size:
                window = min(50, values.size)
                smooth = np.convolve(values, np.ones(window) / window, mode="valid")
                self.diagnostic_line.set_data(
                    np.arange(window, window + smooth.size), smooth
                )
                self.ax_policy.set_xlim(0, max(10, values.size))
                self.ax_policy.relim()
                self.ax_policy.autoscale_view(scalex=False, scaley=True)
    def toggle_pause(self, _event=None):
        self.paused = not self.paused
        self.btn_pause.label.set_text("Continuar" if self.paused else "Pausar")
        self.fig.canvas.draw_idle()

    def speed_up(self, _event=None):
        self.speed = min(100.0, self.speed * 2.0)
        self.speed_status.set_text(f"Velocidad x{self.speed:g}")
        self.fig.canvas.draw_idle()

    def slow_down(self, _event=None):
        self.speed = max(0.25, self.speed / 2.0)
        self.speed_status.set_text(f"Velocidad x{self.speed:g}")
        self.fig.canvas.draw_idle()

    def stop(self, _event=None):
        self.stop_requested = True
        self.status.set_text("Deteniendo al cerrar el episodio actual…")
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == " ":
            self.toggle_pause()
        elif event.key in ("+", "=", "plus"):
            self.speed_up()
        elif event.key in ("-", "minus"):
            self.slow_down()
        elif event.key in ("q", "escape"):
            self.stop()

    def wait_if_paused(self) -> bool:
        while self.paused and not self.stop_requested and not self.closed:
            plt.pause(0.05)
        return not self.stop_requested and not self.closed

    def _set_crossbar(self, X):
        X = np.asarray(X, dtype=float)
        if self.image_X is None or self.image_X.get_array().shape != X.shape:
            if self.colorbar is not None:
                self.colorbar.remove()
                self.colorbar = None
            self.ax_X.clear()
            self.image_X = self.ax_X.imshow(
                X, cmap="viridis", vmin=0, vmax=1, aspect="auto"
            )
            self.ax_X.set_title("Matriz X (memristores)")
            self.ax_X.set_xlabel("Columna (acción)")
            self.ax_X.set_ylabel("Fila (sensor)")
            self.ax_X.set_xticks(np.arange(X.shape[1]))
            self.ax_X.set_yticks(np.arange(0, X.shape[0], max(1, X.shape[0] // 12)))
            self.colorbar = self.fig.colorbar(self.image_X, ax=self.ax_X, fraction=0.046)
        else:
            self.image_X.set_data(X)

    def _draw_environment(self, env=None, observation=None, vision=None):
        ax = self.ax_env
        ax.clear()
        if self.domain == "robot" and observation is not None:
            raw = np.asarray(observation, dtype=float)
            x, theta = float(raw[0]), float(raw[2])
            ax.set_xlim(-2.6, 2.6)
            ax.set_ylim(-0.6, 1.6)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            ax.add_patch(Rectangle((x - 0.2, -0.1), 0.4, 0.2, color="tab:blue"))
            ax.plot([x, x + 0.5 * np.sin(theta)],
                    [0, 0.5 * np.cos(theta)], "o-", lw=3,
                    color="tab:orange", ms=5)
            ax.set_title("Robot (CartPole)")
        elif self.domain == "car" and env is not None:
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            for ox, oy in getattr(env, "obstacles", []):
                ax.add_patch(Circle((ox, oy), 0.4, color="dimgray"))
            ax.add_patch(Circle((8.5, 8.5), 0.8, color="tab:green", alpha=0.55))
            trail = np.asarray(getattr(env, "trail", []), dtype=float)
            if trail.ndim == 2 and trail.size:
                ax.plot(trail[:, 0], trail[:, 1], color="tab:blue", lw=1.3)
            ax.plot([1], [1], "o", color="tab:orange", label="Inicio")
            ax.set_title("Auto 2D — trayectoria")
        elif self.domain == "obstacle" and env is not None:
            size = int(env.size)
            ax.set_xlim(-0.5, size - 0.5)
            ax.set_ylim(size - 0.5, -0.5)
            ax.set_xticks(np.arange(size))
            ax.set_yticks(np.arange(size))
            ax.grid(color="white", lw=0.8)
            for row, col in getattr(env, "obstacles", []):
                ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1,
                                       color="dimgray"))
            path = np.asarray(getattr(env, "path", []), dtype=float)
            if path.ndim == 2 and path.size:
                ax.plot(path[:, 1], path[:, 0], color="tab:blue", marker=".")
            ax.plot(env.start[1], env.start[0], "o", color="tab:orange")
            ax.plot(env.goal[1], env.goal[0], "*", color="tab:green", ms=12)
            ax.set_title("Navegación con obstáculos")
        elif self.domain == "vision" and vision is not None:
            image, label, prediction = vision[:3]
            ax.imshow(image, cmap="gray_r", vmin=0, vmax=1)
            ax.set_title(f"Visión — clase {label}, predicción {prediction}")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.text(0.5, 0.5, "Esperando estado del entorno",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Entorno — {self.domain}")

    def _draw_policy(self):
        if self.domain != "robot" or not self.robot_angles:
            return
        angles = np.asarray(self.robot_angles[-5000:])
        vels = np.asarray(self.robot_vels[-5000:])
        actions = np.asarray(self.robot_actions[-5000:])
        edges = np.linspace(-0.22, 0.22, 12)
        centers = 0.5 * (edges[:-1] + edges[1:])
        values = [np.full(len(centers), np.nan) for _ in range(3)]
        for i in range(len(centers)):
            mask = (angles >= edges[i]) & (angles < edges[i + 1])
            groups = (mask & (np.abs(vels) < 0.2),
                      mask & (vels >= 0.2), mask & (vels <= -0.2))
            for index, group in enumerate(groups):
                if int(group.sum()) > 3:
                    values[index][i] = float(np.mean(actions[group] >= 6))
        for line, data in zip(self.policy_lines, values):
            line.set_data(centers, data)

    def update_robot_step(self, observation, action: int, agent, env):
        self.last_robot_observation = np.asarray(observation, dtype=float).copy()
        self.fig.canvas.flush_events()
        if not self.wait_if_paused():
            return False
        raw = np.asarray(observation, dtype=float)
        self.robot_angles.append(float(raw[2]))
        self.robot_vels.append(float(raw[3]))
        self.robot_actions.append(int(action))
        if len(self.robot_angles) > 5000:
            del self.robot_angles[:-5000]
            del self.robot_vels[:-5000]
            del self.robot_actions[:-5000]
        self.plot_episode += 1
        if self.plot_episode % max(1, int(round(10 * self.speed))):
            return not self.stop_requested and not self.closed
        self._set_crossbar(agent.brain.X)
        self._draw_environment(env=env, observation=raw)
        self._draw_policy()
        self.status.set_text(
            f"{self.domain.upper()} | paso de CartPole | acción {action} | "
            f"X̄={agent.brain.X.mean():.3f}"
        )
        if not self.closed and plt.fignum_exists(self.fig.number):
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
        return not self.stop_requested and not self.closed

    def update(self, episode: int, success: bool, rate: float | None,
               solved: bool = False, agent=None, env=None, vision=None,
               reward: float | None = None):
        if not self.wait_if_paused():
            return False
        self.success.append(int(bool(success)))
        self.rolling.append(float(rate) if rate is not None else np.nan)
        if reward is not None:
            self.episode_rewards.append(float(reward))
            if self.domain == "obstacle":
                self.obstacle_rewards.append(float(reward))
        if self.domain == "car" and env is not None:
            # CarEnv exposes the same terminal distance internally as
            # _dist_goal(); it has no public distance_to_goal() method.
            self.car_distances.append(float(env._dist_goal()))
        if self.domain == "vision" and vision is not None:
            self.vision_probabilities = np.asarray(vision[3], dtype=float).copy()
        # Speed controls simulation throughput, not how long the success chart
        # stays stale. Keep its refresh cadence tied only to --plot-every.
        update_every = self.refresh_every
        if episode % update_every and not solved and not self.stop_requested:
            # Keep processing button/keyboard events even when skipping redraws.
            self.fig.canvas.flush_events()
            return not self.stop_requested and not self.closed
        if self.closed or not plt.fignum_exists(self.fig.number):
            self.closed = True
            return False
        episodes = np.arange(1, len(self.success) + 1)
        if self.domain == "robot":
            reward_values = np.asarray(self.episode_rewards, dtype=float)
            reward_x = np.arange(1, reward_values.size + 1)
            self.success_line.set_data(reward_x, reward_values)
            window = min(20, reward_values.size)
            if window:
                smooth = np.convolve(
                    reward_values, np.ones(window) / window, mode="valid"
                )
                self.rolling_line.set_data(
                    np.arange(window, window + smooth.size), smooth
                )
        else:
            self.success_line.set_data(episodes, self.success)
            window = 100 if self.domain == "vision" else 50
            values = np.asarray(self.success, dtype=float)
            if values.size:
                prefix = np.concatenate(([0.0], np.cumsum(values)))
                ends = np.arange(1, values.size + 1)
                starts = np.maximum(0, ends - window)
                denominators = ends - starts
                sums = prefix[ends] - prefix[starts]
                self.rolling_line.set_data(episodes, sums / denominators)
        self.ax_success.set_xlim(1, max(10, episode))
        rate_text = "n/a" if rate is None else f"{rate:.1%}"
        if agent is not None:
            brain = getattr(agent, "brain", agent)
            self._set_crossbar(brain.X)
            x_mean = float(brain.X.mean())
        else:
            x_mean = float("nan")
        self._draw_environment(env=env, vision=vision)
        if self.domain == "robot" and self.last_robot_observation is not None:
            self._draw_environment(env=env, observation=self.last_robot_observation)
        if self.domain != "robot":
            self.ax_policy.clear()
            self.ax_policy.axis("off")
            self.ax_policy.text(
                0.03, 0.95,
                f"Episodio: {episode}\n"
                f"Éxito últimos {self.window}: {rate_text}\n"
                f"Último episodio: {'éxito' if success else 'fallo'}\n"
                f"Media X: {x_mean:.4f}\n"
                f"Velocidad de actualización: x{self.speed:g}\n"
                f"Estado: {'PAUSA' if self.paused else 'entrenando'}",
                transform=self.ax_policy.transAxes, va="top", fontsize=11,
            )
            self.ax_policy.set_title("Diagnóstico de entrenamiento")
        self._draw_domain_diagnostic(vision=vision)
        self.status.set_text(
            f"{self.domain.upper()} | episodio {episode} | "
            f"éxito móvil: {rate_text} | {'resuelto' if solved else 'entrenando'}"
        )
        self.fig.canvas.draw_idle()
        plt.pause(0.001)
        return not self.stop_requested and not self.closed

    def finish(self):
        if not self.closed and plt.fignum_exists(self.fig.number):
            self.status.set_text("Lote terminado. Cierra esta ventana para salir.")
            plt.ioff()
            plt.show(block=True)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _agent_metadata(agent, seed: int, initial_X: np.ndarray) -> dict:
    # Vision's existing runners return CrossbarBrain directly; the others
    # return the full agent.
    brain = getattr(agent, "brain", agent)
    seed_config = getattr(agent, "seed_config", None)
    return {
        "crossbar_shape": list(brain.X.shape),
        "initial_X_mean": float(initial_X.mean()),
        "initial_X_min": float(initial_X.min()),
        "initial_X_max": float(initial_X.max()),
        "crossbar_seed": int(seed_config["crossbar"] if seed_config else seed),
        "encoder_seed": int(seed_config["encoder"] if seed_config else seed),
        "action_seed": int(seed_config["actions"] if seed_config else seed),
        "volatile_devices": len(getattr(brain, "volatile_devices", [])),
    }


def _make_car_reinforce(seed: int, initial_x: float):
    env = CarEnv(seed=seed)
    agent = ReinforceAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        N=run_car.N,
        M=run_car.M,
        V_read=0.05,
        eta=run_car.ETA,
        encoder_mode=run_car.ENCODER_MODE,
        temp_init=run_car.TEMP_INIT_HD,
        temp_min=run_car.TEMP_MIN,
        temp_decay=run_car.TEMP_DECAY,
        dt=1e-3,
        k_wta=3,
        wta_temperature=1.0,
        baseline_window=200,
        seed=seed,
        crossbar_seed=seed,
        encoder_seed=seed,
        action_seed=seed,
        initial_X=initial_x,
        json_memristor_nv=run_car_rstdp.DEFAULT_NV_PROFILE,
        json_memristor_v="memristor_volatile.json",
        json_lif=run_car_rstdp.DEFAULT_LIF_PROFILE,
        d2d_enabled=False,
        c2c_enabled=False,
    )
    agent.max_weight_update = run_car.MAX_WEIGHT_UPDATE
    return env, agent


def _run_car(algorithm: str, seed: int, initial_x: float, max_episodes: int,
             window: int, target: float, live_plot: LiveTrainingPlot):
    stopper = RollingSuccessStopper(window, target)
    if algorithm == "reinforce":
        env, agent = _make_car_reinforce(seed, initial_x)
        initial_X = agent.brain.X.copy()
        tracker = run_car.GreedyPolicyTracker(seed)
        def on_reinforce_episode(ep, success):
            keep_going = stopper.record(success, ep + 1)
            ui_continue = live_plot.update(
                ep + 1, success, stopper.rate, not keep_going,
                agent=agent, env=env,
            )
            return keep_going and ui_continue

        log = run_car.run_headless(
            env,
            agent,
            episodes=max_episodes,
            seed=seed,
            verbose=True,
            policy_tracker=tracker,
            eval_interval=200,
            adapt_exploration=False,
            on_episode=on_reinforce_episode,
        )
    else:
        env = CarEnv(seed=seed)
        agent = run_car_rstdp.make_agent(
            seed=seed,
            eta=run_car_rstdp.DEFAULT_ETA,
            initial_x=initial_x,
        )
        initial_X = agent.brain.X.copy()
        def on_rstdp_episode(ep, success, env_, agent_, _log):
            keep_going = stopper.record(success, ep + 1)
            ui_continue = live_plot.update(
                ep + 1, success, stopper.rate, not keep_going,
                agent=agent_, env=env_,
            )
            return keep_going and ui_continue

        log = run_car_rstdp.run_training(
            agent,
            env,
            episodes=max_episodes,
            seed=seed,
            verbose=True,
            policy_eval_interval=run_car_rstdp.POLICY_EVAL_INTERVAL,
            on_episode=on_rstdp_episode,
        )
    if hasattr(env, "close"):
        env.close()
    log["episodes_completed"] = len(log.get("success", []))
    return agent, initial_X, log, stopper


def _run_vision(algorithm: str, seed: int, initial_x: float,
                max_episodes: int, window: int, target: float,
                live_plot: LiveTrainingPlot):
    stopper = RollingSuccessStopper(window, target)
    patterns = run_vision.make_patterns()

    def on_vision_step(ep, label, probs, brain, data):
        success = bool(data["greedy_correct"][-1])
        prediction = int(np.argmax(probs))
        keep_going = stopper.record(success, ep + 1)
        ui_continue = live_plot.update(
            ep + 1, success, stopper.rate, not keep_going,
            agent=brain,
            vision=(patterns[int(label)], int(label), prediction, probs),
        )
        return keep_going and ui_continue

    if algorithm == "reinforce":
        brain, log = run_vision.run_classifier(
            patterns,
            episodes=max_episodes,
            eta=5e-4,
            seed=seed,
            initial_X=initial_x,
            verbose=True,
            on_step=on_vision_step,
        )
    else:
        brain, log = run_vision_rstdp.run_classifier(
            patterns,
            episodes=max_episodes,
            eta=run_vision_rstdp.DEFAULT_ETA,
            seed=seed,
            initial_x=initial_x,
            verbose=True,
            on_step=on_vision_step,
        )
    initial_X = np.asarray(log.pop("initial_crossbar_X"), dtype=float)
    log["success"] = [int(value) for value in log.get("greedy_correct", [])]
    log["episodes_completed"] = len(log["success"])
    return brain, initial_X, log, stopper


def _run_obstacle(algorithm: str, seed: int, initial_x: float,
                  max_episodes: int, window: int, target: float,
                  live_plot: LiveTrainingPlot):
    stopper = RollingSuccessStopper(window, target)
    env = ObstacleGridworld(
        seed=seed,
        goal_reward=10.0,
        shaping_w=0.6,
        collision_penalty=-1.0,
    )

    def on_obstacle_step(ep, success, env_, agent_, log):
        keep_going = stopper.record(success, ep + 1)
        rewards = log.get("reward", [])
        ui_continue = live_plot.update(
            ep + 1, success, stopper.rate, not keep_going,
            agent=agent, env=env_,
            reward=float(rewards[-1]) if rewards else None,
        )
        return keep_going and ui_continue

    if algorithm == "reinforce":
        agent = ReinforceAgent(
            obs_dim=env.obs_dim,
            n_actions=env.n_actions,
            N=16,
            M=4,
            eta=1e-2,
            seed=seed,
            crossbar_seed=seed,
            encoder_seed=seed,
            action_seed=seed,
            initial_X=initial_x,
            encoder_mode="random",
        )
        initial_X = agent.brain.X.copy()
        log, early_path, learned_path, greedy_success = run_obstacle.run_episodes(
            agent,
            env,
            n_episodes=max_episodes,
            verbose=True,
            on_step=on_obstacle_step,
        )
        log["early_path"] = early_path
        log["final_greedy_path"] = learned_path
        log["final_greedy_success"] = greedy_success
    else:
        agent = run_obstacle_rstdp.make_agent(
            seed=seed,
            eta=run_obstacle_rstdp.DEFAULT_ETA,
            initial_x=initial_x,
        )
        initial_X = agent.brain.X.copy()
        log, early_path, learned_path, greedy_success = (
            run_obstacle_rstdp.run_training(
                agent,
                env,
                episodes=max_episodes,
                verbose=True,
                on_step=on_obstacle_step,
            )
        )
        log["early_path"] = early_path
        log["final_greedy_path"] = learned_path
        log["final_greedy_success"] = greedy_success
    log["episodes_completed"] = len(log.get("success", []))
    return agent, initial_X, log, stopper


def _make_robot_agent(algorithm: str, env, seed: int, initial_x: float):
    common = dict(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        N=24,
        M=12,
        V_read=0.05,
        initial_X=initial_x,
        seed=seed,
        crossbar_seed=seed,
        encoder_seed=seed,
        action_seed=seed,
        encoder_mode="random",
        json_memristor_nv="strukov_ideal.json",
        json_memristor_v="memristor_volatile.json",
        json_lif="lif_config.json",
        d2d_enabled=False,
        c2c_enabled=False,
    )
    if algorithm == "reinforce":
        return ReinforceAgent(
            **common,
            eta=0.005,
            temp_init=1.0,
            temp_min=0.05,
            temp_decay=0.995,
            baseline_window=200,
            epsilon=0.0,
        )
    environment_dt = float(getattr(env.env.unwrapped, "tau", 0.015))
    substeps = 3
    return RSTDPAgent(
        **common,
        eta=0.2,
        dt=environment_dt / substeps,
        steps_per_action=substeps,
        k_wta=3,
        wta_temperature=1.0,
        a_plus=0.01,
        a_minus=0.01,
        reward_baseline_window=200,
        return_baseline_alpha=0.05,
        max_abs_delta_x=0.02,
        temp_init=1.0,
        temp_min=0.05,
        temp_decay=0.995,
    )


def _run_robot(algorithm: str, seed: int, initial_x: float,
               max_episodes: int, window: int, target: float,
               live_plot: LiveTrainingPlot):
    env = CartPoleWrapper(seed=seed, max_steps=500, death_penalty=50.0)
    agent = _make_robot_agent(algorithm, env, seed, initial_x)
    initial_X = agent.brain.X.copy()
    live_plot._set_crossbar(initial_X)
    stopper = RollingSuccessStopper(window, target)
    log = {
        "episode": [],
        "success": [],
        "reward": [],
        "steps": [],
        "X_mean": [],
        "temperature": [],
    }

    for episode in range(max_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        step_rewards = []
        info = {}
        while not done:
            live_plot.wait_if_paused()
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            reward = float(reward)
            total_reward += reward
            step_rewards.append(reward)
            if algorithm == "rstdp":
                agent.observe_reward(reward)
            if not live_plot.update_robot_step(
                info.get("obs_raw", obs), action, agent, env
            ):
                # Finish this episode cleanly before honoring stop/close.
                live_plot.stop_requested = True
        success = bool(info.get("truncated") and not info.get("terminated"))
        if algorithm == "reinforce":
            agent.learn(total_reward, freeze_update_above=430.0)
        else:
            agent.finish_episode(total_reward)
        agent.end_episode()
        log["episode"].append(episode + 1)
        log["success"].append(int(success))
        log["reward"].append(float(total_reward))
        log["steps"].append(int(env._step_count))
        log["X_mean"].append(float(agent.brain.X.mean()))
        log["temperature"].append(float(agent.current_temperature()))
        if (episode + 1) % 100 == 0:
            print(
                f"[robot {algorithm} seed={seed} X0={initial_x:g}] "
                f"ep={episode + 1} success({window})="
                f"{stopper.rate if stopper.rate is not None else 0.0:.3f}",
                flush=True,
            )
        keep_going = stopper.record(success, episode + 1)
        ui_continue = live_plot.update(
            episode + 1, success, stopper.rate, not keep_going,
            agent=agent, env=env, reward=total_reward,
        )
        if not keep_going or not ui_continue:
            break
    env.close()
    log["episodes_completed"] = len(log["episode"])
    return agent, initial_X, log, stopper


def _plot_run(run_dir: Path, domain: str, initial_X: np.ndarray,
              final_X: np.ndarray, success: list[float], window: int,
              target: float) -> None:
    success_values = np.asarray(success, dtype=float)
    if success_values.size >= window:
        rolling = np.convolve(
            success_values, np.ones(window, dtype=float) / window, mode="valid"
        )
        x = np.arange(window, window + rolling.size)
    else:
        rolling = np.asarray([], dtype=float)
        x = np.asarray([], dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(x, rolling, color="tab:blue", lw=1.5)
    axes[0].axhline(target, color="tab:red", ls="--", label=f"meta {target:.0%}")
    axes[0].set(xlabel="Episodio", ylabel=f"Éxito móvil ({window})", ylim=(0, 1.02))
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[0].set_title(f"{domain}: éxito durante entrenamiento")
    axes[1].plot(np.arange(1, success_values.size + 1), success_values,
                 color="tab:gray", alpha=0.45, lw=0.7)
    axes[1].set(xlabel="Episodio", ylabel="Éxito (0/1)", ylim=(-0.05, 1.05))
    axes[1].grid(alpha=0.25)
    axes[1].set_title("Resultado por episodio")
    fig.tight_layout()
    fig.savefig(run_dir / "success_curve.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, matrix, title in zip(
        axes,
        (initial_X, final_X, np.abs(final_X - initial_X)),
        ("X inicial", "X final", "|ΔX|"),
    ):
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0,
                          vmax=1.0 if title != "|ΔX|" else None)
        ax.set_title(title)
        ax.set_xlabel("Acción / columna")
        ax.set_ylabel("Sensor / fila")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(f"{domain}: estado del crossbar")
    fig.tight_layout()
    fig.savefig(run_dir / "crossbar.png", dpi=160)
    plt.close(fig)


def _run_one(domain: str, algorithm: str, seed: int, initial_x: float,
             args, output_dir: Path, live_plot: LiveTrainingPlot) -> dict:
    run_name = f"{domain}_{algorithm}_seed{seed}_X0_{initial_x:g}"
    run_dir = output_dir / "runs" / run_name
    result_path = run_dir / "result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        expected = (domain, algorithm.lower().replace("-", ""), seed,
                    initial_x, args.window,
                    args.target_success, args.max_episodes)
        found = (
            existing.get("domain"),
            str(existing.get("algorithm", "")).lower().replace("-", ""),
            existing.get("seed"), existing.get("initial_X"),
            existing.get("stop_rule", {}).get("window"),
            existing.get("stop_rule", {}).get("target_success"),
            existing.get("max_episodes"),
        )
        if expected == found:
            print(f"[ya completado] {run_name}", flush=True)
            return existing
        raise FileExistsError(f"Result conflicts with requested run: {result_path}")
    run_dir.mkdir(parents=True, exist_ok=True)
    live_plot.begin(
        f"{domain.upper()} | {algorithm.upper()} | seed={seed} | X0={initial_x:g}",
        domain,
    )

    started = time.monotonic()
    if domain == "car":
        agent, initial_X, log, stopper = _run_car(
            algorithm, seed, initial_x, args.max_episodes,
            args.window, args.target_success, live_plot,
        )
    elif domain == "vision":
        agent, initial_X, log, stopper = _run_vision(
            algorithm, seed, initial_x, args.max_episodes,
            args.window, args.target_success, live_plot,
        )
    elif domain == "obstacle":
        agent, initial_X, log, stopper = _run_obstacle(
            algorithm, seed, initial_x, args.max_episodes,
            args.window, args.target_success, live_plot,
        )
    elif domain == "robot":
        agent, initial_X, log, stopper = _run_robot(
            algorithm, seed, initial_x, args.max_episodes,
            args.window, args.target_success, live_plot,
        )
    else:
        raise ValueError(f"Unknown domain: {domain}")

    brain = getattr(agent, "brain", agent)
    final_X = np.asarray(brain.X, dtype=float).copy()
    np.save(run_dir / "crossbar_initial_X.npy", initial_X)
    np.save(run_dir / "crossbar_final_X.npy", final_X)
    brain.save_weights(str(run_dir / "weights_final.npy"))
    success = [int(bool(value)) for value in log.get("success", [])]
    episodes_completed = int(log.get("episodes_completed", len(success)))
    solved = stopper.solved_episode is not None
    run_metadata = {
        "domain": domain,
        "algorithm": algorithm.upper() if algorithm == "reinforce" else "R-STDP",
        "seed": int(seed),
        "initial_X": float(initial_x),
        "max_episodes": int(args.max_episodes),
        "episodes_completed": episodes_completed,
        "stop_rule": {
            "metric": "binary per-episode success",
            "window": int(args.window),
            "target_success": float(args.target_success),
            "minimum_successes_in_window": int(
                np.floor(args.window * args.target_success) + 1
            ),
            "solved": bool(solved),
            "stopped_at_episode": stopper.solved_episode,
            "final_window_success_rate": stopper.rate,
        },
        "elapsed_seconds": float(time.monotonic() - started),
        "crossbar": {
            "shape": list(final_X.shape),
            "initial_mean": float(initial_X.mean()),
            "final_mean": float(final_X.mean()),
            "mean_abs_change": float(np.abs(final_X - initial_X).mean()),
        },
        "agent_seeds": _agent_metadata(agent, seed, initial_X),
        "files": {
            "initial_crossbar": "crossbar_initial_X.npy",
            "final_crossbar": "crossbar_final_X.npy",
            "final_weights": "weights_final.npy",
            "success_plot": "success_curve.png",
            "crossbar_plot": "crossbar.png",
        },
        "log": log,
    }
    _write_json(result_path, run_metadata)
    _plot_run(run_dir, domain, initial_X, final_X, success,
              args.window, args.target_success)
    print(
        f"[guardado] {run_name}: {episodes_completed} episodios; "
        f"solved={solved}; X {initial_X.mean():.4f}→{final_X.mean():.4f}",
        flush=True,
    )
    return run_metadata


def _write_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        "domain", "algorithm", "seed", "initial_X", "episodes_completed",
        "solved", "stopped_at_episode", "final_window_success_rate",
        "mean_abs_crossbar_change", "elapsed_seconds", "result_path",
    ]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "domain": row["domain"],
                "algorithm": row["algorithm"],
                "seed": row["seed"],
                "initial_X": row["initial_X"],
                "episodes_completed": row["episodes_completed"],
                "solved": row["stop_rule"]["solved"],
                "stopped_at_episode": row["stop_rule"]["stopped_at_episode"],
                "final_window_success_rate": row["stop_rule"]["final_window_success_rate"],
                "mean_abs_crossbar_change": row["crossbar"]["mean_abs_change"],
                "elapsed_seconds": row["elapsed_seconds"],
                "result_path": str(
                    Path("runs")
                    / f"{row['domain']}_{row['algorithm'].lower().replace('-', '')}"
                      f"_seed{row['seed']}_X0_{row['initial_X']:g}"
                    / "result.json"
                ),
            })
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS,
                        default=list(DOMAINS))
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS,
                        default=list(ALGORITHMS))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(DEFAULT_SEEDS))
    parser.add_argument("--x0", nargs="+", type=float,
                        choices=INITIAL_X_VALUES,
                        default=list(INITIAL_X_VALUES))
    parser.add_argument("--max-episodes", type=int,
                        default=DEFAULT_MAX_EPISODES)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--target-success", type=float,
                        default=DEFAULT_TARGET_SUCCESS)
    parser.add_argument("--plot-every", type=int, default=DEFAULT_PLOT_EVERY,
                        help="Actualizar la gráfica en vivo cada N episodios.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip matching completed runs in an existing output folder.")
    parser.add_argument(
        "--skip-car-reinforce", action="store_true",
        help="Skip Car + REINFORCE when that baseline was already run.",
    )
    args = parser.parse_args()
    if args.max_episodes < args.window:
        parser.error("--max-episodes debe ser al menos --window")
    if args.window < 1:
        parser.error("--window debe ser positivo")
    if args.plot_every < 1:
        parser.error("--plot-every debe ser positivo")
    if not 0.0 <= args.target_success < 1.0:
        parser.error("--target-success debe estar en [0, 1)")
    if any(seed < 0 for seed in args.seeds):
        parser.error("Las semillas deben ser enteros no negativos")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("No repitas semillas")
    if len(set(args.domains)) != len(args.domains):
        parser.error("No repitas entornos")
    if len(set(args.algorithms)) != len(args.algorithms):
        parser.error("No repitas algoritmos")
    if len(set(args.x0)) != len(args.x0):
        parser.error("No repitas valores X0")
    return args


def main():
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else OUTPUT_ROOT / f"comparison_until_success_{stamp}"
    )
    protocol = {
        "domains": args.domains,
        "algorithms": args.algorithms,
        "seeds": args.seeds,
        "X0": args.x0,
        "max_episodes_per_run": args.max_episodes,
        "success_window": args.window,
        "target_success": args.target_success,
        "skip_car_reinforce": bool(args.skip_car_reinforce),
        "runs_expected": sum(
            1 for domain in args.domains for algorithm in args.algorithms
            if not (args.skip_car_reinforce and domain == "car"
                    and algorithm == "reinforce")
        ) * len(args.seeds) * len(args.x0),
        "success_definitions": {
            "car": "episode reaches goal",
            "vision": "greedy prediction is correct",
            "obstacle": "agent reaches goal",
            "robot": "CartPole reaches its time limit without termination",
        },
        "stop_definition": (
            "stop after the rolling window first contains at least "
            "strictly more than window * target_success successful episodes"
        ),
    }
    protocol_path = output_dir / "protocol.json"
    if output_dir.exists():
        if not args.resume:
            raise FileExistsError(
                f"La carpeta ya existe; usa otra o --resume: {output_dir}"
            )
        if not protocol_path.is_file():
            raise FileNotFoundError(
                f"No hay protocolo para reanudar: {protocol_path}"
            )
        existing_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing_protocol != protocol:
            raise ValueError("El protocolo no coincide con el lote existente.")
    else:
        output_dir.mkdir(parents=True)
        _write_json(protocol_path, protocol)

    expected_runs = [
        (domain, algorithm, seed, initial_x)
        for domain in args.domains
        for algorithm in args.algorithms
        if not (args.skip_car_reinforce and domain == "car"
                and algorithm == "reinforce")
        for seed in args.seeds
        for initial_x in args.x0
    ]
    rows = []
    live_plot = LiveTrainingPlot(
        args.window, args.target_success, refresh_every=args.plot_every
    )
    try:
        for domain, algorithm, seed, initial_x in expected_runs:
            row = _run_one(
                domain, algorithm, seed, initial_x, args, output_dir, live_plot
            )
            rows.append(row)
            _write_summary(output_dir / "summary.csv", rows)
            if live_plot.stop_requested or live_plot.closed:
                break

        print(
            f"[listo] {len(rows)} corridas guardadas en {output_dir}",
            flush=True,
        )
    finally:
        live_plot.finish()


if __name__ == "__main__":
    main()
