"""Generate reproducible CartPole and vision-pattern illustrations for the thesis."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from neurobot.agent import NeuromorphicAgent
from neurobot.env_wrapper import CartPoleWrapper
from run_vision import make_patterns

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "outputs" / "run_24x12_A_d2d_c2c_seed0_X.npy"
OUTPUT_DIRS = (
    ROOT.parent / "PLANTILLA_PDF_FINAL" / "figuras",
    ROOT.parent / "docs" / "latex" / "figuras",
)


def save_to_output_dirs(fig, filename):
    for directory in OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def cartpole_test_figure():
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Saved CartPole weights not found: {WEIGHTS}")

    env = CartPoleWrapper(seed=0, death_penalty=50.0)
    agent = NeuromorphicAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        N=24,
        M=12,
        V_read=0.05,
        eta=0.005,
        temp_init=1.0,
        temp_min=0.05,
        temp_decay=0.995,
        baseline_window=200,
        initial_X=0.1,
        seed=0,
        json_memristor_nv="strukov_ideal.json",
        json_memristor_v="memristor_volatile.json",
        json_lif="lif_config.json",
        d2d_enabled=True,
        c2c_enabled=True,
        d2d_sigma=0.08,
        c2c_sigma=0.05,
        c2c_theta=1.0,
        c2c_dt=0.02,
    )
    agent.brain.load_weights(str(WEIGHTS))
    agent.decoder.sample = False

    obs = env.reset()
    states = [np.asarray(env.env.unwrapped.state, dtype=float).copy()]
    total_reward = 0.0
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        states.append(np.asarray(info["obs_raw"], dtype=float))

    env.close()
    selected = [states[0], states[len(states) // 2], states[-1]]
    selected_steps = [0, len(states) // 2, len(states) - 1]
    fig = plt.figure(figsize=(10.8, 5.0))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.2, 1.0))
    axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    for axis, state, step in zip(axes, selected, selected_steps):
        cart_x, _, angle, _ = state
        axis.plot([-2.4, 2.4], [0.32, 0.32], color="#444444", linewidth=2)
        axis.add_patch(Rectangle((cart_x - 0.2, 0.32), 0.4, 0.18,
                                 facecolor="#4c78a8", edgecolor="#263746"))
        pole_length = 0.8
        pivot = (cart_x, 0.5)
        tip = (cart_x + pole_length * np.sin(angle),
               0.5 + pole_length * np.cos(angle))
        axis.plot([pivot[0], tip[0]], [pivot[1], tip[1]],
                  color="#e45756", linewidth=4, solid_capstyle="round")
        axis.scatter(*pivot, color="#222222", s=18, zorder=3)
        axis.set_xlim(-2.5, 2.5)
        axis.set_ylim(0.25, 1.55)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"Paso {step}")
        axis.set_xlabel("Posición del carro")
        axis.grid(axis="x", linestyle=":", alpha=0.4)
    axes[0].set_ylabel("Altura")
    trace_axis = fig.add_subplot(grid[1, :])
    steps = np.arange(len(states))
    state_array = np.asarray(states)
    trace_axis.plot(steps, np.degrees(state_array[:, 2]),
                    color="#e45756", label="Ángulo del poste")
    trace_axis.set_xlabel("Paso")
    trace_axis.set_ylabel("Ángulo (°)", color="#e45756")
    trace_axis.tick_params(axis="y", labelcolor="#e45756")
    trace_axis.grid(True, linestyle=":", alpha=0.4)
    position_axis = trace_axis.twinx()
    position_axis.plot(steps, state_array[:, 0], color="#4c78a8",
                       linestyle="--", label="Posición del carro")
    position_axis.set_ylabel("Posición del carro", color="#4c78a8")
    position_axis.tick_params(axis="y", labelcolor="#4c78a8")
    handles, labels = trace_axis.get_legend_handles_labels()
    extra_handles, extra_labels = position_axis.get_legend_handles_labels()
    trace_axis.legend(handles + extra_handles, labels + extra_labels,
                      loc="upper right", frameon=False)
    fig.suptitle(
        f"Evaluación ilustrativa CartPole-v1 | pesos guardados, semilla 0 | "
        f"{len(states) - 1} pasos, recompensa {total_reward:.1f}",
        fontsize=11,
    )
    fig.tight_layout()
    save_to_output_dirs(fig, "cartpole_test_seed0.png")


def vision_patterns_figure():
    patterns = make_patterns()
    titles = (
        "Horizontal", "Vertical", "Diagonal \\",
        "Diagonal /", "Cruz", "Marco", "X", "Cuadrado",
    )
    fig, axes = plt.subplots(2, 4, figsize=(8.4, 4.2))
    for axis, pattern, title in zip(axes.flat, patterns, titles):
        axis.imshow(pattern, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#777777")
    fig.suptitle("Patrones binarios de prueba (8 × 8)", fontsize=12)
    fig.tight_layout()
    save_to_output_dirs(fig, "vision_patrones_prueba.png")


def cartpole_crossbar_figure():
    weights_path = WEIGHTS
    if not weights_path.exists():
        raise FileNotFoundError(f"Saved CartPole weights not found: {weights_path}")
    metadata_path = Path(str(weights_path) + ".meta.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Weight metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("convention") != "X = 1 - x":
        raise ValueError("Saved CartPole weights do not use the declared X convention")
    variability = metadata.get("variability", {})
    if not variability.get("d2d_enabled") or not variability.get("c2c_enabled"):
        raise ValueError("CartPole weights must include D2D and C2C variability")
    initial_X = float(metadata["initial_X"])
    initial = np.clip(
        initial_X + np.random.default_rng(0).normal(0.0, 0.01, (24, 12)),
        0.0,
        1.0,
    )
    final = np.load(weights_path)
    if final.shape != initial.shape:
        raise ValueError(f"Expected a 24 x 12 crossbar matrix, got {final.shape}")
    delta = final - initial

    fig = plt.figure(figsize=(9.0, 3.3), constrained_layout=True)
    grid = fig.add_gridspec(1, 5, width_ratios=(1, 1, 0.045, 1, 0.045))
    axes = [
        fig.add_subplot(grid[0, index])
        for index in (0, 1, 3)
    ]
    initial_cbar_axis = fig.add_subplot(grid[0, 2])
    difference_cbar_axis = fig.add_subplot(grid[0, 4])
    common = {"cmap": "viridis", "vmin": 0.0, "vmax": 1.0, "aspect": "auto"}
    first = axes[0].imshow(initial, **common)
    axes[0].set_title("Estado inicial")
    second = axes[1].imshow(final, **common)
    axes[1].set_title("Tras 300 episodios")
    difference_limit = float(np.max(np.abs(delta)))
    third = axes[2].imshow(
        delta,
        cmap="coolwarm",
        vmin=-difference_limit,
        vmax=difference_limit,
        aspect="auto",
    )
    axes[2].set_title(r"Cambio $\Delta X$")

    for axis in axes:
        axis.set_xlabel("Columna / acción")
        axis.set_xticks([0, 3, 7, 11])
        axis.set_yticks([0, 5, 11, 17, 23])
    axes[0].set_ylabel("Fila del codificador")

    fig.colorbar(first, cax=initial_cbar_axis, label="Estado $X$")
    fig.colorbar(third, cax=difference_cbar_axis, label="$\\Delta X$")
    fig.suptitle(
        "Cambio del estado del crossbar CartPole (semilla 0)",
        fontsize=11,
    )
    save_to_output_dirs(fig, "cartpole_crossbar_estado.png")


if __name__ == "__main__":
    cartpole_test_figure()
    vision_patterns_figure()
    cartpole_crossbar_figure()
