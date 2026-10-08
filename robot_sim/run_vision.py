"""
Clasificador de patrones 8×8 con crossbar memristivo.
Prueba de concepto: crossbar 64×8, R-STDP, 8 clases.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from neurobot.crossbar_brain import CrossbarBrain

OUT_DIR = Path(__file__).resolve().parent / "outputs"
FIG_DIR = Path(__file__).resolve().parent / "figuras"
OUT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)


# ==================================================================
# Dataset: 8 patrones binarios 8×8
# ==================================================================
def make_patterns():
    P = np.zeros((8, 8, 8), dtype=float)

    # 0: horizontal
    P[0, 3, :] = 1.0

    # 1: vertical
    P[1, :, 3] = 1.0

    # 2: diagonal \
    for i in range(8):
        P[2, i, i] = 1.0

    # 3: diagonal /
    for i in range(8):
        P[3, i, 7 - i] = 1.0

    # 4: cruz
    P[4, 3, :] = 1.0
    P[4, :, 3] = 1.0

    # 5: marco
    P[5, 0, :] = 1.0
    P[5, 7, :] = 1.0
    P[5, :, 0] = 1.0
    P[5, :, 7] = 1.0

    # 6: X
    for i in range(8):
        P[6, i, i] = 1.0
        P[6, i, 7 - i] = 1.0

    # 7: cuadrado relleno
    P[7, 2:6, 2:6] = 1.0

    return P


# ==================================================================
# Clasificador
# ==================================================================
def run_classifier(patterns, episodes=5000, V_read=0.1, eta=5e-4,
                   baseline_window=200, seed=0, verbose=True):
    n_classes = patterns.shape[0]
    N = 64
    M = n_classes

    rng = np.random.default_rng(seed)
    brain = CrossbarBrain(N=N, M=M, seed=seed, scheme="V2")

    log = {
        "ep": [], "reward": [], "correct": [],
        "class_true": [], "class_pred": [],
        "X_mean": [], "baseline": [],
    }
    E_accum = np.zeros((N, M))
    baseline_hist = []

    t0 = time.time()
    for ep in range(episodes):
        idx = int(rng.integers(0, n_classes))
        image = patterns[idx].ravel()

        # Lectura analógica
        V_rows = image * V_read
        I_col, _ = brain.read(V_rows)

        # Softmax normalizada
        scale = np.mean(np.abs(I_col)) + 1e-30
        logits = I_col / scale
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        pred = int(np.argmax(p))

        correct = int(pred == idx)
        reward = 1.0 if correct else -1.0

        # Elegibilidad REINFORCE
        indicator = np.zeros(M)
        indicator[pred] = 1.0
        delta = indicator - p
        pre = (image > 0.5).astype(float)
        E_accum += np.outer(pre, delta)

        # Baseline
        baseline_hist.append(reward)
        window = baseline_hist[-baseline_window:]
        baseline = float(np.mean(window))
        signal = reward - baseline

        # Update (mismo signo que CartPole: negativo)
        dX = -eta * E_accum * signal
        max_grad = float(np.max(np.abs(dX)))
        if max_grad > 0.05:
            dX = dX * (0.05 / max_grad)
        brain.X = np.clip(brain.X + dX, 0.0, 1.0)
        E_accum.fill(0.0)

        # Log
        log["ep"].append(ep)
        log["reward"].append(reward)
        log["correct"].append(correct)
        log["class_true"].append(idx)
        log["class_pred"].append(pred)
        log["X_mean"].append(float(brain.X.mean()))
        log["baseline"].append(baseline)

        if verbose and (ep % 500 == 0 or ep == episodes - 1):
            acc = float(np.mean(log["correct"][-200:])) if ep >= 200 else float(np.mean(log["correct"]))
            print(f"[ep {ep:5d}]  acc={acc:.3f}  "
                  f"reward={np.mean(log['reward'][-200:]):+.3f}  "
                  f"X_mean={brain.X.mean():.4f}")

    elapsed = time.time() - t0
    acc_final = float(np.mean(log["correct"][-500:])) if episodes >= 500 else float(np.mean(log["correct"]))
    print(f"\n[resultado] accuracy últimos episodios = {acc_final:.3f}  "
          f"tiempo = {elapsed:.1f}s")

    return brain, log


# ==================================================================
# Figuras
# ==================================================================
def plot_dataset(patterns, path):
    fig, axes = plt.subplots(1, 8, figsize=(12, 2))
    for i, ax in enumerate(axes):
        ax.imshow(patterns[i], cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"Clase {i}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_learning(log, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    acc = np.array(log["correct"], dtype=float)
    window = 100
    if len(acc) >= window:
        smooth = np.convolve(acc, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(len(smooth)), smooth, lw=1.8, color="tab:blue")
    ax.axhline(1 / 8, ls="--", color="gray", label="random (1/8)")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Accuracy (media móvil 100)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Curva de aprendizaje — Visión 8×8")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==================================================================
# Main
# ==================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--eta", type=float, default=5e-4)
    args = ap.parse_args()

    patterns = make_patterns()

    print(f"[vision] seed={args.seed}  ep={args.episodes}  eta={args.eta}")
    plot_dataset(patterns, FIG_DIR / "vision_dataset.png")

    brain, log = run_classifier(
        patterns,
        episodes=args.episodes,
        eta=args.eta,
        seed=args.seed,
        verbose=True,
    )

    plot_learning(log, FIG_DIR / f"vision_learning_seed{args.seed}.png")

    # Matriz de confusión (últimos 500 episodios)
    last = slice(-500, None)
    true = np.array(log["class_true"][last])
    pred = np.array(log["class_pred"][last])
    conf = np.zeros((8, 8), dtype=int)
    for t, p in zip(true, pred):
        conf[t, p] += 1

    print("\n[matriz de confusión]")
    print(conf)

    # Guardar
    fname = f"run_vision_seed{args.seed}"
    (OUT_DIR / f"{fname}.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )
    np.save(OUT_DIR / f"{fname}_X.npy", brain.X)
    print(f"[ok] guardado en {OUT_DIR / fname}.json")


if __name__ == "__main__":
    main()
