"""
Suite de pruebas del crossbar aislado.
Cada test es independiente y guarda su figura + JSON.

Uso:
    python test_crossbar_suite.py                    # corre todos
    python test_crossbar_suite.py --test sneak_N     # corre uno
    python test_crossbar_suite.py --list             # lista tests
"""
import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --- path setup ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurobot.crossbar_brain import CrossbarBrain
from neurobot import lab_bridge

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 200,
})


# ==================================================================
# Utilidades
# ==================================================================
def make_brain(N, M, seed=0, json_nv="strukov_ideal.json", json_lif="lif_config.json"):
    """
    Crossbar para tests aislados.
    R_sense = 1e-3 Ohm para que la corriente dependa de G y no de la
    impedancia de sensado. En el agente (run_robot.py) se mantiene
    R_sense=1e2 para preservar los resultados ya obtenidos.
    """
    return CrossbarBrain(
        N=N, M=M, seed=seed,
        R_sense=1e-3,          # <<< override para tests del crossbar
        json_memristor=json_nv,
        json_lif=json_lif,
    )


def save_json(name, data):
    p = FIG / name
    p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"  [json] {p.name}")


def save_fig(fig, name):
    p = FIG / name
    fig.tight_layout()
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig]  {p.name}")


# ==================================================================
# TEST 1 - Sneak ratio vs N
# ==================================================================
def test_sneak_N():
    print("\n[TEST 1] Sneak ratio vs N")
    Ns = [4, 8, 12, 16, 24, 32]
    M = 12
    ratios = []

    for N in Ns:
        b = make_brain(N, M, seed=42)
        b.X = np.random.default_rng(N).uniform(0.05, 0.95, (N, M))
        V = np.full(N, 0.1)
        _, sneak = b.read(V)
        ratios.append(sneak)
        print(f"  N={N:3d}  sneak={sneak:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Ns, ratios, "o-", color="tab:blue", lw=2, ms=8)
    ax.set_xlabel("Tamano del crossbar (N filas, M=12)")
    ax.set_ylabel("Sneak ratio")
    ax.set_title("Degradacion por corrientes parasitas vs tamano")
    ax.grid(alpha=0.3)
    for x, y in zip(Ns, ratios):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    save_fig(fig, "exp01_sneak_vs_N.png")
    save_json("exp01_sneak_vs_N.json", {"N": Ns, "sneak_ratio": ratios})


# ==================================================================
# TEST 2 - MNA real vs MNA ideal
# ==================================================================
def test_mna_vs_ideal():
    print("\n[TEST 2] MNA real vs ideal")
    Ns = [4, 8, 16, 24]
    M = 12
    n_samples = 100
    results = {}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, N in zip(axes.ravel(), Ns):
        b = make_brain(N, M, seed=42)
        b.X = np.random.default_rng(N).uniform(0.1, 0.9, (N, M))

        rng = np.random.default_rng(0)
        errors = []
        I_mna_all, I_ideal_all = [], []

        for _ in range(n_samples):
            V = rng.uniform(-0.1, 0.1, N)
            I_mna, _ = b.read(V)
            I_ideal = b.G.T @ V
            err = np.linalg.norm(I_mna - I_ideal) / (np.linalg.norm(I_ideal) + 1e-30)
            errors.append(err)
            I_mna_all.extend(I_mna)
            I_ideal_all.extend(I_ideal)

        err_mean = float(np.mean(errors))
        results[f"N={N}"] = {"error_mean": err_mean,
                             "error_std": float(np.std(errors))}

        ax.scatter(I_ideal_all, I_mna_all, s=8, alpha=0.5, color="tab:blue")
        lim = max(max(np.abs(I_ideal_all)), max(np.abs(I_mna_all))) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y = x")
        ax.set_title(f"N = {N}  (error medio {err_mean*100:.1f}%)")
        ax.set_xlabel("I ideal (A)")
        ax.set_ylabel("I MNA (A)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("MNA real vs ideal - efecto de sneak paths", fontsize=12)
    save_fig(fig, "exp02_mna_vs_ideal.png")
    save_json("exp02_mna_vs_ideal.json", results)


# ==================================================================
# TEST 3 - Efecto de R_wire (IR drops)
# ==================================================================
def test_rwire():
    print("\n[TEST 3] Efecto de R_wire")
    N, M = 16, 12
    R_wires = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    results = {}

    fig, ax = plt.subplots(figsize=(8, 5))

    for Rw in R_wires:
        b = make_brain(N, M, seed=42)
        b.R_wire = Rw
        b.X = np.random.default_rng(1).uniform(0.1, 0.9, (N, M))
        V = np.full(N, 0.1)
        I_col, _ = b.read(V)

        ax.plot(range(M), I_col * 1e6, "o-", lw=1.8, ms=5,
                label=f"R_wire = {Rw:.0e} Ohm")
        results[f"R_wire={Rw}"] = I_col.tolist()

    ax.set_xlabel("Columna")
    ax.set_ylabel("I_col (uA)")
    ax.set_title(f"IR drops: efecto de R_wire (N={N}, M={M})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    save_fig(fig, "exp03_rwire_effect.png")
    save_json("exp03_rwire_effect.json", results)


# ==================================================================
# TEST 4 - Precision de multiplicacion matriz-vector
# ==================================================================
def test_matvec():
    print("\n[TEST 4] Precision VMM")
    Ns = [4, 8, 16, 24]
    M = 12
    n_samples = 1000
    results = {}

    fig, ax = plt.subplots(figsize=(8, 5))

    for N in Ns:
        b = make_brain(N, M, seed=42)
        b.X = np.random.default_rng(N).uniform(0.1, 0.9, (N, M))
        rng = np.random.default_rng(0)
        errors = []
        for _ in range(n_samples):
            V = rng.uniform(-0.1, 0.1, N)
            I_mna, _ = b.read(V)
            I_ideal = b.G.T @ V
            err = np.linalg.norm(I_mna - I_ideal) / (np.linalg.norm(I_ideal) + 1e-30)
            errors.append(err * 100)

        ax.hist(errors, bins=40, alpha=0.6, label=f"N={N}  (media {np.mean(errors):.1f}%)")
        results[f"N={N}"] = {
            "mean_pct": float(np.mean(errors)),
            "std_pct": float(np.std(errors)),
        }

    ax.set_xlabel("Error relativo (%)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribucion del error en la multiplicacion matriz-vector")
    ax.grid(alpha=0.3)
    ax.legend()
    save_fig(fig, "exp04_vmm_precision.png")
    save_json("exp04_vmm_precision.json", results)


# ==================================================================
# TEST 5 - V/2 vs V/3 (half-select disturb)
# ==================================================================
def test_v2_vs_v3():
    """
    Mide directamente el delta_x fisico de las celdas half-selected vs
    las fully-selected tras N pulsos. Muestra las curvas temporales de
    evolucion de x para los tres tipos de celda.

    Nota: el modelo Strukov lineal produce delta_x proporcional a V,
    por lo que el disturb de V/2 es ~50% del de V_write y el de V/3 es ~33%.
    Esto es exactamente el resultado esperado por teoria.
    """
    print("\n[TEST 5] V/2 vs V/3")
    V_write = 2.0
    n_pulses = 3    # 3 pulsos: evita saturacion (x satura en <10 pulsos con 2V)
    dt = 1e-3
    results = {}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for idx, scheme in enumerate(["V2", "V3"]):
        V_half = V_write / 2.0 if scheme == "V2" else V_write / 3.0

        # 3 tipos de celda en el crossbar
        cell_target = lab_bridge.make_memristor(filename="strukov_ideal.json")
        cell_half   = lab_bridge.make_memristor(filename="strukov_ideal.json")
        cell_none   = lab_bridge.make_memristor(filename="strukov_ideal.json")

        x0 = float(cell_target.x)
        X_target, X_half, X_none = [], [], []

        for _ in range(n_pulses):
            cell_target.update(voltage=V_write, dt=dt)
            cell_half.update(voltage=V_half,   dt=dt)
            cell_none.update(voltage=0.0,       dt=dt)
            X_target.append(float(cell_target.x))
            X_half.append(float(cell_half.x))
            X_none.append(float(cell_none.x))

        delta_target = abs(X_target[-1] - x0)
        delta_half   = abs(X_half[-1]   - x0)
        delta_none   = abs(X_none[-1]   - x0)
        # disturb_pct: cuanto pertuba half-select RELATIVO al cambio del target
        disturb_pct  = delta_half / (delta_target + 1e-12) * 100

        results[scheme] = {
            "V_half":        float(V_half),
            "delta_target":  float(delta_target),
            "delta_half":    float(delta_half),
            "delta_none":    float(delta_none),
            "disturb_pct":   float(disturb_pct),
            "selectivity":   float(delta_target / (delta_half + 1e-12)),
        }
        print(f"  [{scheme}] V_half={V_half:.3f}V | "
              f"target={delta_target:.4f}  half={delta_half:.4f}  none={delta_none:.2e}  "
              f"disturb={disturb_pct:.1f}%  selectividad={results[scheme]['selectivity']:.1f}x")

        ax = axes[idx]
        ax.plot(X_target, color="tab:red",  lw=2.0, label=f"Fully-selected ({V_write:.1f} V)")
        ax.plot(X_half,   color="tab:orange", lw=1.8, ls="--", label=f"Half-selected ({V_half:.2f} V)")
        ax.plot(X_none,   color="tab:blue",  lw=1.2, ls=":",  label="No-selected (0 V)")
        ax.axhline(x0, color="k", lw=0.7, ls="--", alpha=0.5, label=f"x0 = {x0:.3f}")
        ax.set_xlabel("Pulso")
        ax.set_ylabel("Variable de estado x")
        ax.set_title(f"Esquema {scheme}\ndisturb = {disturb_pct:.1f}% | selectividad = {results[scheme]['selectivity']:.1f}x")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Panel 3: comparativa disturb y selectividad
    ax = axes[2]
    schemes = ["V2", "V3"]
    disturbs      = [results[s]["disturb_pct"]  for s in schemes]
    selectivities = [results[s]["selectivity"]   for s in schemes]
    x = np.arange(2)
    ax2 = ax.twinx()
    bars = ax.bar(x - 0.15, disturbs, width=0.3,
                  label="Disturb (%)", color="tab:red", alpha=0.8)
    line, = ax2.plot(x, selectivities, "D-", color="tab:blue", ms=10, lw=2,
                     label="Selectividad (target/half)")
    for bar, v in zip(bars, disturbs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                f"{v:.1f}%", ha="center", fontsize=9, color="tab:red")
    for xi, sv in zip(x, selectivities):
        ax2.text(xi + 0.15, sv + 0.02, f"{sv:.1f}x", ha="center", fontsize=9, color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(schemes)
    ax.set_ylabel("Disturb relativo (%)", color="tab:red")
    ax2.set_ylabel("Selectividad (x)", color="tab:blue")
    ax.set_title("V/2 vs V/3\nDisturb y Selectividad")
    handles = [bars, line]
    labels  = ["Disturb (%)", "Selectividad"]
    ax.legend(handles, labels, fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Half-select disturb: V/2 vs V/3 (modelo fisico Strukov)", fontsize=12)
    save_fig(fig, "exp05_v2_vs_v3.png")
    save_json("exp05_v2_vs_v3.json", results)


# ==================================================================
# TEST 6 - Retencion de estado (no volatilidad)
# ==================================================================
def test_retention():
    print("\n[TEST 6] Retencion de estado")
    N, M = 8, 8
    b = make_brain(N, M, seed=42)
    b.X = np.random.default_rng(0).uniform(0.2, 0.8, (N, M))

    X0 = b.X.copy()
    n_steps = 1000
    X_means = []

    V_zero = np.zeros(N)
    for _ in range(n_steps):
        b.read(V_zero)
        X_means.append(float(b.X.mean()))

    delta = float(np.abs(b.X - X0).max())
    status = "RETENCION OK" if delta < 1e-10 else "ALERTA: deriva"
    print(f"  Cambio maximo tras {n_steps} pasos en reposo: {delta:.2e}")
    print(f"  {status}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(X_means, lw=1.5, color="tab:blue")
    ax.set_xlabel("Paso de simulacion")
    ax.set_ylabel("Media de X")
    ax.set_title(f"Retencion de estado (delta_max = {delta:.2e}) - {status}")
    ax.grid(alpha=0.3)
    ax.set_ylim(X0.mean() - 0.01, X0.mean() + 0.01)
    save_fig(fig, "exp06_retention.png")
    save_json("exp06_retention.json", {"delta_max": delta, "n_steps": n_steps,
                                       "status": status})


# ==================================================================
# TEST 7 - Linealidad de LTP/LTD
# ==================================================================
def test_ltp_ltd():
    print("\n[TEST 7] Linealidad LTP/LTD")
    n_pulses = 50
    V_ltp = 1.0
    V_ltd = -1.0
    dt = 1e-3

    X_ltp, X_ltd = [], []

    # LTP: desde x0
    cell_ltp = lab_bridge.make_memristor(filename="strukov_ideal.json")
    for _ in range(n_pulses):
        cell_ltp.update(voltage=V_ltp, dt=dt)
        X_ltp.append(float(cell_ltp.x))

    # LTD: desde x0
    cell_ltd = lab_bridge.make_memristor(filename="strukov_ideal.json")
    for _ in range(n_pulses):
        cell_ltd.update(voltage=V_ltd, dt=dt)
        X_ltd.append(float(cell_ltd.x))

    print(f"  LTP: x0={lab_bridge.make_memristor('strukov_ideal.json').x:.3f}  "
          f"x_final={X_ltp[-1]:.4f}")
    print(f"  LTD: x0={lab_bridge.make_memristor('strukov_ideal.json').x:.3f}  "
          f"x_final={X_ltd[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(X_ltp, "o-", color="tab:red", label=f"LTP (+{V_ltp}V)", lw=1.8, ms=4)
    axes[0].plot(X_ltd, "s-", color="tab:blue", label=f"LTD ({V_ltd}V)", lw=1.8, ms=4)
    axes[0].set_xlabel("Numero de pulsos")
    axes[0].set_ylabel("Variable de estado x")
    axes[0].set_title("Evolucion de x bajo pulsos LTP/LTD")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Calcular el incremento diferencial por pulso
    dx_ltp = np.diff([lab_bridge.make_memristor("strukov_ideal.json").x] + X_ltp)
    dx_ltd = np.diff([lab_bridge.make_memristor("strukov_ideal.json").x] + X_ltd)
    axes[1].plot(dx_ltp, "o-", color="tab:red", label="dX por pulso (LTP)", lw=1.5, ms=4)
    axes[1].plot(dx_ltd, "s-", color="tab:blue", label="dX por pulso (LTD)", lw=1.5, ms=4)
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].set_xlabel("Numero de pulso")
    axes[1].set_ylabel("dX (cambio incremental)")
    axes[1].set_title("No-linealidad potenciacion/depresion")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    save_fig(fig, "exp07_ltp_ltd.png")
    save_json("exp07_ltp_ltd.json", {"X_ltp": X_ltp, "X_ltd": X_ltd,
                                     "dx_ltp": dx_ltp.tolist(), "dx_ltd": dx_ltd.tolist()})


# ==================================================================
# TEST 8 - Reproducibilidad D2D (variacion proceso a proceso)
# ==================================================================
def test_d2d():
    """
    Simula variacion D2D realista a dos niveles:
    1. Variacion wafer-a-wafer: cada crossbar tiene un x0 medio diferente.
    2. Variacion celda-a-celda: ruido gaussiano de proceso sobre x0.

    La metrica de CV se calcula relativa al rango dinamico de conductancia
    (G_max - G_min) del dispositivo, que es la magnitud fisicamente significativa.
    """
    print("\n[TEST 8] Reproducibilidad D2D")
    N, M = 8, 8
    n_crossbars = 20
    sigma_proc = 0.05   # dispersion de proceso tipica: ~5-10% del rango de x
    V = np.full(N, 0.1)
    I_cols  = []
    G_means = []
    x0_means = []

    # Obtenemos G_range del dispositivo para normalizar
    b_ref = make_brain(N, M, seed=0)
    G_range = b_ref.G_max - b_ref.G_min

    rng_proc = np.random.default_rng(999)
    for k in range(n_crossbars):
        b = make_brain(N, M, seed=k)
        # Variacion wafer-a-wafer: x0 nominal diferente por crossbar
        x0_nominal = rng_proc.uniform(0.2, 0.8)
        # Variacion celda-a-celda: ruido de proceso gaussiano
        b.X = np.clip(
            x0_nominal + rng_proc.normal(0, sigma_proc, (N, M)),
            0.01, 0.99
        )
        x0_means.append(float(b.X.mean()))
        G_means.append(float(b.G.mean()))
        I_col, _ = b.read(V)
        I_cols.append(I_col)

    I_cols  = np.asarray(I_cols)
    G_means = np.asarray(G_means)
    I_mean  = I_cols.mean(axis=0)
    I_std   = I_cols.std(axis=0)

    # CV normalizado por G_range (mas significativo que sobre I_mean)
    G_std_per_col = I_std / V[0]   # I = G*V => sigma_G = sigma_I / V
    cv_normalized = float((G_std_per_col / G_range).mean() * 100)
    cv_mean = float((I_std / (np.abs(I_mean) + 1e-30)).mean() * 100)
    print(f"  CV medio (I_col):            {cv_mean:.2f}%")
    print(f"  CV normalizado por G_range:  {cv_normalized:.1f}% del rango dinamico")
    print(f"  x0 nominal range: [{min(x0_means):.3f}, {max(x0_means):.3f}]")
    print(f"  G_range del dispositivo: {G_range:.2e} S  (muy comprimido -> efecto D2D pequeno)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: dispersion de I_col por columna
    x = np.arange(M)
    for k in range(n_crossbars):
        axes[0].plot(x, I_cols[k] * 1e6, alpha=0.2, color="gray", lw=0.8)
    axes[0].errorbar(x, I_mean * 1e6, yerr=I_std * 1e6,
                     fmt="o-", capsize=5, color="tab:blue", lw=2, ms=6, zorder=5,
                     label=f"Media +/- std  (CV={cv_mean:.1f}%)")
    axes[0].set_xlabel("Columna")
    axes[0].set_ylabel("I_col (uA)")
    axes[0].set_title(f"Dispersion I_col: {n_crossbars} crossbars (D2D)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Panel 2: G_mean por crossbar (muestra la variacion wafer-a-wafer)
    axes[1].plot(range(n_crossbars), G_means * 1e3, "o-", color="tab:orange",
                 lw=1.5, ms=6)
    axes[1].axhline(np.mean(G_means) * 1e3, color="k", ls="--", lw=1,
                    label=f"Media={np.mean(G_means)*1e3:.3f} mS")
    axes[1].fill_between(
        range(n_crossbars),
        [(np.mean(G_means) - np.std(G_means)) * 1e3] * n_crossbars,
        [(np.mean(G_means) + np.std(G_means)) * 1e3] * n_crossbars,
        alpha=0.2, color="tab:orange", label=f"Banda +/- std"
    )
    axes[1].set_xlabel("Crossbar # (wafer virtual)")
    axes[1].set_ylabel("G_mean (mS)")
    axes[1].set_title(f"Variacion G_mean entre crossbars\nCV_G_range = {cv_normalized:.1f}%")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle(f"Variacion D2D - sigma_proc={sigma_proc} | G_range={G_range:.2e} S", fontsize=11)
    save_fig(fig, "exp08_d2d_reproducibility.png")
    save_json("exp08_d2d_reproducibility.json", {
        "I_mean":           I_mean.tolist(),
        "I_std":            I_std.tolist(),
        "cv_mean_pct":      cv_mean,
        "cv_G_range_pct":   cv_normalized,
        "G_range":          float(G_range),
        "x0_means":         x0_means,
        "sigma_proc":       sigma_proc,
    })


# ==================================================================
# TEST 9 - Costo computacional
# ==================================================================
def test_cost():
    print("\n[TEST 9] Costo computacional del solver")
    Ns = [4, 8, 12, 16, 24, 32, 48]
    M = 12
    n_repeat = 50
    times = []

    for N in Ns:
        b = make_brain(N, M, seed=0)
        b.X = np.random.default_rng(0).uniform(0.1, 0.9, (N, M))
        V = np.full(N, 0.1)
        # warm-up
        b.read(V)
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            b.read(V)
        elapsed = (time.perf_counter() - t0) / n_repeat * 1e3
        times.append(elapsed)
        print(f"  N={N:3d}  {elapsed:.3f} ms/paso")

    # Fit O(N^p) en log-log
    coeffs = np.polyfit(np.log(Ns), np.log(times), 1)
    p_fit = coeffs[0]
    print(f"  Exponente de escala ajustado: O(N^{p_fit:.2f})")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(Ns, times, "o-", color="tab:purple", lw=2, ms=8, label="Medido")
    Ns_fit = np.linspace(Ns[0], Ns[-1], 100)
    ax.loglog(Ns_fit, np.exp(coeffs[1]) * Ns_fit**p_fit, "--",
              color="tab:gray", lw=1.5, label=f"Ajuste O(N^{p_fit:.2f})")
    ax.set_xlabel("N (numero de filas)")
    ax.set_ylabel("Tiempo por paso (ms)")
    ax.set_title("Costo computacional del solver MNA")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    save_fig(fig, "exp09_computational_cost.png")
    save_json("exp09_computational_cost.json",
              {"N": Ns, "time_ms": times, "fit_exponent": float(p_fit)})


# ==================================================================
# TEST 10 - Crosstalk entre columnas
# ==================================================================
def test_crosstalk():
    print("\n[TEST 10] Crosstalk entre columnas")
    N, M = 12, 12
    b = make_brain(N, M, seed=42)
    b.X = np.random.default_rng(0).uniform(0.3, 0.7, (N, M))

    # Excitar solo la fila 0
    V = np.zeros(N)
    V[0] = 0.1
    I_col, _ = b.read(V)
    I_ideal = b.G[0, :] * V[0]
    ratios = I_col / (I_ideal + 1e-30)

    xtalk_mean = float(np.mean(np.abs(I_col - I_ideal) / (np.abs(I_ideal) + 1e-30)) * 100)
    print(f"  Error de crosstalk medio: {xtalk_mean:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(M)
    axes[0].bar(x - 0.2, I_ideal * 1e6, width=0.4, label="Ideal (solo fila 0)",
                color="tab:green", alpha=0.8)
    axes[0].bar(x + 0.2, I_col * 1e6, width=0.4, label="MNA (con sneak)",
                color="tab:red", alpha=0.8)
    axes[0].set_xlabel("Columna")
    axes[0].set_ylabel("I_col (uA)")
    axes[0].set_title("Crosstalk: corrientes ideal vs MNA")
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].legend()

    axes[1].bar(x, (ratios - 1) * 100, color=["tab:red" if r > 1 else "tab:blue" for r in ratios],
                alpha=0.8)
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_xlabel("Columna")
    axes[1].set_ylabel("Error relativo (%)")
    axes[1].set_title(f"Desviacion porcentual por sneak path (media={xtalk_mean:.1f}%)")
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("Crosstalk entre columnas con una sola fila excitada", fontsize=12)
    save_fig(fig, "exp10_crosstalk.png")
    save_json("exp10_crosstalk.json", {
        "I_ideal": I_ideal.tolist(),
        "I_mna": I_col.tolist(),
        "ratio": ratios.tolist(),
        "xtalk_mean_pct": xtalk_mean,
    })


# ==================================================================
# Registry + CLI
# ==================================================================
TESTS = {
    "sneak_N":       test_sneak_N,
    "mna_vs_ideal":  test_mna_vs_ideal,
    "rwire":         test_rwire,
    "vmm_precision": test_matvec,
    "v2_vs_v3":      test_v2_vs_v3,
    "retention":     test_retention,
    "ltp_ltd":       test_ltp_ltd,
    "d2d":           test_d2d,
    "cost":          test_cost,
    "crosstalk":     test_crosstalk,
}


def main():
    ap = argparse.ArgumentParser(description="Suite de pruebas del crossbar memristivo")
    ap.add_argument("--test", default=None,
                    help="Nombre del test especifico a correr")
    ap.add_argument("--list", action="store_true", help="Listar todos los tests disponibles")
    args = ap.parse_args()

    if args.list:
        print("Tests disponibles:")
        for k in TESTS:
            print(f"  - {k}")
        return

    if args.test:
        if args.test not in TESTS:
            print(f"Test '{args.test}' no existe. Opciones: {list(TESTS.keys())}")
            sys.exit(1)
        TESTS[args.test]()
    else:
        t0 = time.time()
        failed = []
        for name, fn in TESTS.items():
            try:
                fn()
            except Exception as e:
                import traceback
                print(f"  [ERROR] {name}: {e}")
                traceback.print_exc()
                failed.append(name)
        elapsed = time.time() - t0
        ok = len(TESTS) - len(failed)
        print(f"\n[listo] {ok}/{len(TESTS)} tests OK en {elapsed:.1f} s")
        if failed:
            print(f"[fallidos] {failed}")
        print(f"[salida] {FIG}")


if __name__ == "__main__":
    main()
