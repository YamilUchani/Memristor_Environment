"""
visualization/plot_lif_validation.py
=====================================
Gráficas de validación de la neurona LIF aislada (Paso 4).

Genera un dashboard de cuatro paneles:

  Panel 1: EXP-1 — Rampa de corriente (V aumenta con I).
  Panel 2: EXP-2 — Umbral (subthreshold vs suprathreshold).
  Panel 3: EXP-3 — Spike único (disparo + reset).
  Panel 4: EXP-4 — Tren de spikes (curva f-I).

Estilo visual coherente con el resto del proyecto (fondo blanco,
paleta contenida, grid sutil, anotaciones informativas).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from typing import Optional
from ..simulations.lif_validation import LIFValidationResults, LIFExperimentResult


# ── Paleta del proyecto ───────────────────────────────────────────────────────
_C_VOLTAGE  = "#1565C0"   # azul oscuro — potencial de membrana
_C_CURRENT  = "#E65100"   # naranja  — corriente de entrada
_C_THRESH   = "#C62828"   # rojo     — umbral V_th
_C_RESET    = "#6A1B9A"   # violeta  — V_reset / E_L
_C_SPIKE    = "#2E7D32"   # verde    — marcador de spike
_C_SUBTH    = "#90CAF9"   # azul claro — zona subthreshold
_C_SUPRATH  = "#FFCC80"   # naranja claro — zona suprathreshold


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_threshold(ax: plt.Axes, V_th_mV: float,
                         V_reset_mV: float, E_L_mV: float):
    """Dibuja líneas horizontales de referencia con etiquetas."""
    ax.axhline(V_th_mV,    color=_C_THRESH, lw=1.4, ls="--", alpha=0.85,
               label=f"$V_{{th}}$ = {V_th_mV:.0f} mV")
    ax.axhline(V_reset_mV, color=_C_RESET,  lw=1.0, ls=":",  alpha=0.75,
               label=f"$V_{{reset}}$ = {V_reset_mV:.0f} mV")
    ax.axhline(E_L_mV,     color="gray",    lw=1.0, ls="-.", alpha=0.55,
               label=f"$E_L$ = {E_L_mV:.0f} mV")


def _mark_spikes(ax: plt.Axes, spike_times_s: np.ndarray,
                 V_th_mV: float, color: str = _C_SPIKE):
    """Dibuja marcadores verticales en los instantes de spike."""
    for t_sp in spike_times_s:
        ax.axvline(t_sp * 1e3, color=color, lw=1.2, ls="-", alpha=0.55)
    if len(spike_times_s) > 0:
        ax.scatter(spike_times_s * 1e3,
                   np.full(len(spike_times_s), V_th_mV),
                   color=color, s=60, zorder=5,
                   marker="^", label=f"Spike ({len(spike_times_s)})")


def _twin_current(ax: plt.Axes, time_ms: np.ndarray,
                  current_nA: np.ndarray) -> plt.Axes:
    """Añade un eje derecho con la corriente de entrada."""
    ax2 = ax.twinx()
    ax2.plot(time_ms, current_nA, color=_C_CURRENT,
             lw=1.6, ls="--", alpha=0.75, label="$I(t)$")
    ax2.set_ylabel("Corriente (nA)", color=_C_CURRENT, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=_C_CURRENT, labelsize=8)
    ax2.set_ylim(bottom=0)
    return ax2


def _style_ax(ax: plt.Axes, xlabel: str = "Tiempo (ms)",
              ylabel: str = "Potencial (mV)", title: str = ""):
    """Aplica el estilo común a un eje."""
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, color=_C_VOLTAGE, fontsize=10)
    ax.tick_params(axis="y", labelcolor=_C_VOLTAGE, labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.22, ls=":")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


# ─────────────────────────────────────────────────────────────────────────────
# Paneles individuales
# ─────────────────────────────────────────────────────────────────────────────

def _plot_exp1_ramp(ax: plt.Axes, res: LIFExperimentResult):
    """Panel 1: Rampa de corriente — V aumenta con I."""
    p = res.params
    t_ms = res.time * 1e3

    ax.plot(t_ms, res.voltage_mV, color=_C_VOLTAGE, lw=2.0, label="$V(t)$")
    _annotate_threshold(ax, p.V_th * 1e3, p.V_reset * 1e3, p.E_L * 1e3)
    _mark_spikes(ax, res.spike_times, p.V_th * 1e3)

    ax2 = _twin_current(ax, t_ms, res.current_nA)

    # Anotación explicativa
    ax.annotate("V sube con\ncorriente creciente",
                xy=(t_ms[len(t_ms)//3], res.voltage_mV[len(t_ms)//3]),
                xytext=(t_ms[len(t_ms)//4], (p.E_L + (p.V_th - p.E_L) * 0.3) * 1e3),
                arrowprops=dict(arrowstyle="->", color=_C_VOLTAGE, lw=1.2),
                fontsize=8, color=_C_VOLTAGE,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    _style_ax(ax, title="EXP-1: Rampa de corriente — V aumenta con I")

    # Texto de resultado
    txt = (f"Spikes = {res.n_spikes}\n"
           f"$\\tau_m$ = {p.tau_m*1e3:.0f} ms")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))


def _plot_exp2_threshold(ax: plt.Axes, res: LIFExperimentResult):
    """Panel 2: Umbral — subthreshold vs suprathreshold."""
    p = res.params
    t_ms = res.time * 1e3

    ax.plot(t_ms, res.voltage_mV, color=_C_VOLTAGE, lw=2.0, label="$V(t)$", zorder=3)
    _annotate_threshold(ax, p.V_th * 1e3, p.V_reset * 1e3, p.E_L * 1e3)
    _mark_spikes(ax, res.spike_times, p.V_th * 1e3)

    # Colorear regiones por tipo de corriente
    n = len(t_ms)
    n_sub   = int(0.20 / (res.time[1] - res.time[0]))
    n_pause = int(0.05 / (res.time[1] - res.time[0]))
    ax.axvspan(t_ms[0],        t_ms[min(n_sub, n)-1],
               alpha=0.08, color=_C_SUBTH,   label="Subthreshold (1 nA)")
    ax.axvspan(t_ms[min(n_sub+n_pause, n-1)], t_ms[-1],
               alpha=0.08, color=_C_SUPRATH, label="Suprathreshold (4 nA)")

    ax2 = _twin_current(ax, t_ms, res.current_nA)

    # Anotaciones
    ax.annotate("No dispara\n(I insuficiente)",
                xy=(100, (p.E_L + (p.V_th - p.E_L) * 0.5) * 1e3),
                fontsize=8, ha="center", color=_C_SUBTH,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    if res.n_spikes > 0:
        ax.annotate("DISPARO",
                    xy=(res.spike_times[0] * 1e3, p.V_th * 1e3),
                    xytext=(res.spike_times[0] * 1e3 + 30,
                            (p.V_th - 0.005) * 1e3),
                    arrowprops=dict(arrowstyle="->", color=_C_SPIKE, lw=1.2),
                    fontsize=9, fontweight="bold", color=_C_SPIKE)

    _style_ax(ax, title="EXP-2: Umbral — subthreshold vs suprathreshold")

    txt = f"$V_{{th}}$ = {p.V_th*1e3:.0f} mV\nSpikes = {res.n_spikes}"
    ax.text(0.02, 0.97, txt, transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))


def _plot_exp3_spike(ax: plt.Axes, res: LIFExperimentResult):
    """Panel 3: Spike único — disparo + reset detallado."""
    p = res.params
    t_ms = res.time * 1e3

    ax.plot(t_ms, res.voltage_mV, color=_C_VOLTAGE, lw=2.2, label="$V(t)$", zorder=3)
    _annotate_threshold(ax, p.V_th * 1e3, p.V_reset * 1e3, p.E_L * 1e3)
    _mark_spikes(ax, res.spike_times, p.V_th * 1e3)

    ax2 = _twin_current(ax, t_ms, res.current_nA)

    # Anotar las cuatro fases
    t_pulse_end_ms = int(0.030 / (res.time[1] - res.time[0])) * (res.time[1] - res.time[0]) * 1e3
    ax.annotate("1. Integracion\n   (V sube)",
                xy=(t_ms[10], res.voltage_mV[10]),
                xytext=(20, (p.E_L + 0.004) * 1e3),
                arrowprops=dict(arrowstyle="->", color=_C_VOLTAGE, lw=1.0),
                fontsize=7.5, color=_C_VOLTAGE,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

    if res.n_spikes > 0:
        t_sp_ms = res.spike_times[0] * 1e3
        ax.annotate("2. Disparo\n   (V ≥ V_th)",
                    xy=(t_sp_ms, p.V_th * 1e3),
                    xytext=(t_sp_ms + 15, p.V_th * 1e3 + 3),
                    arrowprops=dict(arrowstyle="->", color=_C_SPIKE, lw=1.0),
                    fontsize=7.5, color=_C_SPIKE,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

        ax.annotate("3. Reset\n   (V ← V_reset)",
                    xy=(t_sp_ms + 1, p.V_reset * 1e3),
                    xytext=(t_sp_ms + 20, p.V_reset * 1e3 - 6),
                    arrowprops=dict(arrowstyle="->", color=_C_RESET, lw=1.0),
                    fontsize=7.5, color=_C_RESET,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

    ax.annotate("4. Decaimiento\n   (V → E_L)",
                xy=(t_ms[-80], res.voltage_mV[-80]),
                xytext=(t_ms[-80] - 50, (p.E_L + 0.008) * 1e3),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
                fontsize=7.5, color="gray",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

    _style_ax(ax, title="EXP-3: Spike unico — disparo, reset y decaimiento")


def _plot_exp4_train(ax: plt.Axes, res: LIFExperimentResult):
    """Panel 4: Tren de spikes — curva f-I."""
    p = res.params
    t_ms = res.time * 1e3
    T_total_ms = t_ms[-1]
    T_seg_ms = T_total_ms / 3.0

    ax.plot(t_ms, res.voltage_mV, color=_C_VOLTAGE, lw=1.6, label="$V(t)$", zorder=3)
    _annotate_threshold(ax, p.V_th * 1e3, p.V_reset * 1e3, p.E_L * 1e3)

    # Colores por segmento
    seg_colors  = ["#BBDEFB", "#90CAF9", "#1565C0"]
    seg_labels  = ["I = 2 nA", "I = 4 nA", "I = 7 nA"]
    seg_alphas  = [0.12, 0.12, 0.08]
    for i in range(3):
        ax.axvspan(i * T_seg_ms, (i+1) * T_seg_ms,
                   alpha=seg_alphas[i], color=seg_colors[i],
                   label=seg_labels[i])

    # Spikes
    for t_sp in res.spike_times:
        ax.axvline(t_sp * 1e3, color=_C_SPIKE, lw=0.8, alpha=0.5)

    ax2 = _twin_current(ax, t_ms, res.current_nA)

    # Calcular frecuencias por segmento
    dt = res.time[1] - res.time[0]
    T_seg = res.time[-1] / 3.0
    freqs = []
    for i in range(3):
        t0, t1 = i * T_seg, (i+1) * T_seg
        n_sp = np.sum((res.spike_times >= t0) & (res.spike_times < t1))
        freqs.append(n_sp / T_seg)

    # Anotaciones de frecuencia
    I_levels = [2, 4, 7]
    for i, (f, I) in enumerate(zip(freqs, I_levels)):
        mid_ms = (i + 0.5) * T_seg_ms
        ax.text(mid_ms, (p.V_th - 0.003) * 1e3,
                f"{f:.1f} Hz",
                ha="center", fontsize=8.5, fontweight="bold",
                color=seg_colors[i] if i < 2 else _C_VOLTAGE,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

    _style_ax(ax, title="EXP-4: Tren de spikes — Curva f-I")

    txt = (f"Total spikes = {res.n_spikes}\n"
           f"Freq: {freqs[0]:.1f} / {freqs[1]:.1f} / {freqs[2]:.1f} Hz")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard completo
# ─────────────────────────────────────────────────────────────────────────────

def plot_lif_validation_dashboard(results: LIFValidationResults,
                                   save_path: Optional[str] = None,
                                   dpi: int = 150) -> plt.Figure:
    """
    Genera el dashboard de validación LIF con los cuatro experimentos.

    Parámetros
    ----------
    results : LIFValidationResults
        Resultados de LIFValidation.run().
    save_path : str, opcional
        Ruta para guardar la figura PNG.
    dpi : int
        Resolución de la figura guardada.

    Retorna
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.42, wspace=0.38,
                           left=0.06, right=0.96,
                           top=0.91, bottom=0.07)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    _plot_exp1_ramp(axes[0],      results.ramp)
    _plot_exp2_threshold(axes[1], results.threshold)
    _plot_exp3_spike(axes[2],     results.single)
    _plot_exp4_train(axes[3],     results.train)

    # Etiquetas de panel
    for i, (ax, label) in enumerate(zip(axes, ["(A)", "(B)", "(C)", "(D)"])):
        ax.text(-0.05, 1.04, label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom")

    # Título principal
    p = results.single.params
    fig.suptitle(
        "Validacion Neurona LIF — Paso 4: Neurona Aislada\n"
        rf"$\tau_m$ = {p.tau_m*1e3:.0f} ms | "
        rf"$V_{{th}}$ = {p.V_th*1e3:.0f} mV | "
        rf"$E_L$ = {p.E_L*1e3:.0f} mV | "
        rf"$t_{{ref}}$ = {p.t_ref*1e3:.0f} ms",
        fontsize=13, fontweight="bold", y=0.97
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white")
        print(f"  [PNG] Guardado: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Script standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from memristor_simulator.simulations.lif_validation import LIFValidation

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output_modular")
    os.makedirs(out_dir, exist_ok=True)

    results = LIFValidation().run(verbose=True)
    save = os.path.join(out_dir, "lif_validation_dashboard.png")
    plot_lif_validation_dashboard(results, save_path=save, dpi=150)
    plt.show()
