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
from ..validation.lif_experimental_data import VC_T, VC_V, VOUT_T, VOUT_V


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

def _annotate_threshold(ax: plt.Axes, V_th: float, V_hold: float):
    """Dibuja líneas horizontales de referencia con etiquetas en voltios."""
    ax.axhline(V_th,  color=_C_THRESH, lw=1.4, ls="--", alpha=0.85,
               label=f"$V_{{th}}$ = {V_th:.2f} V")
    ax.axhline(V_hold, color="gray", lw=1.0, ls="-.", alpha=0.55,
               label=f"$V_{{hold}}$ = {V_hold:.2f} V")


def _mark_spikes(ax: plt.Axes, spike_times_s: np.ndarray,
                 V_th: float, color: str = _C_SPIKE):
    """Dibuja marcadores verticales en los instantes de spike."""
    for t_sp in spike_times_s:
        ax.axvline(t_sp * 1e3, color=color, lw=1.2, ls="-", alpha=0.55)
    if len(spike_times_s) > 0:
        ax.scatter(spike_times_s * 1e3,
                   np.full(len(spike_times_s), V_th),
                   color=color, s=60, zorder=5,
                   marker="^", label=f"Spike ({len(spike_times_s)})")


def _twin_current(ax: plt.Axes, time_ms: np.ndarray,
                  vin_V: np.ndarray) -> plt.Axes:
    """Añade un eje derecho con el voltaje de entrada Vin."""
    ax2 = ax.twinx()
    ax2.plot(time_ms, vin_V, color=_C_CURRENT,
             lw=1.6, ls="--", alpha=0.75, label="$V_{in}(t)$")
    ax2.set_ylabel("Entrada $V_{in}$ (V)", color=_C_CURRENT, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=_C_CURRENT, labelsize=8)
    ax2.set_ylim(-0.2, 5.5)
    return ax2


def _style_ax(ax: plt.Axes, xlabel: str = "Tiempo (ms)",
              ylabel: str = "Voltaje $V_c$ (V)", title: str = ""):
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
    """Panel 1: Rampa de voltaje — Vc aumenta con Vin."""
    p = res.params
    t_ms = res.time * 1e3

    ax.plot(t_ms, res.voltage, color=_C_VOLTAGE, lw=2.0, label="$V_c(t)$")
    _annotate_threshold(ax, p.V_th, p.V_hold)
    _mark_spikes(ax, res.spike_times, p.V_th)

    ax2 = _twin_current(ax, t_ms, res.current)

    _style_ax(ax, title="EXP-1: Rampa de voltaje — Vc aumenta con Vin")

    txt = (f"Spikes = {res.n_spikes}\n"
           f"$C$ = {p.C*1e9:.0f} nF\n"
           f"$R_s$ = {p.R_s/1e3:.0f} kOhm")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))


def _plot_exp2_threshold(ax: plt.Axes, res: LIFExperimentResult):
    """Panel 2: Umbral — Vin subumbral vs sobreumbral."""
    p = res.params
    t_ms = res.time * 1e3

    ax.plot(t_ms, res.voltage, color=_C_VOLTAGE, lw=2.0, label="$V_c(t)$", zorder=3)
    _annotate_threshold(ax, p.V_th, p.V_hold)
    _mark_spikes(ax, res.spike_times, p.V_th)

    # Colorear regiones
    n = len(t_ms)
    n_sub = int(0.20 / (res.time[1] - res.time[0]))
    n_pause = int(0.05 / (res.time[1] - res.time[0]))
    ax.axvspan(t_ms[0], t_ms[min(n_sub, n)-1],
               alpha=0.08, color=_C_SUBTH,   label="Subumbral (0.5 V)")
    ax.axvspan(t_ms[min(n_sub+n_pause, n-1)], t_ms[-1],
               alpha=0.08, color=_C_SUPRATH, label="Sobreumbral (5.0 V)")

    ax2 = _twin_current(ax, t_ms, res.current)

    _style_ax(ax, title="EXP-2: Umbral — Vin subumbral vs sobreumbral")

    txt = f"$V_{{th}}$ = {p.V_th:.2f} V\nSpikes = {res.n_spikes}"
    ax.text(0.02, 0.97, txt, transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))


def _plot_exp3_spike(ax: plt.Axes, res: LIFExperimentResult):
    """Panel 3: Spike único — Vc y Vout detallados."""
    p = res.params
    t_ms = res.time * 1e3

    ax.plot(t_ms, res.voltage, color=_C_VOLTAGE, lw=2.2, label="$V_c(t)$", zorder=3)
    ax.plot(t_ms, res.vout, color=_C_SPIKE, lw=2.0, label="$V_{out}(t)$ (Spike)", zorder=4)
    _annotate_threshold(ax, p.V_th, p.V_hold)
    _mark_spikes(ax, res.spike_times, p.V_th)

    ax2 = _twin_current(ax, t_ms, res.current)

    _style_ax(ax, title="EXP-3: Spike único — disparo y descarga física")

    txt = (f"Vspike máx = {res.vout.max():.2f} V\n"
           f"$R_0$ = {p.R_0/1e3:.2f} kOhm")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))


def _plot_exp4_train(ax: plt.Axes, res: LIFExperimentResult):
    """Panel 4: Tren de spikes físicos."""
    p = res.params
    t_ms = res.time * 1e3

    ax.plot(t_ms, res.voltage, color=_C_VOLTAGE, lw=1.6, label="$V_c(t)$", zorder=3)
    ax.plot(t_ms, res.vout, color=_C_SPIKE, lw=1.2, alpha=0.8, label="$V_{out}(t)$", zorder=4)
    _annotate_threshold(ax, p.V_th, p.V_hold)

    for t_sp in res.spike_times:
        ax.axvline(t_sp * 1e3, color=_C_SPIKE, lw=0.8, alpha=0.5)

    ax2 = _twin_current(ax, t_ms, res.current)

    _style_ax(ax, title="EXP-4: Tren de spikes físicos (Vin = 5 V)")

    txt = (f"Total spikes = {res.n_spikes}\n"
           f"Freq. media = {res.mean_firing_rate_Hz:.2f} Hz\n"
           f"Vspike máx = {res.vout.max():.2f} V")
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
def _plot_exp5_vc_comparison(ax: plt.Axes, res: LIFExperimentResult):
    """Panel 5: Comparación directa de Vc (Simulación vs Datos Digitalizados)."""
    p = res.params
    t_ms = res.time * 1e3
    ax.plot(t_ms, res.voltage, color=_C_VOLTAGE, lw=2.0, label="Simulado $V_c(t)$")
    ax.scatter(VC_T * 1e3, VC_V, color="red", marker="x", s=30, label="Digitalizado $V_c$ (Papel)", zorder=5)
    _annotate_threshold(ax, p.V_th, p.V_hold)
    _style_ax(ax, title="EXP-5: Comparación Vc (Simulado vs Experimental)")
    ax.set_xlim([0, 500])
    ax.set_ylim([-0.05, 1.2])


def _plot_exp5_vout_comparison(ax: plt.Axes, res: LIFExperimentResult):
    """Panel 6: Comparación directa de Vout (Simulación vs Datos Digitalizados)."""
    p = res.params
    t_ms = res.time * 1e3
    ax.plot(t_ms, res.vout, color=_C_SPIKE, lw=2.0, label="Simulado $V_{out}(t)$")
    ax.scatter(VOUT_T * 1e3, VOUT_V, color="red", marker="x", s=30, label="Digitalizado $V_{out}$ (Papel)", zorder=5)
    _style_ax(ax, ylabel="Voltaje $V_{out}$ (V)", title="EXP-5: Comparación Vout (Simulado vs Experimental)")
    ax.set_xlim([0, 500])
    ax.set_ylim([-0.05, 0.75])


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard completo
# ─────────────────────────────────────────────────────────────────────────────

def plot_lif_validation_dashboard(results: LIFValidationResults,
                                   save_path: Optional[str] = None,
                                   dpi: int = 150) -> plt.Figure:
    fig = plt.figure(figsize=(18, 16))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.45, wspace=0.38,
                           left=0.06, right=0.96,
                           top=0.92, bottom=0.06)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]

    _plot_exp1_ramp(axes[0],      results.ramp)
    _plot_exp2_threshold(axes[1], results.threshold)
    _plot_exp3_spike(axes[2],     results.single)
    _plot_exp4_train(axes[3],     results.train)
    _plot_exp5_vc_comparison(axes[4],   results.comparison)
    _plot_exp5_vout_comparison(axes[5], results.comparison)

    # Etiquetas de panel
    for i, (ax, label) in enumerate(zip(axes, ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)"])):
        ax.text(-0.05, 1.04, label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom")

    # Título principal
    p = results.single.params
    fig.suptitle(
        "Validación Neurona TSM-LIF Física — Paso 4: Neurona Aislada\n"
        rf"$C$ = {p.C*1e9:.0f} nF | "
        rf"$R_s$ = {p.R_s/1e3:.0f} kOhm | "
        rf"$V_{{th}}$ = {p.V_th:.4f} V | "
        rf"$V_{{hold}}$ = {p.V_hold:.4f} V | "
        rf"$R_{{0}}$ = {p.R_0:.1f} Ohm",
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
