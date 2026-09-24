"""
visualization/plot_hysteresis.py
==================================
Gráficas de curvas I-V (histéresis) del memristor Strukov (2008).

Reproduce el estilo visual del paper de Nature:
  - Ejes limpios con labels dimensionados
  - Color por ciclo (Fig 2c numeración 1-6)
  - Área sombreada del lazo
  - Anotaciones de Δ(w/D) para análisis de colapso
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional
from ..simulations.iv_characterization import SimulationResult
from ..validation.metrics import hysteresis_area


def plot_iv_hysteresis(result: SimulationResult,
                        ax: plt.Axes = None,
                        color: str = "k",
                        title: str = None,
                        show_area: bool = True,
                        show_stats: bool = True,
                        v_ref: np.ndarray = None,
                        i_ref: np.ndarray = None,
                        scale_sim_current: float = 1.0) -> plt.Figure:
    """
    Grafica la curva I-V (lazo de histéresis) de un resultado de simulacion.
    Opcionalmente añade una curva de referencia.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    v = result.voltage
    i = result.current_mA * scale_sim_current

    if v_ref is not None and i_ref is not None:
        ax.plot(v_ref, i_ref, color="red", lw=2.0, ls="--", alpha=0.7, zorder=2, label="Paper Data (Truth)")

    ax.plot(v, i, color=color, lw=2.5, zorder=3, label="Simulation" + (f" (x{scale_sim_current})" if scale_sim_current != 1.0 else ""))
    if show_area:
        ax.fill(v, i, alpha=0.08, color=color, zorder=1)
    
    if v_ref is not None:
        ax.legend()

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)

    ax.set_xlabel("Voltaje (V)", fontsize=11)
    ax.set_ylabel("Corriente (mA)", fontsize=11)
    ax.grid(True, alpha=0.25, ls=":")

    label = title or result.label or "Curva I-V"
    ax.set_title(label, fontsize=12, fontweight="bold")

    if show_stats:
        dx = result.state_variable.max() - result.state_variable.min()
        area = hysteresis_area(v, result.current)
        txt = (f"$\\Delta(x)$ = {dx:.4f}\n"
               f"Area = {area:.2e} V·A")
        ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                fontsize=9, va="top", ha="right",
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

    if standalone:
        plt.tight_layout()
    return fig


def plot_multi_frequency(freq_results: Dict[str, SimulationResult],
                          save_path: str = None,
                          dpi: int = 150) -> plt.Figure:
    """
    Figura de 4 subplots mostrando la curva I-V a cada frecuencia.
    Demuestra el colapso de histéresis (Fig del paper).

    Parámetros
    ----------
    freq_results : dict
        {etiqueta: SimulationResult} del barrido de frecuencias.
    save_path : str | None
        Si se especifica, guarda la figura.
    """
    n = len(freq_results)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = np.array(axes).flatten()

    # Paleta: azul (baja freq) → rojo (alta freq)
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, n))

    for idx, (label, result) in enumerate(freq_results.items()):
        ax = axes[idx]
        freq = result.settings.frequency
        dx = result.state_variable.max() - result.state_variable.min()
        area = hysteresis_area(result.voltage, result.current)

        # Determinar estado del colapso
        if dx > 0.05:
            marker = "Histéresis visible"
            c = colors[idx]
        elif dx > 0.01:
            marker = "Transición"
            c = colors[idx]
        else:
            marker = "COLAPSADO"
            c = "red"

        ax.plot(result.voltage, result.current_mA * 100.0, color=c, lw=2.5)
        ax.fill(result.voltage, result.current_mA * 100.0, alpha=0.1, color=c)
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.4)
        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.4)
        ax.set_xlabel("Voltaje (V)", fontsize=10)
        ax.set_ylabel("Corriente (mA) [Escala ×100]", fontsize=10)
        ax.set_title(f"{label} = {freq:.1f} Hz — {marker}",
                     fontsize=11, fontweight="bold", color=c if dx <= 0.01 else "black")
        ax.grid(True, alpha=0.2, ls=":")

        txt = f"$\\Delta(x)$ = {dx:.5f}\nArea = {area:.2e}"
        ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                fontsize=9, va="top",
                bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    # Ocultar ejes sobrantes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Colapso de Histéresis I-V — Strukov (2008)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  [FIG] Guardado: {save_path}")

    return fig
