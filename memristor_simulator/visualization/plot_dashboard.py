"""
visualization/plot_dashboard.py
================================
Dashboard consolidado con todas las métricas del Taller de Grado I.

Genera la figura principal que integra:
  - Curvas I-V (Fig 2b y 2c)
  - Evolución de x(t) y M(t)
  - Histograma de variabilidad C2C
  - Métricas cuantitativas (R², error, área)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Optional
from ..simulations.iv_characterization import SimulationResult
from ..simulations.stochastic_analysis import StochasticResult
from ..validation.metrics import (hysteresis_area, calculate_r_squared,
                                   state_excursion, hysteresis_collapse_index)


def plot_full_dashboard(result_2b: SimulationResult,
                         result_tri: SimulationResult = None,
                         stoch_result: StochasticResult = None,
                         freq_results: Dict[str, SimulationResult] = None,
                         save_path: str = "dashboard_completo.png",
                         dpi: int = 150,
                         show: bool = False) -> plt.Figure:
    """
    Dashboard consolidado del Gemelo Digital del Memristor.

    Layout (3 filas x 3 columnas):
      [0,0] Curva I-V senoidal (Fig 2b)
      [0,1] Curva I-V triangular (validación metodológica)
      [0,2] Evolución temporal v(t) y x(t)
      [1,0] Relación q(Φ) — monovalencia del memristor
      [1,1] Memristancia M(t) dinámica
      [1,2] Barrido de frecuencias (colapso)
      [2,0] Histograma R_on C2C
      [2,1] Dispersión I_max por ciclo
      [2,2] Tabla de métricas cuantitativas

    Parámetros
    ----------
    result_2b : SimulationResult
        Resultado de la simulación principal (Figura 2b).
    result_tri : SimulationResult | None
        Resultado con onda triangular (puede ser None).
    stoch_result : StochasticResult | None
        Resultado del análisis C2C (puede ser None).
    freq_results : dict | None
        Resultados del barrido de frecuencias (puede ser None).
    save_path : str
        Ruta de guardado del dashboard.
    dpi : int
        Resolución de exportación.
    show : bool
        Si True, llama plt.show().
    """
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35,
                           top=0.93, bottom=0.06, left=0.07, right=0.97)

    DARK_BG = 'white'
    GRID_COLOR = '#d0d0d0'
    TEXT_COLOR = 'black'
    ACCENT = '#1f77b4'
    GREEN = '#2ca02c'
    RED = '#d62728'
    PURPLE = '#9467bd'
    ORANGE = '#ff7f0e'

    def _style_ax(ax, title=""):
        ax.set_facecolor(DARK_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.grid(True, color=GRID_COLOR, alpha=0.5, ls=':')
        if title:
            ax.set_title(title, color=TEXT_COLOR, fontsize=10.5,
                         fontweight='bold', pad=6)

    # ── [0,0] Curva I-V senoidal ─────────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    _style_ax(ax00, "I-V Senoidal — Figura 2b (Strukov 2008)")
    v = result_2b.voltage
    i = result_2b.current_mA
    ax00.plot(v, i, color=ACCENT, lw=2.2)
    ax00.fill(v, i, alpha=0.08, color=ACCENT)
    ax00.axhline(0, color=GRID_COLOR, lw=0.8)
    ax00.axvline(0, color=GRID_COLOR, lw=0.8)
    ax00.set_xlabel("Voltaje (V)", fontsize=9)
    ax00.set_ylabel("Corriente (mA)", fontsize=9)
    area_2b = hysteresis_area(v, result_2b.current)
    ax00.text(0.97, 0.97, f"Area = {area_2b:.2e} V·A",
              transform=ax00.transAxes, fontsize=8.5, va='top', ha='right',
              color=TEXT_COLOR,
              bbox=dict(boxstyle='round', fc=DARK_BG, ec=GRID_COLOR, alpha=0.9))

    # ── [0,1] Curva I-V triangular ────────────────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    if result_tri is not None:
        _style_ax(ax01, "I-V Triangular — Validación Metodológica")
        vt = result_tri.voltage
        it = result_tri.current_mA
        ax01.plot(vt, it, color=GREEN, lw=2.2)
        ax01.fill(vt, it, alpha=0.08, color=GREEN)
        ax01.axhline(0, color=GRID_COLOR, lw=0.8)
        ax01.axvline(0, color=GRID_COLOR, lw=0.8)
        ax01.set_xlabel("Voltaje (V)", fontsize=9)
        ax01.set_ylabel("Corriente (mA)", fontsize=9)
    else:
        _style_ax(ax01, "I-V Triangular (no disponible)")
        ax01.text(0.5, 0.5, "Sin datos", transform=ax01.transAxes,
                  ha='center', va='center', color=TEXT_COLOR, fontsize=11)

    # ── [0,2] Evolución temporal x(t) ────────────────────────────────────
    ax02 = fig.add_subplot(gs[0, 2])
    _style_ax(ax02, "Variable de Estado x(t) = w/D")
    ax02.plot(result_2b.time, result_2b.state_variable, color=RED, lw=2)
    ax02.axhline(0, color=GRID_COLOR, ls='--', lw=1, alpha=0.5)
    ax02.axhline(1, color=GRID_COLOR, ls='--', lw=1, alpha=0.5)
    ax02.set_ylim(-0.05, 1.05)
    ax02.set_xlabel("Tiempo (s)", fontsize=9)
    ax02.set_ylabel("x = w/D", fontsize=9, color=RED)
    ax02r = ax02.twinx()
    ax02r.plot(result_2b.time, result_2b.voltage, color=ACCENT, lw=1.2,
               alpha=0.5, ls='--')
    ax02r.set_ylabel("Voltaje (V)", fontsize=9, color=ACCENT)
    ax02r.tick_params(colors=ACCENT, labelsize=8)

    # ── [1,0] Relación q(Φ) ──────────────────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    _style_ax(ax10, "Carga vs Flujo q(Φ) — Univalencia")
    ax10.plot(result_2b.flux, result_2b.charge, color=PURPLE, lw=2)
    ax10.set_xlabel("Flujo Φ = ∫V dt  (V·s)", fontsize=9)
    ax10.set_ylabel("Carga q = ∫I dt  (C)", fontsize=9)

    # ── [1,1] Memristancia M(t) ──────────────────────────────────────────
    ax11 = fig.add_subplot(gs[1, 1])
    _style_ax(ax11, "Memristancia M(t) Dinámica")
    ax11.plot(result_2b.time, result_2b.resistance / 1000, color=ORANGE, lw=2)
    ax11.set_xlabel("Tiempo (s)", fontsize=9)
    ax11.set_ylabel("M(t) (kΩ)", fontsize=9)
    ax11.axhline(result_2b.params.R_on / 1000, color=GREEN, ls='--',
                 lw=1, alpha=0.7, label=f"R_on={result_2b.params.R_on:.0f}Ω")
    ax11.axhline(result_2b.params.R_off / 1000, color=RED, ls='--',
                 lw=1, alpha=0.7, label=f"R_off={result_2b.params.R_off/1000:.1f}kΩ")
    ax11.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COLOR,
                edgecolor=GRID_COLOR)

    # ── [1,2] Barrido de frecuencias ─────────────────────────────────────
    ax12 = fig.add_subplot(gs[1, 2])
    _style_ax(ax12, "Colapso de Histéresis por Frecuencia")
    if freq_results:
        colors_freq = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(freq_results)))
        for (label, res), c in zip(freq_results.items(), colors_freq):
            ax12.plot(res.voltage, res.current_mA, lw=1.8, color=c, label=label)
        ax12.axhline(0, color=GRID_COLOR, lw=0.8)
        ax12.axvline(0, color=GRID_COLOR, lw=0.8)
        ax12.set_xlabel("Voltaje (V)", fontsize=9)
        ax12.set_ylabel("Corriente (mA)", fontsize=9)
        ax12.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COLOR,
                    edgecolor=GRID_COLOR)
    else:
        ax12.text(0.5, 0.5, "Barrido de frecuencias\nno disponible",
                  transform=ax12.transAxes, ha='center', va='center',
                  color=TEXT_COLOR, fontsize=10)

    # ── [2,0] Histograma R_on C2C ─────────────────────────────────────────
    ax20 = fig.add_subplot(gs[2, 0])
    _style_ax(ax20, "Distribución R_on — Variabilidad C2C")
    if stoch_result and stoch_result.r_on_samples:
        r_on_arr = np.array(stoch_result.r_on_samples)
        ax20.hist(r_on_arr, bins=15, color=ACCENT, alpha=0.75,
                  edgecolor=DARK_BG, density=True)
        ax20.axvline(r_on_arr.mean(), color=RED, ls='--', lw=2,
                     label=f"Media = {r_on_arr.mean():.1f} Ω")
        ax20.set_xlabel("R_on (Ω)", fontsize=9)
        ax20.set_ylabel("Densidad", fontsize=9)
        ax20.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COLOR,
                    edgecolor=GRID_COLOR)
    else:
        ax20.text(0.5, 0.5, "Sin datos C2C", transform=ax20.transAxes,
                  ha='center', va='center', color=TEXT_COLOR, fontsize=11)

    # ── [2,1] Dispersión I_max por ciclo ─────────────────────────────────
    ax21 = fig.add_subplot(gs[2, 1])
    _style_ax(ax21, "Dispersión I_max por Ciclo (C2C)")
    if stoch_result and stoch_result.i_max_samples:
        i_arr = np.array(stoch_result.i_max_samples) * 1e3
        cycles = np.arange(1, len(i_arr) + 1)
        ax21.bar(cycles, i_arr, color=GREEN, alpha=0.65, edgecolor=DARK_BG)
        ax21.axhline(i_arr.mean(), color=RED, ls='--', lw=2,
                     label=f"Media = {i_arr.mean():.2f} mA")
        ax21.fill_between(cycles, i_arr.mean() - i_arr.std(),
                          i_arr.mean() + i_arr.std(),
                          color=RED, alpha=0.12, label="±1σ")
        ax21.set_xlabel("Ciclo #", fontsize=9)
        ax21.set_ylabel("I_max (mA)", fontsize=9)
        ax21.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COLOR,
                    edgecolor=GRID_COLOR)
    else:
        ax21.text(0.5, 0.5, "Sin datos C2C", transform=ax21.transAxes,
                  ha='center', va='center', color=TEXT_COLOR, fontsize=11)

    # ── [2,2] Tabla de métricas ───────────────────────────────────────────
    ax22 = fig.add_subplot(gs[2, 2])
    ax22.set_facecolor(DARK_BG)
    for spine in ax22.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax22.set_title("Métricas Cuantitativas", color=TEXT_COLOR,
                   fontsize=10.5, fontweight='bold', pad=6)
    ax22.axis('off')

    x_min, x_max, delta_x = state_excursion(result_2b.state_variable)
    hci = hysteresis_collapse_index(result_2b.voltage, result_2b.current)

    rows = [
        ["Parámetro", "Valor"],
        ["R_on", f"{result_2b.params.R_on:.0f} Ω"],
        ["R_off", f"{result_2b.params.R_off:.0f} Ω"],
        ["Ratio R_off/R_on", f"{result_2b.params.ratio:.0f}:1"],
        ["D (espesor)", f"{result_2b.params.D*1e9:.0f} nm"],
        ["x_min", f"{x_min:.4f}"],
        ["x_max", f"{x_max:.4f}"],
        ["Δ(x)", f"{delta_x:.4f}"],
        ["Área I-V", f"{area_2b:.3e} V·A"],
        ["HCI", f"{hci:.4f}"],
    ]
    if stoch_result and stoch_result.r_on_samples:
        r2 = stoch_result.compute_r_squared()
        rows.append(["R² (C2C)", f"{r2:.4f}"])
        rows.append(["R_on CV", f"{stoch_result.r_on_cv*100:.1f} %"])

    tbl = ax22.table(cellText=rows[1:], colLabels=rows[0],
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor('#f9f9f9' if row % 2 == 0 else 'white')
        cell.set_text_props(color='black')
        cell.set_edgecolor(GRID_COLOR)
        if row == 0:
            cell.set_facecolor('#eaeaea')
            cell.set_text_props(color='black', fontweight='bold')

    # ── Título principal ─────────────────────────────────────────────────
    fig.suptitle(
        "Gemelo Digital — Memristor Strukov (2008) | Taller de Grado I",
        fontsize=15, fontweight='bold', color=TEXT_COLOR, y=0.98
    )

    fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  [DASHBOARD] Guardado: {save_path}")

    if show:
        plt.show()

    return fig
