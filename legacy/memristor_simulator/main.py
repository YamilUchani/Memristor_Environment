"""
memristor_simulator/main.py
============================
Script principal de ejecucion del simulador modular Strukov (2008).

Fases de ejecucion:
  Fase 1 (OBJ-1): Caracterizacion I-V — Figuras 2b y 2c del paper
  Fase 2 (OBJ-2): Analisis estocastico C2C — variabilidad de parametros
  Fase 3 (VAL):   Dashboard consolidado + exportacion para Unity

Uso:
    cd memristor_simulator/
    python main.py                  # Ejecuta todo
    python main.py --phase 1        # Solo Fase 1
    python main.py --phase 2        # Solo Fase 2
    python main.py --no-show        # Sin ventanas interactivas
"""

import sys
import os
import argparse
import numpy as np
import matplotlib as mpl

# Forzar fondo blanco puro sin transparencia en todas las figuras exportadas
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['savefig.facecolor'] = 'white'
mpl.rcParams['savefig.transparent'] = False

# Asegurar que el paquete padre sea encontrable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── Importaciones del paquete ────────────────────────────────────────────────
from memristor_simulator.config.parameters import fig2b_params, fig2c_params
from memristor_simulator.config.simulation_settings import (
    SimulationSettings, WaveformType, settings_frequency_sweep)

from memristor_simulator.simulations.iv_characterization import (
    IVCharacterization, run_figure_2b, run_figure_2c, run_triangular_sweep)
from memristor_simulator.simulations.frequency_analysis import FrequencyAnalysis
from memristor_simulator.simulations.stochastic_analysis import (
    StochasticAnalysis, StochasticResult)

from memristor_simulator.validation.metrics import (
    compute_all_metrics, print_metrics_report)
from memristor_simulator.validation.comparison import PaperComparison

from memristor_simulator.visualization.plot_hysteresis import (
    plot_iv_hysteresis, plot_multi_frequency)
from memristor_simulator.visualization.plot_state import (
    plot_state_evolution, plot_charge_flux)
from memristor_simulator.visualization.plot_dashboard import plot_full_dashboard

from memristor_simulator.utils.data_export import (
    export_to_csv, export_to_json, export_unity_package)


# ── Configuracion global ─────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output_modular')


def banner(msg: str):
    print("\n" + "=" * 65)
    print(f"  {msg}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1: Caracterizacion I-V
# ─────────────────────────────────────────────────────────────────────────────

def phase1_iv_characterization(show: bool = False) -> dict:
    """
    OBJ-1 (100%): Reproduce las Figuras 2b y 2c del paper Strukov (2008)
    y genera los barridos triangular y senoidal de la metodologia del taller.
    """
    banner("FASE 1 — CARACTERIZACION I-V (OBJ-1: 100%)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}

    # ── Figura 2b ─────────────────────────────────────────────────────────
    result_2b = run_figure_2b(verbose=True)
    results["fig2b"] = result_2b

    # Cargar datos del paper (si existen)
    v_ref, i_ref = None, None
    try:
        if os.path.exists("Time-Voltage.csv") and os.path.exists("Time-Current.csv"):
            import pandas as pd
            import numpy as np
            df_v = pd.read_csv("Time-Voltage.csv", header=None)
            df_i = pd.read_csv("Time-Current.csv", header=None)
            t_v, v_ref = df_v.iloc[:, 0].values, df_v.iloc[:, 1].values
            t_i, i_raw = df_i.iloc[:, 0].values, df_i.iloc[:, 1].values
            # Interpolar la corriente al eje de tiempo del voltaje
            i_ref = np.interp(t_v, t_i, i_raw)
    except Exception as e:
        print(f"  [AVISO] No se pudo cargar los CSV del paper: {e}")

    fig = plot_iv_hysteresis(result_2b, color="#4fc3f7",
                              title="Fig 2b — Senoidal, R_off/R_on = 160",
                              v_ref=v_ref, i_ref=i_ref, scale_sim_current=100.0)
    path_2b = os.path.join(OUTPUT_DIR, "fig2b_reproduced.png")
    fig.savefig(path_2b, dpi=150, bbox_inches="tight")
    print(f"  [PNG] {path_2b}")
    if show:
        import matplotlib.pyplot as plt; plt.show()

    # ── Figura 2c ─────────────────────────────────────────────────────────
    result_2c = run_figure_2c(verbose=True)
    results["fig2c"] = result_2c

    fig = plot_iv_hysteresis(result_2c, color="#a5d6a7",
                              title="Fig 2c — sin², R_off/R_on = 380")
    path_2c = os.path.join(OUTPUT_DIR, "fig2c_reproduced.png")
    fig.savefig(path_2c, dpi=150, bbox_inches="tight")
    print(f"  [PNG] {path_2c}")

    # ── Barrido triangular ────────────────────────────────────────────────
    result_tri = run_triangular_sweep(verbose=True)
    results["triangular"] = result_tri

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_iv_hysteresis(result_2b, ax=axes[0], color="#4fc3f7",
                       title="Senoidal — Validacion Metodologica",
                       scale_sim_current=100.0)
    plot_iv_hysteresis(result_tri, ax=axes[1], color="#ffb74d",
                       title="Triangular — Validacion Metodologica",
                       scale_sim_current=100.0)
    plt.suptitle("Validacion Morfologica Strukov (2008) — Pinched Hysteresis Loop",
                 fontweight="bold")
    plt.tight_layout()
    path_valid = os.path.join(OUTPUT_DIR, "metodologia_strukov_valid.png")
    fig.savefig(path_valid, dpi=150, bbox_inches="tight")
    print(f"  [PNG] {path_valid}")

    # ── Evolucion de estado ───────────────────────────────────────────────
    fig_state = plot_state_evolution(result_2b)
    path_state = os.path.join(OUTPUT_DIR, "estado_temporal_fig2b.png")
    fig_state.savefig(path_state, dpi=150, bbox_inches="tight")
    print(f"  [PNG] {path_state}")

    # ── Superposición Voltaje vs Corriente (sin variabilidad) ─────────────
    import matplotlib.pyplot as plt
    _SCALE = 100.0   # igual que en fig2b: i normalizada = i_fisica * 100
    fig_vi, ax_v = plt.subplots(figsize=(10, 5))
    ax_i = ax_v.twinx()

    ax_v.plot(result_2b.time, result_2b.voltage,
              color='#d62728', lw=2.0, label="Voltaje V(t)")
    ax_i.plot(result_2b.time, result_2b.current_mA * _SCALE,
              color='#1f77b4', lw=2.0, label="Corriente I(t)")

    ax_v.set_xlabel("Tiempo (s)", fontsize=12)
    ax_v.set_ylabel("Voltaje (V)", color='#d62728', fontsize=12)
    ax_i.set_ylabel("Corriente (mA)  [norm. i/i₀ × 10]", color='#1f77b4', fontsize=12)
    ax_v.tick_params(axis='y', labelcolor='#d62728')
    ax_i.tick_params(axis='y', labelcolor='#1f77b4')
    ax_v.axhline(0, color='gray', lw=0.5, ls='--')

    lines_v, labels_v = ax_v.get_legend_handles_labels()
    lines_i, labels_i = ax_i.get_legend_handles_labels()
    ax_v.legend(lines_v + lines_i, labels_v + labels_i, loc='upper right', fontsize=10)

    ax_v.set_title("Voltaje y Corriente — Modelo Strukov (sin ruido)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    path_vi = os.path.join(OUTPUT_DIR, "voltaje_corriente_temporal.png")
    fig_vi.savefig(path_vi, dpi=150, bbox_inches="tight")
    print(f"  [PNG] {path_vi}")

    # ── Validacion vs paper ───────────────────────────────────────────────
    comp = PaperComparison()
    report = comp.validate_fig2b(result_2b.as_dict())
    comp.print_report(report)

    # ── Metricas cuantitativas ─────────────────────────────────────────────
    metrics = compute_all_metrics(result_2b.voltage, result_2b.current,
                                   result_2b.state_variable)
    print_metrics_report(metrics, "METRICAS — FIGURA 2b")

    # ── Exportacion CSV ───────────────────────────────────────────────────
    export_to_csv(
        result_2b.as_dict(),
        os.path.join(OUTPUT_DIR, "fig2b_data.csv"),
        {"figure": "2b", "waveform": "sine", "ratio": "160"}
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2: Analisis estocastico y frecuencia
# ─────────────────────────────────────────────────────────────────────────────

def phase2_stochastic_and_frequency(show: bool = False) -> dict:
    """
    OBJ-2 (80%): Variabilidad C2C y colapso de histéresis por frecuencia.
    """
    banner("FASE 2 — ESTOCASTICO + FRECUENCIA (OBJ-2: 80%)")

    results = {}

    # ── Barrido de frecuencias ─────────────────────────────────────────────
    fa = FrequencyAnalysis(fig2b_params(), base_frequency=0.5,
                            amplitude=1.0, n_cycles=3)
    freq_results = fa.run_sweep(verbose=True)
    results["frequency_sweep"] = freq_results

    path_freq = os.path.join(OUTPUT_DIR, "frequency_collapse_demonstration.png")
    plot_multi_frequency(freq_results, save_path=path_freq, dpi=150)

    # ── Analisis C2C ─────────────────────────────────────────────────────
    from memristor_simulator.models.stochastic_model import C2CConfig
    c2c_cfg = C2CConfig(enabled=True, r_on_cv=0.15, r_off_cv=0.10, seed=42)
    sa = StochasticAnalysis(fig2b_params(), c2c_config=c2c_cfg,
                            n_cycles=15, frequency=0.5, amplitude=1.0,
                            signal_cycles=3)  # 3 periodos = 6 s por ciclo C2C
    stoch_result = sa.run(verbose=True)
    results["stochastic"] = stoch_result

    # Figura C2C
    import matplotlib.pyplot as plt
    
    # 1. Gráfica original de R_on (Estadística)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    r_on_arr = np.array(stoch_result.r_on_samples)
    axes[0].bar(range(len(r_on_arr)), r_on_arr, color='#1f77b4', alpha=0.7)
    axes[0].axhline(r_on_arr.mean(), color='#d62728', ls='--',
                    label=f"Media: {r_on_arr.mean():.1f} Ohm")
    axes[0].fill_between(range(len(r_on_arr)),
                         r_on_arr.mean() - r_on_arr.std(),
                         r_on_arr.mean() + r_on_arr.std(),
                         color='#d62728', alpha=0.1, label="+-1sigma")
    axes[0].set_title("Variabilidad C2C — R_on por Ciclo", fontweight="bold")
    axes[0].set_xlabel("Ciclo #")
    axes[0].set_ylabel("R_on (Ohm)")
    axes[0].legend()

    axes[1].hist(r_on_arr, bins=12, color='#1f77b4', alpha=0.75,
                 edgecolor='white', density=True)
    axes[1].axvline(r_on_arr.mean(), color='#d62728', ls='--', lw=2)
    axes[1].set_title(f"Distribucion R_on — CV = {r_on_arr.std()/r_on_arr.mean()*100:.1f}%",
                      fontweight="bold")
    axes[1].set_xlabel("R_on (Ohm)")
    axes[1].set_ylabel("Densidad de probabilidad")

    plt.suptitle("Caracterizacion Estocastica Ciclo-a-Ciclo (C2C)",
                 fontweight="bold")
    plt.tight_layout()
    path_c2c = os.path.join(OUTPUT_DIR, "caracterizacion_estocastica.png")
    fig.savefig(path_c2c, dpi=150, bbox_inches="tight")
    print(f"  [PNG] {path_c2c}")

    # 2. NUEVA GRÁFICA: Visualización Física del Ruido (3 Paneles)
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 0: Voltaje y Corriente en el Tiempo
    axes2[0].set_title("Ondas Temporales (V, I)", fontweight="bold")
    axes2[0].set_xlabel("Tiempo (s)")
    axes2[0].set_ylabel("Corriente (mA)", color='#1f77b4')
    ax0_twin = axes2[0].twinx()
    ax0_twin.set_ylabel("Voltaje (V)", color='gray')
    axes2[0].axhline(0, color='gray', lw=0.5, ls='--')
    
    # Panel 1: w/D en el tiempo
    axes2[1].set_title("Evolución del Estado Interno (w/D)", fontweight="bold")
    axes2[1].set_xlabel("Tiempo (s)")
    axes2[1].set_ylabel("x = w/D (normalizado)", color='#d62728')
    ax1_twin = axes2[1].twinx()
    ax1_twin.set_ylabel("Voltaje (V)", color='gray')
    axes2[1].axhline(1, color='gray', lw=0.5, ls='--')
    axes2[1].axhline(0, color='gray', lw=0.5, ls='--')

    # Panel 2: I-V Hysteresis
    axes2[2].set_title("Lazo de Histéresis (I-V)", fontweight="bold")
    axes2[2].set_xlabel("Voltaje (V)")
    axes2[2].set_ylabel("Corriente (mA)")
    axes2[2].axhline(0, color='gray', lw=0.5, ls='--')
    axes2[2].axvline(0, color='gray', lw=0.5, ls='--')

    # Solo graficamos el voltaje una vez en los twin axes (es el mismo para todos los ciclos)
    t_ref = stoch_result.cycles[0].time
    v_ref = stoch_result.cycles[0].voltage
    ax0_twin.plot(t_ref, v_ref, color='gray', alpha=0.5, lw=1.5, ls='--')
    ax1_twin.plot(t_ref, v_ref, color='gray', alpha=0.5, lw=1.5, ls='--')

    # Superponer todos los ciclos
    for cycle_res in stoch_result.cycles:
        # Panel 0: Corriente vs Tiempo
        axes2[0].plot(cycle_res.time, cycle_res.current_mA, color='#1f77b4', alpha=0.2, lw=1.5)
        # Panel 1: Estado vs Tiempo
        axes2[1].plot(cycle_res.time, cycle_res.state_variable, color='#d62728', alpha=0.2, lw=1.5)
        # Panel 2: I-V Hysteresis
        axes2[2].plot(cycle_res.voltage, cycle_res.current_mA, color='#2ca02c', alpha=0.15, lw=1.5)

    plt.suptitle("Impacto de Variabilidad Estocástica (Ruido Térmico)", fontweight="bold", fontsize=15)
    plt.tight_layout()
    path_noise = os.path.join(OUTPUT_DIR, "ruido_fisico_c2c.png")
    fig2.savefig(path_noise, dpi=150, bbox_inches="tight")
    print(f"  [PNG] {path_noise}")

    r2 = stoch_result.compute_r_squared()
    print(f"\n  R^2 (ciclos C2C)      : {r2:.4f}")
    print(f"  R_on media            : {stoch_result.r_on_mean:.2f} Ohm")
    print(f"  R_on desv. estandar   : {stoch_result.r_on_std:.2f} Ohm")
    print(f"  I_max media           : {stoch_result.i_max_mean_mA:.3f} mA")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# FASE 3: Dashboard y exportacion Unity
# ─────────────────────────────────────────────────────────────────────────────

def phase3_dashboard_and_export(phase1_results: dict,
                                 phase2_results: dict,
                                 show: bool = False):
    """Genera el dashboard consolidado y el paquete de exportacion Unity."""
    banner("FASE 3 — DASHBOARD + EXPORTACION UNITY")

    result_2b = phase1_results.get("fig2b")
    result_tri = phase1_results.get("triangular")
    stoch = phase2_results.get("stochastic")
    freq = phase2_results.get("frequency_sweep")

    path_dash = os.path.join(OUTPUT_DIR, "dashboard_completo.png")
    plot_full_dashboard(
        result_2b=result_2b,
        result_tri=result_tri,
        stoch_result=stoch,
        freq_results=freq,
        save_path=path_dash,
        dpi=150,
        show=show,
    )

    # Exportar paquete Unity
    export_unity_package(
        results=result_2b.as_dict(),
        output_dir=OUTPUT_DIR,
        scenario_name="fig2b_unity",
        params=result_2b.params,
    )

    export_to_json(
        result_2b.as_dict(),
        os.path.join(OUTPUT_DIR, "fig2b_complete.json"),
        params=result_2b.params,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Simulador Strukov (2008) — Taller de Grado I")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3],
                        help="Ejecutar solo una fase (1, 2 o 3). Default: todas.")
    parser.add_argument("--no-show", action="store_true",
                        help="No mostrar ventanas de matplotlib.")
    args = parser.parse_args()
    show = not args.no_show

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    banner("SIMULADOR STRUKOV (2008) — GEMELO DIGITAL NEUROMORFICO")
    print(f"  Autor  : Yamil Ronald Uchani Guachalla")
    print(f"  Salida : {os.path.abspath(OUTPUT_DIR)}")

    p1, p2 = {}, {}

    if args.phase in (None, 1):
        p1 = phase1_iv_characterization(show=show)

    if args.phase in (None, 2):
        p2 = phase2_stochastic_and_frequency(show=show)

    if args.phase in (None, 3) and p1 and p2:
        phase3_dashboard_and_export(p1, p2, show=show)
    elif args.phase == 3:
        print("  [WARN] Fase 3 requiere datos de fases 1 y 2. "
              "Ejecute sin --phase primero.")

    banner("SIMULACION COMPLETADA")
    print(f"  Archivos en: {os.path.abspath(OUTPUT_DIR)}/")
    files = os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else []
    for f in sorted(files):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"    {f:<45} {size/1024:>7.1f} KB")


if __name__ == "__main__":
    main()
