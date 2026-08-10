"""
validate_lif.py
===============
Script de ejecucion del PASO 4: Validacion aislada de la neurona LIF.

Demuestra que la neurona LIF funciona correctamente ANTES de conectarla
al memristor Strukov, verificando cada uno de sus mecanismos fundamentales:

  (A) El potencial aumenta al recibir corriente.
  (B) Existe un umbral de disparo definido.
  (C) Se produce un spike cuando V supera V_th.
  (D) El potencial se reinicia correctamente tras el disparo.

Uso:
    python validate_lif.py              # Genera figura + muestra ventana
    python validate_lif.py --no-show   # Solo genera el PNG
    python validate_lif.py --fast      # Parametros de neurona rapida

Salida:
    output_modular/lif_validation_dashboard.png
"""

import sys
import os
import argparse
import numpy as np
from memristor_simulator.validation.lif_experimental_data import VC_T, VC_V, VOUT_T, VOUT_V

import matplotlib as mpl
mpl.rcParams['figure.facecolor']   = 'white'
mpl.rcParams['axes.facecolor']     = 'white'
mpl.rcParams['savefig.facecolor']  = 'white'
mpl.rcParams['savefig.transparent'] = False

# Asegurar que el paquete sea encontrado desde cualquier directorio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from memristor_simulator.models.lif_neuron import (
    default_lif_params, fast_lif_params)
from memristor_simulator.simulations.lif_validation import LIFValidation
from memristor_simulator.visualization.plot_lif_validation import (
    plot_lif_validation_dashboard)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))


def banner(msg: str):
    print("\n" + "=" * 65)
    print(f"  {msg}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="PASO 4 — Validacion aislada de la neurona LIF")
    parser.add_argument("--no-show", action="store_true",
                        help="No mostrar ventana interactiva de matplotlib.")
    parser.add_argument("--fast", action="store_true",
                        help="Usar parametros de neurona rapida (tau=10ms).")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    banner("PASO 4 — VALIDACION DE NEURONA LIF AISLADA")
    print("  Proyecto: Gemelo Digital Neurormorfico — Taller de Grado I")
    print("  Autor:    Yamil Ronald Uchani Guachalla")

    # ── Seleccion de parametros ────────────────────────────────────────────
    params = fast_lif_params() if args.fast else default_lif_params()
    label  = "rapida" if args.fast else "estandar"
    print(f"\n  Neurona: {label}")

    # ── Ejecutar simulaciones ──────────────────────────────────────────────
    validator = LIFValidation(params=params, dt=5e-6)  # dt = 5 us para alta estabilidad y precisión física
    results   = validator.run(verbose=True)

    # Calcular RMSE para la comparación experimental
    Vc_interp = np.interp(VC_T, results.comparison.time, results.comparison.voltage)
    Vout_interp = np.interp(VOUT_T, results.comparison.time, results.comparison.vout)
    rmse_vc = np.sqrt(np.mean((Vc_interp - VC_V)**2))
    rmse_vout = np.sqrt(np.mean((Vout_interp - VOUT_V)**2))

    # ── Resumen de validacion ──────────────────────────────────────────────
    banner("RESUMEN DE VALIDACION")

    checks = [
        ("(A) Vc aumenta con Vin",
         results.ramp.voltage.max() > 0.8),
        ("(B) Umbral existe (subumbral no dispara)",
         results.threshold.spike_times[0] > 0.20
         if results.threshold.n_spikes > 0 else False),
        ("(C) Spike generado al superar umbral",
         results.single.n_spikes >= 1),
        ("(D) Descarga física (auto-reset post-spike)",
         results.single.voltage.min() < (params.V_hold + 0.1)
         if results.single.n_spikes >= 1 else False),
        (f"(E) Ajuste de Vc (RMSE = {rmse_vc*1e3:.1f} mV < 100 mV)",
         rmse_vc < 0.100),
        (f"(F) Ajuste de Vout (RMSE = {rmse_vout*1e3:.1f} mV < 100 mV)",
         rmse_vout < 0.100),
    ]

    all_pass = True
    for desc, ok in checks:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {desc}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("  [OK] Todos los mecanismos verificados.")
        print("       La neurona LIF esta lista para conectarse al memristor.")
    else:
        print("  [!!] Algunos checks fallaron. Revisar parametros.")

    # ── Generacion de figura ───────────────────────────────────────────────
    banner("GENERANDO FIGURA")
    save_path = os.path.join(OUTPUT_DIR, "lif_validation_dashboard.png")
    fig = plot_lif_validation_dashboard(results, save_path=save_path, dpi=150)

    print(f"\n  Figura guardada en:")
    print(f"    {os.path.abspath(save_path)}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()

    banner("VALIDACION COMPLETADA")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
