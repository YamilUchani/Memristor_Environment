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

import matplotlib as mpl
mpl.rcParams['figure.facecolor']   = 'white'
mpl.rcParams['axes.facecolor']     = 'white'
mpl.rcParams['savefig.facecolor']  = 'white'
mpl.rcParams['savefig.transparent'] = False

# Asegurar que el paquete sea encontrado desde cualquier directorio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memristor_simulator.models.lif_neuron import (
    default_lif_params, fast_lif_params)
from memristor_simulator.simulations.lif_validation import LIFValidation
from memristor_simulator.visualization.plot_lif_validation import (
    plot_lif_validation_dashboard)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output_modular')


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
    validator = LIFValidation(params=params, dt=0.1e-3)
    results   = validator.run(verbose=True)

    # ── Resumen de validacion ──────────────────────────────────────────────
    banner("RESUMEN DE VALIDACION")

    checks = [
        ("(A) V aumenta con corriente",
         results.ramp.voltage_mV.max() > params.E_L * 1e3 + 1.0),
        ("(B) Umbral existe (subthreshold no dispara)",
         results.threshold.spike_times[0] * 1e3 > 200.0
         if results.threshold.n_spikes > 0 else False),
        ("(C) Spike generado al superar umbral",
         results.single.n_spikes >= 1),
        ("(D) Reset correcto (V_reset tras spike)",
         abs(results.single.voltage_mV[
             int(results.single.spike_times[0] / 0.1e-3) + 1
         ] - params.V_reset * 1e3) < 1.0
         if results.single.n_spikes >= 1 else False),
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
