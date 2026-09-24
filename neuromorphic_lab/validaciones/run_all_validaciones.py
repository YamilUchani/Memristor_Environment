"""
validaciones/run_all_validaciones.py
=====================================
Runner maestro: ejecuta TODAS las validaciones en orden.

Uso:
    python run_all_validaciones.py            # ejecuta todas
    python run_all_validaciones.py 01 05 10   # solo las especificadas
    python run_all_validaciones.py --list     # lista disponibles
"""

import sys
import argparse
import importlib
import traceback
from pathlib import Path

# Asegurar que neurolab sea importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VALIDACIONES = [
    ("01", "strukov_2008",         "Strukov 2008 vs CSV"),
    ("02", "prezioso_2014",        "Prezioso 2014 vs CSV"),
    ("03", "lif_analitica",        "LIF vs solución analítica"),
    ("04", "convergencia",         "Convergencia temporal"),
    ("05", "d2d_c2c",              "Variabilidad D2D y C2C"),
    ("06", "hfo2_isi",             "Sensor HfO2+LIF"),
    ("07", "r_x_consistencia",     "Consistencia R(x)"),
    ("08", "rs_de_x",              "R_S viene de x"),
    ("09", "dt_extremos",          "Estabilidad dt extremos"),
    ("10", "triangular",           "Señal triangular"),
    ("11", "ruido",                "Ruido controlado"),
    ("12", "estadistica",          "Estadística completa"),
    ("13", "wang_2025",            "Comparación Wang 2025"),
    ("14", "ltp_ltd",              "LTP / LTD"),
    ("15", "stdp",                 "STDP ventana"),
    ("16", "ltp_ltd_ciclo",        "Ciclo LTP-LTD"),
    ("17", "stdp_spikes",          "STDP spikes"),
    ("18", "stdp_temporal",        "STDP temporal"),
    ("19", "iv_histeresis",        "Curva I-V"),
    ("20", "vi_ltp_ltd",           "V-I-R-G evidencia"),
    ("21", "crossbar_1x1",         "Crossbar 1x1"),
    ("22", "prezioso_v2",          "Prezioso v2 asimetría"),
]


def listar():
    """Muestra la lista de validaciones disponibles."""
    print("\n" + "=" * 70)
    print("  VALIDACIONES DISPONIBLES")
    print("=" * 70)
    for num, name, desc in VALIDACIONES:
        print(f"  {num}  | {name:<20} | {desc}")
    print("=" * 70)


def ejecutar(num, name, desc):
    """Ejecuta una validación individual."""
    print(f"\n{'=' * 70}")
    print(f"  [{num}] {desc}")
    print(f"{'=' * 70}")

    module_name = f"{num}_{name}"
    try:
        mod = importlib.import_module(f"validaciones.{module_name}")
        if hasattr(mod, "main"):
            mod.main()
            print(f"  [OK] [{num}] OK")
            return True
        else:
            print(f"  [!] [{num}] Sin funcion main()")
            return False
    except Exception as e:
        print(f"  [X] [{num}] ERROR: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Runner de validaciones")
    parser.add_argument("ids", nargs="*", help="IDs de validaciones a ejecutar")
    parser.add_argument("--list", action="store_true", help="Listar validaciones")
    args = parser.parse_args()

    if args.list:
        listar()
        return

    if args.ids:
        seleccionadas = [v for v in VALIDACIONES if v[0] in args.ids]
    else:
        seleccionadas = VALIDACIONES

    print(f"\n{'=' * 70}")
    print(f"  EJECUTANDO {len(seleccionadas)} VALIDACIONES")
    print(f"{'=' * 70}")

    exitos = 0
    fallos = 0

    for num, name, desc in seleccionadas:
        if ejecutar(num, name, desc):
            exitos += 1
        else:
            fallos += 1

    print(f"\n{'=' * 70}")
    print(f"  RESUMEN: {exitos}/{len(seleccionadas)} OK | {fallos} fallos")
    print(f"{'=' * 70}")

    sys.exit(0 if fallos == 0 else 1)


if __name__ == "__main__":
    main()
