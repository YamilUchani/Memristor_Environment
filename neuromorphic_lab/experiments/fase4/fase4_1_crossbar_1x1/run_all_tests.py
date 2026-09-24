"""
run_all_tests.py
================
Ejecuta la batería completa de 20 pruebas del Crossbar 1×1.
"""

import os
import time
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from test_01_init import test_01_initialization
from test_02_ideal import test_02_ideal_operation
from test_03_v_sweep import test_03_voltage_sweep
from test_04_g_sweep import test_04_conductance_sweep
from test_05_non_destructive import test_05_non_destructive
from test_06_retention import test_06_retention
from test_07_ltp import test_07_ltp
from test_08_ltd import test_08_ltd
from test_09_cycle import test_09_cycle
from test_10_lif import test_10_lif_integration
from test_11_stdp import test_11_stdp
from test_12_matrix_api import test_12_matrix_api
from test_13_uniform import test_13_uniform
from test_14_reset import test_14_reset
from test_15_vs_resistor import test_15_vs_resistor
from test_16_read_window import test_16_read_window
from test_17_write_window import test_17_write_window
from test_18_closed_loop_prog import test_18_closed_loop_prog
from test_19_d2d_variability import test_19_d2d_variability
from test_20_endurance import test_20_endurance


def main():
    tests = [
        ('01', 'Inicialización',           test_01_initialization),
        ('02', 'Operación I = G·V',        test_02_ideal_operation),
        ('03', 'Barrido de voltaje',       test_03_voltage_sweep),
        ('04', 'Barrido de conductancia',  test_04_conductance_sweep),
        ('05', 'Lectura no destructiva',   test_05_non_destructive),
        ('06', 'Persistencia (retención)', test_06_retention),
        ('07', 'Programación LTP',         test_07_ltp),
        ('08', 'Programación LTD',         test_08_ltd),
        ('09', 'Ciclo LTP → LTD → LTP',    test_09_cycle),
        ('10', 'Integración con LIF',      test_10_lif_integration),
        ('11', 'Integración con STDP',     test_11_stdp),
        ('12', 'Interfaz matricial',       test_12_matrix_api),
        ('13', 'Uniformidad',              test_13_uniform),
        ('14', 'Reset',                    test_14_reset),
        ('15', 'vs Resistencia pura',      test_15_vs_resistor),
        ('16', 'Ventana de Lectura',       test_16_read_window),
        ('17', 'Ventana de Escritura',     test_17_write_window),
        ('18', 'Programación Analógica',   test_18_closed_loop_prog),
        ('19', 'Variabilidad D2D',         test_19_d2d_variability),
        ('20', 'Endurance (1000 ciclos)',  test_20_endurance),
    ]

    print("\n" + "=" * 70)
    print("  BATERÍA COMPLETA DE 20 PRUEBAS — CROSSBAR 1×1")
    print("=" * 70 + "\n")

    t0 = time.time()
    passed = 0
    failed = 0
    results = []

    for num, name, test_fn in tests:
        try:
            t_start = time.time()
            test_fn()
            t_elapsed = time.time() - t_start
            results.append((num, name, 'PASS', t_elapsed))
            passed += 1
        except AssertionError as e:
            results.append((num, name, f'FAIL: {e}', 0))
            failed += 1
        except Exception as e:
            results.append((num, name, f'ERROR: {e}', 0))
            failed += 1

    t_total = time.time() - t0

    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    # --- Reporte final ---
    print("\n" + "=" * 70)
    print("  REPORTE FINAL — CROSSBAR 1×1 (20/20)")
    print("=" * 70)
    print(f"\n{'#':<4} {'Prueba':<35} {'Estado':<10} {'Tiempo':<10}")
    print("-" * 70)
    for num, name, status, t_el in results:
        t_str = f'{t_el*1e3:.1f} ms' if t_el else '—'
        status_icon = '[OK]' if status == 'PASS' else '[FAIL]'
        print(f"{num:<4} {name:<35} {status_icon:<6} {status:<8} {t_str:<10}")

    print("-" * 70)
    print(f"\nTotal: {passed}/{len(tests)} pruebas pasadas")
    print(f"Fallidas: {failed}")
    print(f"Tiempo total: {t_total:.2f} s")
    print(f"\nEstado final: {'[OK] TODAS PASARON (100% COMPLETADO)' if failed == 0 else '[FAIL] HAY FALLOS'}")
    print("=" * 70)


if __name__ == '__main__':
    main()
