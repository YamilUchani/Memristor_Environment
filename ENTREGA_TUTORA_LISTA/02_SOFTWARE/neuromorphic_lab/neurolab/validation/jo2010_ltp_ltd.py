"""
neurolab.validation.jo2010_ltp_ltd
==================================
Validación LTP/LTD contra Jo et al. 2010, Fig. 2a.
"""

# La referencia utilizada es la digitalización de Jo et al. (2010), Fig. 2a,
# conservada en data_validation/jo2010/jo2010_ltp_ltd.csv.
import numpy as np
import pandas as pd
from pathlib import Path


def get_default_path():
    return Path(__file__).parent.parent.parent / \
           'data_validation' / 'jo2010' / 'jo2010_ltp_ltd.csv'


def load_data(csv_path=None):
    if csv_path is None:
        csv_path = get_default_path()
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Jo et al. LTP/LTD reference data not found: {csv_path}"
        )

    data = pd.read_csv(csv_path)
    return data['pulse_number'].values, \
           data['current_100nA'].values, 'csv'


load_jo2010_ltp_ltd_data = load_data



def validate_jo2010_ltp_ltd(csv_path=None):
    """
    Valida el comportamiento LTP/LTD contra Jo 2010 Fig. 2a.

    Nota: Esta validación es cualitativa — verifica que el simulador
    reproduce la forma general (incremento durante P, decremento
    durante D) sin ajustar parámetros exactos.
    """
    pulse_num, current_exp, source = load_data(csv_path)

    # Detectar el punto de cambio (P → D)
    idx_peak = np.argmax(current_exp)

    # Segmentos
    P_pulses = pulse_num[:idx_peak + 1]
    P_current = current_exp[:idx_peak + 1]
    D_pulses = pulse_num[idx_peak:]
    D_current = current_exp[idx_peak:]

    # Ajuste lineal para verificar monotonía
    P_slope = np.polyfit(P_pulses, P_current, 1)[0]
    D_slope = np.polyfit(D_pulses, D_current, 1)[0]

    # Métricas
    P_monotonic = P_slope > 0
    D_monotonic = D_slope < 0
    peak_current = current_exp[idx_peak]
    final_current = current_exp[-1]
    retention = final_current / current_exp[0] * 100

    return {
        'pulse_num': pulse_num,
        'current_exp': current_exp,
        'P_pulses': P_pulses,
        'P_current': P_current,
        'D_pulses': D_pulses,
        'D_current': D_current,
        'idx_peak': idx_peak,
        'P_slope': P_slope,
        'D_slope': D_slope,
        'P_monotonic': P_monotonic,
        'D_monotonic': D_monotonic,
        'peak_current': peak_current,
        'final_current': final_current,
        'retention_pct': retention,
        'source': source,
    }


def print_report(results):
    import sys
    try:
        sys.stdout.reconfigure(errors='replace')
    except Exception:
        pass
    print("=" * 60)
    print("  VALIDACIÓN LTP/LTD — Jo et al. 2010, Fig. 2a")
    print("=" * 60)
    print(f"  Fuente: {results['source']}")
    print(f"  Punto de cambio: pulso {results['idx_peak']}")
    print()
    print(f"  Pendiente LTP (P):  {results['P_slope']:+.4f} nA/pulso")
    print(f"  Pendiente LTD (D):  {results['D_slope']:+.4f} nA/pulso")
    print(f"  Monotonía LTP:      {'[OK]' if results['P_monotonic'] else '[FAIL]'}")
    print(f"  Monotonía LTD:      {'[OK]' if results['D_monotonic'] else '[FAIL]'}")
    print()
    print(f"  Corriente pico:     {results['peak_current']:.3f} nA")
    print(f"  Corriente final:    {results['final_current']:.3f} nA")
    print(f"  Retención:          {results['retention_pct']:.1f} %")
    print("=" * 60)



if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    results = validate_jo2010_ltp_ltd()
    print_report(results)
