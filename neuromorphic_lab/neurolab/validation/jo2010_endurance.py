"""
neurolab.validation.jo2010_endurance
====================================
Validación de endurance contra Jo et al. 2010, Fig. 4b.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def get_default_path():
    return Path(__file__).parent.parent.parent / \
           'data_validation' / 'jo2010' / 'jo2010_endurance.csv'


def generate_synthetic_data():
    """Datos sintéticos basados en Fig. 4b."""
    pulse_num = np.arange(0, 81, 5)
    current_na = np.array([
        0.45, 0.52, 0.58, 0.62, 0.65, 0.68, 0.72, 0.70, 0.68, 0.65,
        0.60, 0.55, 0.50, 0.48, 0.45, 0.42, 0.40
    ])
    return pulse_num, current_na


def load_data(csv_path=None):
    if csv_path is None:
        csv_path = get_default_path()
    csv_path = Path(csv_path)

    if csv_path.exists():
        data = pd.read_csv(csv_path)
        return data['pulse_number'].values, \
               data['current_100nA'].values, 'csv'
    else:
        pulse, current = generate_synthetic_data()
        return pulse, current, 'synthetic'


load_jo2010_endurance_data = load_data



def validate_jo2010_endurance(csv_path=None):
    """
    Valida endurance: la ventana LTP/LTD se mantiene tras muchos ciclos.
    """
    pulse_num, current_exp, source = load_data(csv_path)

    # Métricas de endurance
    current_max = current_exp.max()
    current_min = current_exp.min()
    window = current_max - current_min
    window_initial = current_max  # referencia
    retention_window = (current_min / current_max) * 100

    # Detección de degradación (último 20% vs primero 20%)
    n = len(current_exp)
    n_20 = max(1, n // 5)
    first_20 = current_exp[:n_20].mean()
    last_20 = current_exp[-n_20:].mean()
    degradation = (first_20 - last_20) / first_20 * 100

    return {
        'pulse_num': pulse_num,
        'current_exp': current_exp,
        'current_max': current_max,
        'current_min': current_min,
        'window': window,
        'retention_window': retention_window,
        'degradation': degradation,
        'source': source,
    }


def print_report(results):
    import sys
    try:
        sys.stdout.reconfigure(errors='replace')
    except Exception:
        pass
    print("=" * 60)
    print("  VALIDACIÓN ENDURANCE — Jo et al. 2010, Fig. 4b")
    print("=" * 60)
    print(f"  Fuente: {results['source']}")
    print(f"  Corriente máxima: {results['current_max']:.3f} nA")
    print(f"  Corriente mínima: {results['current_min']:.3f} nA")
    print(f"  Ventana:          {results['window']:.3f} nA")
    print(f"  Retención:        {results['retention_window']:.1f} %")
    print(f"  Degradación:      {results['degradation']:.1f} %")
    print("=" * 60)



if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    results = validate_jo2010_endurance()
    print_report(results)

