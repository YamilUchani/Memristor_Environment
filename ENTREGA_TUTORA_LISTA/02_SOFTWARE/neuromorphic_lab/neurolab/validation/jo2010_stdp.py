"""
neurolab.validation.jo2010_stdp
===============================
Validación de STDP contra Jo et al. 2010 (Nano Letters).

Paper: Jo, Chang, Ebong, Bhadviya, Mazumder, Lu
       "Nanoscale Memristor Device as Synapse in Neuromorphic Systems"
       Nano Letters 10(4), 1297-1301, 2010.
       DOI: 10.1021/nl904092h

Referencia: Fig. 3a (ventana STDP experimental)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from neurolab.core.validation_metrics import coefficient_of_determination


def get_csv_path():
    """Ruta por defecto del CSV de Jo 2010."""
    root = Path(__file__).parent.parent.parent
    return root / 'data_validation' / 'jo2010' / 'jo2010_stdp.csv'


def load_jo2010_stdp_data(csv_path=None):
    """
    Carga datos experimentales de Jo 2010, Fig. 3a.

    Parameters
    ----------
    csv_path : str or Path, optional
        Ruta al CSV. Si None, usa la default.

    Returns
    -------
    dt_ms : ndarray
        Δt en ms (negativo = LTD, positivo = LTP)
    dw_pct : ndarray
        ΔW en % (normalizado al peso máximo)
    source : str
        'csv' para los puntos de referencia cargados del archivo
    """
    if csv_path is None:
        csv_path = get_csv_path()

    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Jo et al. STDP reference data not found: {csv_path}")

    data = pd.read_csv(csv_path)
    dt_ms = np.asarray(data['delta_t_ms'].values, dtype=float)
    dw_pct = np.asarray(data['delta_w_percent'].values, dtype=float)

    # Normalize the sign if the CSV uses t_pre - t_post instead of t_post - t_pre.
    pos_mask = dt_ms > 0
    if pos_mask.any() and np.mean(dw_pct[pos_mask]) < 0:
        dt_ms = -dt_ms

    sort_idx = np.argsort(dt_ms)
    dt_ms = dt_ms[sort_idx]
    dw_pct = dw_pct[sort_idx]
    return dt_ms, dw_pct, 'csv'


# Alias para retrocompatibilidad
load_data = load_jo2010_stdp_data


def validate_jo2010_stdp(dt_values_or_rule, dw_sim=None, csv_path=None):
    """
    Compara simulación STDP contra datos experimentales de Jo 2010.

    Parameters
    ----------
    dt_values_or_rule : ndarray o STDPRule
        Δt en ms (donde se evalúa la simulación) O una instancia de STDPRule.
    dw_sim : ndarray, optional
        ΔW simulado (mismo tamaño que dt_values). Si el primer param es STDPRule,
        este parámetro puede ser el csv_path.
    csv_path : str or Path, optional

    Returns
    -------
    dict con métricas y vectores de datos.
    """
    if hasattr(dt_values_or_rule, 'delta_w'):
        # Es un objeto STDPRule
        stdp_rule = dt_values_or_rule
        if isinstance(dw_sim, (str, Path)):
            csv_path = dw_sim
        dt_exp_ms, dw_exp_pct, source = load_jo2010_stdp_data(csv_path)
        dt_s = dt_exp_ms * 1e-3
        dw_sim_at_exp = np.array([stdp_rule.delta_w(dt) for dt in dt_s]) * 100.0

        dt_values = np.linspace(-80.0, 80.0, 161)
        dw_sim_curve = np.array([stdp_rule.delta_w(dt * 1e-3) for dt in dt_values]) * 100.0
    else:
        dt_values = np.asarray(dt_values_or_rule, dtype=float)
        dw_sim_curve = np.asarray(dw_sim, dtype=float)
        dt_exp_ms, dw_exp_pct, source = load_jo2010_stdp_data(csv_path)
        dw_sim_at_exp = np.interp(dt_exp_ms, dt_values, dw_sim_curve)

    # Métricas (en espacio normalizado [0,1])
    dw_exp = dw_exp_pct / 100.0
    dw_sim_norm = dw_sim_at_exp / 100.0

    mae = float(np.mean(np.abs(dw_sim_norm - dw_exp)))
    rmse = float(np.sqrt(np.mean((dw_sim_norm - dw_exp) ** 2)))

    r2 = coefficient_of_determination(dw_exp, dw_sim_norm)

    # Error relativo máximo (puntos con |dw| > 1%)
    mask = np.abs(dw_exp_pct) > 1.0
    if mask.any():
        err_max = float(np.max(
            np.abs(dw_sim_at_exp[mask] - dw_exp_pct[mask]) /
            np.abs(dw_exp_pct[mask])
        ) * 100)
    else:
        err_max = 0.0

    return {
        'dt_exp_ms': dt_exp_ms,
        'dw_exp_pct': dw_exp_pct,
        'dt_sim_ms': dt_values,
        'dw_sim_pct': dw_sim_curve,
        'dw_sim_at_exp': dw_sim_at_exp,
        'dt_ms': dt_exp_ms,
        'dw_sim': dw_sim_at_exp,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'err_max': err_max,
        'err_rel_max': err_max,
        'source': source,
        'n_points': len(dt_exp_ms),
    }
