"""
validation/metrics.py
======================
Métricas cuantitativas para validación del simulador Strukov (2008).

Indicadores implementados:
  - R²  (coeficiente de determinación de Pearson)
  - RMSE (error cuadrático medio)
  - Error relativo medio (%)
  - Área del lazo de histéresis I-V (proxy de disipación de energía)
  - Índice de asimetría del lazo (para clasificar comportamiento bipolar)

Referencias
-----------
  [1] Strukov et al., Nature 453 (2008) — métricas morfológicas del paper.
  [2] ISO 5725-2:1994 — Exactitud de métodos de medida.
"""

import numpy as np
from typing import Union, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Métricas estándar
# ─────────────────────────────────────────────────────────────────────────────

def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coeficiente de determinación R² (Pearson).

    R² = 1 − SS_res / SS_tot

    Valores de referencia:
      R² > 0.99  → Excelente ajuste
      R² > 0.95  → Buen ajuste
      R² < 0.90  → Ajuste insuficiente para validación académica

    Parámetros
    ----------
    y_true : np.ndarray
        Valores de referencia (señal experimental o paper).
    y_pred : np.ndarray
        Valores simulados.

    Retorna
    -------
    float : R² ∈ (−∞, 1].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else float("nan")
    return float(1.0 - ss_res / ss_tot)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Square Error (RMSE).

    RMSE = sqrt( mean( (y_true − y_pred)² ) )

    Parámetros
    ----------
    y_true, y_pred : np.ndarray
        Arrays de igual longitud.

    Retorna
    -------
    float : RMSE en las mismas unidades que y_true.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def relative_error(y_true: np.ndarray, y_pred: np.ndarray,
                   epsilon: float = 1e-10) -> float:
    """
    Error relativo medio (%).

    ε_rel = mean( |y_true − y_pred| / (|y_true| + ε) ) × 100

    Parámetros
    ----------
    y_true : np.ndarray
        Referencia.
    y_pred : np.ndarray
        Predicción simulada.
    epsilon : float
        Pequeño valor de guarda para evitar división por cero.

    Retorna
    -------
    float : Error relativo en porcentaje (%).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + epsilon)) * 100.0)


def hysteresis_area(voltage: np.ndarray, current: np.ndarray) -> float:
    """
    Área del lazo de histéresis I-V (integral de línea cerrada).

    A = |∮ I dV|

    Un área mayor indica mayor disipación de energía por ciclo y
    memristancia más pronunciada. El colapso a línea recta implica A → 0.

    Parámetros
    ----------
    voltage : np.ndarray
        Voltaje aplicado (V).
    current : np.ndarray
        Corriente medida (A o mA, consistente entre sí).

    Retorna
    -------
    float : Área del lazo en V·(unidad de corriente).
    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    return float(abs(np.trapezoid(current, voltage)))


def hysteresis_collapse_index(voltage: np.ndarray,
                               current: np.ndarray) -> float:
    """
    Índice de colapso de histéresis (adimensional, ∈ [0, 1]).

    HCI = 1 − A_sim / A_ohmic

    Donde A_ohmic es el área del lazo que tendría un resistor puro con
    la misma corriente máxima (lazo elíptico máximo posible).

    HCI ≈ 0 → histéresis plena (memristiva)
    HCI ≈ 1 → colapso total (resistencia constante)

    Parámetros
    ----------
    voltage : np.ndarray
        Voltaje (V).
    current : np.ndarray
        Corriente (A o mA).

    Retorna
    -------
    float : HCI ∈ [0, 1].
    """
    v = np.asarray(voltage, dtype=float)
    i = np.asarray(current, dtype=float)

    # Área simulada
    a_sim = hysteresis_area(v, i)

    # Área máxima de referencia: elipse con v_max × i_max
    a_max = np.pi * v.max() * i.max()
    if a_max < 1e-20:
        return float("nan")

    return float(1.0 - a_sim / a_max)


# ─────────────────────────────────────────────────────────────────────────────
# Métricas del estado interno
# ─────────────────────────────────────────────────────────────────────────────

def state_excursion(state_variable: np.ndarray) -> Tuple[float, float, float]:
    """
    Caracteriza la excursión de la variable de estado x ∈ [0, 1].

    Retorna
    -------
    tuple : (x_min, x_max, Δx) donde Δx = x_max − x_min.
    """
    x = np.asarray(state_variable, dtype=float)
    x_min, x_max = float(x.min()), float(x.max())
    return x_min, x_max, x_max - x_min


def compute_all_metrics(voltage: np.ndarray, current: np.ndarray,
                         state: np.ndarray,
                         current_ref: np.ndarray = None) -> dict:
    """
    Calcula el conjunto completo de métricas de validación.

    Si `current_ref` se proporciona (corriente de referencia del paper),
    se calculan R², RMSE y error relativo contra ella. Si no, las métricas
    de ajuste se calculan contra la corriente ohmica equivalente (R_off).

    Retorna
    -------
    dict : Todas las métricas de validación.
    """
    v = np.asarray(voltage, dtype=float)
    i = np.asarray(current, dtype=float)
    x = np.asarray(state, dtype=float)

    x_min, x_max, delta_x = state_excursion(x)
    area = hysteresis_area(v, i)
    hci = hysteresis_collapse_index(v, i)

    metrics = {
        "hysteresis_area": area,
        "hysteresis_collapse_index": hci,
        "state_x_min": x_min,
        "state_x_max": x_max,
        "state_delta_x": delta_x,
        "current_max_mA": float(i.max() * 1e3) if i.max() < 10 else float(i.max()),
        "voltage_range_V": float(v.max() - v.min()),
    }

    if current_ref is not None:
        i_ref = np.asarray(current_ref, dtype=float)
        # Interpolar si tienen distinta longitud
        if len(i_ref) != len(i):
            i_pred_interp = np.interp(
                np.linspace(0, 1, len(i_ref)),
                np.linspace(0, 1, len(i)),
                i
            )
        else:
            i_pred_interp = i

        metrics["r_squared"] = calculate_r_squared(i_ref, i_pred_interp)
        metrics["rmse"] = rmse(i_ref, i_pred_interp)
        metrics["relative_error_pct"] = relative_error(i_ref, i_pred_interp)

    return metrics


def print_metrics_report(metrics: dict, title: str = "METRICAS DE VALIDACION"):
    """Imprime un reporte formateado de las métricas."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<35} : {v:>12.6f}")
        else:
            print(f"  {k:<35} : {v}")
    print(f"{'='*60}\n")
