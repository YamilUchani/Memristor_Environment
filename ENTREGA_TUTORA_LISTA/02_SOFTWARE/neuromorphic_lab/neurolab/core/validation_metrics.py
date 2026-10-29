"""
neurolab.core.validation_metrics
==================================
Cálculo de métricas cuantitativas de validación entre la simulación y datos de referencia (CSV).

Métricas implementadas:
  - MAE   : Error Absoluto Medio
  - RMSE  : Raíz del Error Cuadrático Medio
  - Error relativo máximo normalizado por pico (%), usado por la GUI
  - ERP normalizado por rango (%), usado en comparaciones de curvas
  - Coeficiente de determinación R² = 1 - SSE/SST
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class QuantityMetrics:
    """Métricas de error para una magnitud física específica."""
    name: str
    units: str
    mae: float = 0.0
    rmse: float = 0.0
    max_rel_error_pct: float = 0.0
    erp_range_pct: float = 0.0
    r2: float = 0.0
    n_points: int = 0

    def to_html_row(self) -> str:
        """Devuelve una fila HTML <tr> para insertar en una tabla de resultados."""
        r2_color = "#a6e3a1" if self.r2 >= 0.95 else ("#f9e2af" if self.r2 >= 0.80 else "#f38ba8")
        err_color = "#a6e3a1" if self.max_rel_error_pct <= 5.0 else ("#f9e2af" if self.max_rel_error_pct <= 20.0 else "#f38ba8")
        return (
            f"<tr>"
            f"<td style='padding:3px 6px; color:#cdd6f4;'><b>{self.name}</b></td>"
            f"<td style='padding:3px 6px; color:#cdd6f4; text-align:right;'>{self.mae:.4g} {self.units}</td>"
            f"<td style='padding:3px 6px; color:#cdd6f4; text-align:right;'>{self.rmse:.4g} {self.units}</td>"
            f"<td style='padding:3px 6px; color:{err_color}; text-align:right;'>{self.max_rel_error_pct:.2f}%</td>"
            f"<td style='padding:3px 6px; color:{r2_color}; text-align:right;'>{self.r2:.4f}</td>"
            f"</tr>"
        )


def coefficient_of_determination(y_ref: np.ndarray, y_sim: np.ndarray) -> float:
    """Calcula R² = 1 - SSE/SST; devuelve NaN si la referencia es constante."""
    y_ref = np.asarray(y_ref, dtype=float)
    y_sim = np.asarray(y_sim, dtype=float)
    if y_ref.shape != y_sim.shape:
        raise ValueError("y_ref y y_sim deben tener la misma forma")
    if y_ref.size < 2:
        return float("nan")
    total_sum_squares = float(np.sum((y_ref - np.mean(y_ref)) ** 2))
    if total_sum_squares == 0.0:
        return float("nan")
    residual_sum_squares = float(np.sum((y_ref - y_sim) ** 2))
    return 1.0 - residual_sum_squares / total_sum_squares


def compute_metrics(
    y_ref: np.ndarray,
    y_sim: np.ndarray,
    name: str,
    units: str,
    threshold: float = 1e-12
) -> QuantityMetrics:
    """
    Calcula MAE, RMSE, dos errores relativos y R² entre y_ref e y_sim.
    max_rel_error_pct se normaliza por el pico absoluto de referencia;
    erp_range_pct se normaliza por el rango de referencia.

    Args:
        y_ref:     Vector de referencia (datos CSV).
        y_sim:     Vector simulado (interpolado en los mismos instantes).
        name:      Nombre de la magnitud (e.g. "x(t) — w/D").
        units:     Unidades (e.g. "adim.", "mA", "kΩ").
        threshold: Umbral mínimo para calcular error relativo (evita división por cero).

    Returns:
        QuantityMetrics con los resultados.
    """
    mask = np.isfinite(y_ref) & np.isfinite(y_sim)
    y_r = y_ref[mask]
    y_s = y_sim[mask]
    n = len(y_r)

    if n == 0:
        return QuantityMetrics(name=name, units=units, n_points=0)

    diff = np.abs(y_r - y_s)
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # The GUI's relative-error indicator is peak-normalized; ERP uses the full
    # reference range, as defined for the thesis comparisons.
    peak = float(np.max(np.abs(y_r)))
    if peak > threshold:
        max_rel_err = float(np.max(diff) / peak * 100.0)
    else:
        max_rel_err = float("nan")

    reference_range = float(np.ptp(y_r))
    if reference_range > threshold:
        erp_range = float(np.max(diff) / reference_range * 100.0)
    else:
        erp_range = float("nan")

    r2 = coefficient_of_determination(y_r, y_s)
    return QuantityMetrics(
        name=name, units=units,
        mae=mae, rmse=rmse,
        max_rel_error_pct=max_rel_err,
        erp_range_pct=erp_range,
        r2=r2, n_points=n
    )


def compute_all_metrics(
    t_sim: np.ndarray,
    i_sim_mA: np.ndarray,
    x_sim: np.ndarray,
    r_sim_kohm: np.ndarray,
    g_sim_us: np.ndarray,
    val_data: dict,
    v_sim: Optional[np.ndarray] = None,
) -> list[QuantityMetrics]:
    """
    Calcula métricas para todas las magnitudes disponibles en val_data,
    interpolando la simulación en los instantes de tiempo o voltaje del CSV.

    Args:
        t_sim:       Vector de tiempo de la simulación (s).
        i_sim_mA:    Corriente simulada en mA.
        x_sim:       Estado interno simulado x = w/D (adim.).
        r_sim_kohm:  Resistencia simulada en kΩ.
        g_sim_us:    Conductancia simulada en µS.
        val_data:    Diccionario devuelto por ValidationDataLoader.load_all().
        v_sim:       Vector de voltaje simulado (V) opcional.

    Returns:
        Lista de QuantityMetrics ordenada por magnitud.
    """
    results = []

    # ── Estado interno x(t) = w/D ──────────────────────────────────────────
    t_w = val_data.get("t_w")
    wd_ref = val_data.get("wd_val")
    if t_w is not None and wd_ref is not None:
        mask_valid_wd = np.isfinite(wd_ref)
        if mask_valid_wd.any():
            x_interp = np.interp(t_w[mask_valid_wd], t_sim, x_sim)
            results.append(compute_metrics(wd_ref[mask_valid_wd], x_interp, "x(t) — w/D", "adim.", threshold=1e-3))

    # ── Corriente I(t) / I(V) en mA ────────────────────────────────────────
    t_i = val_data.get("t_i")
    i_ref_mA = val_data.get("i_val_mA")
    v_ref = val_data.get("v_interp")
    if t_i is not None and i_ref_mA is not None:
        if v_sim is not None and v_ref is not None and (wd_ref is None or not np.isfinite(wd_ref).any()):
            raise ValueError(
                "No se pueden calcular métricas punto a punto de una curva I-V "
                "sin tiempos o etiquetas de rama de referencia. La selección "
                "de muestras usando simultáneamente voltaje y corriente sesga "
                "MAE, RMSE y R²."
            )
        else:
            if v_ref is not None and (wd_ref is None or not np.isfinite(wd_ref).any()):
                raise ValueError(
                    "Los datos I-V no tienen correspondencia temporal; no se "
                    "pueden tratar como una serie I(t)."
                )
            i_interp = np.interp(t_i, t_sim, i_sim_mA)
            results.append(compute_metrics(i_ref_mA, i_interp, "I(t)", "mA", threshold=1e-4))

    has_time_series_ref = wd_ref is not None and np.isfinite(wd_ref).any()

    # ── Resistencia R(t) en kΩ ──────────────────────────────────────────────
    r_ref = val_data.get("r_val_kohm")
    if has_time_series_ref and t_i is not None and r_ref is not None:
        r_interp = np.interp(t_i, t_sim, r_sim_kohm)
        mask_valid = np.isfinite(r_ref)
        if mask_valid.any():
            results.append(compute_metrics(
                r_ref[mask_valid], r_interp[mask_valid], "R(t)", "kΩ", threshold=0.01
            ))

    # ── Conductancia G(t) en µS ─────────────────────────────────────────────
    g_ref = val_data.get("g_val_us")
    if has_time_series_ref and t_i is not None and g_ref is not None:
        g_interp = np.interp(t_i, t_sim, g_sim_us)
        mask_valid = np.isfinite(g_ref)
        if mask_valid.any():
            results.append(compute_metrics(
                g_ref[mask_valid], g_interp[mask_valid], "G(t)", "µS", threshold=1.0
            ))

    return results
