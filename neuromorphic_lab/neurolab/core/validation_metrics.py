"""
neurolab.core.validation_metrics
==================================
Cálculo de métricas cuantitativas de validación entre la simulación y datos de referencia (CSV).

Métricas implementadas:
  - MAE   : Error Absoluto Medio
  - RMSE  : Raíz del Error Cuadrático Medio
  - Error Relativo Máximo (%)
  - Correlación de Pearson (R²)
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


def _pearson_r2(y_ref: np.ndarray, y_sim: np.ndarray) -> float:
    """Calcula el coeficiente de determinación R² de Pearson entre dos vectores."""
    if len(y_ref) < 2 or np.std(y_ref) == 0.0 or np.std(y_sim) == 0.0:
        return float("nan")
    corr = np.corrcoef(y_ref, y_sim)
    return float(corr[0, 1] ** 2) if not np.isnan(corr[0, 1]) else float("nan")


def compute_metrics(
    y_ref: np.ndarray,
    y_sim: np.ndarray,
    name: str,
    units: str,
    threshold: float = 1e-12
) -> QuantityMetrics:
    """
    Calcula MAE, RMSE, error relativo máximo y R² entre y_ref (CSV) e y_sim (simulado).

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

    # Error relativo normalizado por la amplitud pico (evita singularidades en cruces por cero)
    peak = float(np.max(np.abs(y_r)))
    if peak > threshold:
        max_rel_err = float(np.max(diff) / peak * 100.0)
    else:
        max_rel_err = float("nan")

    r2 = _pearson_r2(y_r, y_s)
    return QuantityMetrics(
        name=name, units=units,
        mae=mae, rmse=rmse,
        max_rel_error_pct=max_rel_err,
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
            # Para curvas de histéresis V-I (Prezioso 2014), usar ciclo estabilizado (t >= 50% sim)
            # y emparejamiento por mínima distancia en el plano 2D (V, I) para respetar las ramas SET/RESET
            half = len(v_sim) // 2 if len(v_sim) >= 4 else 0
            v_s = v_sim[half:]
            i_s = i_sim_mA[half:]
            
            # Decimación acelerada para emparejamiento 2D (< 2ms execution time)
            if len(v_s) > 2000:
                step_m = len(v_s) // 2000
                v_s_m = v_s[::step_m]
                i_s_m = i_s[::step_m]
            else:
                v_s_m = v_s
                i_s_m = i_s

            v_span = np.ptp(v_s_m) if np.ptp(v_s_m) > 0 else 1.0
            i_span = np.ptp(i_s_m) if np.ptp(i_s_m) > 0 else 1.0

            i_interp = np.array([
                i_s_m[np.argmin(((v_s_m - vr) / v_span)**2 + ((i_s_m - ir) / i_span)**2)]
                for vr, ir in zip(v_ref, i_ref_mA)
            ])

            mask_set = v_ref >= 0.0
            mask_reset = v_ref < 0.0

            if mask_set.any():
                results.append(compute_metrics(i_ref_mA[mask_set], i_interp[mask_set], "I(V) SET (V>=0)", "mA", threshold=1e-4))
            if mask_reset.any():
                results.append(compute_metrics(i_ref_mA[mask_reset], i_interp[mask_reset], "I(V) RESET (V<0)", "mA", threshold=1e-4))
            results.append(compute_metrics(i_ref_mA, i_interp, "I(V) Completo", "mA", threshold=1e-4))
        else:
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
