"""
neurolab.core.lif_validation
============================
Módulo de validación analítica matemática para la neurona LIF (Sección E).
Calcula la solución analítica diferencial exacta V_teorico(t) y reporta
métricas cuantitativas de exactitud (MAE, RMSE, Error Relativo, R²).
"""

import numpy as np
from typing import Dict, Any
from neurolab.neurons.config import LIFConfig


def compute_lif_analytical_trajectory(
    t: np.ndarray,
    v_sim: np.ndarray,
    v_signal: np.ndarray,
    config: LIFConfig,
    is_voltage_input: bool = True
) -> np.ndarray:
    """
    Calcula la trayectoria analítica exacta V_teorico(t) del potencial de membrana.
    
    Ecuación diferencial:
      tau_eq * dV/dt = V_inf(t) - V
      V_teorico(k) = V_inf(k) + (V_teorico(k-1) - V_inf(k)) * exp(-dt / tau_eq)
    """
    steps = len(t)
    v_analytical = np.zeros(steps)
    
    if steps == 0:
        return v_analytical
        
    dt = t[1] - t[0] if steps > 1 else 1e-4
    
    if is_voltage_input:
        r_eq = config.r_eq
        tau_eq = config.tau
        # V_inf(t) = (R_leak / (R_S + R_leak)) * V_IN(t) + (R_S / (R_S + R_leak)) * V_rest
        r_sum = config.r_series + config.r_leak
        v_inf_array = (config.r_leak / r_sum) * v_signal + (config.r_series / r_sum) * config.v_rest
    else:
        # Modo corriente directa I_in
        tau_eq = config.r_leak * config.c_m
        v_inf_array = config.v_rest + config.r_leak * v_signal

    v_current = config.v_rest
    v_analytical[0] = v_current

    for k in range(1, steps):
        # Si la simulación reseteó la membrana por spike en el paso anterior
        if v_sim[k-1] <= config.v_reset + 1e-6 and v_sim[max(0, k-2)] >= config.v_th - 0.05:
            v_current = config.v_reset

        v_inf = v_inf_array[k]
        v_current = v_inf + (v_current - v_inf) * np.exp(-dt / tau_eq)
        v_analytical[k] = v_current

    return v_analytical


def compute_lif_validation_metrics(
    t: np.ndarray,
    v_sim: np.ndarray,
    v_signal: np.ndarray,
    config: LIFConfig,
    is_voltage_input: bool = True
) -> Dict[str, float]:
    """
    Calcula las métricas cuantitativas formales comparando V_simulado(t) vs V_teórico(t).
    
    Returns:
        Dict con MAE, RMSE, Error_Relativo_Max_Pct, R2 y V_analytical.
    """
    v_analytical = compute_lif_analytical_trajectory(t, v_sim, v_signal, config, is_voltage_input)
    
    # 1. MAE (Error Absoluto Medio)
    mae = float(np.mean(np.abs(v_sim - v_analytical)))
    
    # 2. RMSE (Raíz del Error Cuadrático Medio)
    rmse = float(np.sqrt(np.mean((v_sim - v_analytical) ** 2)))
    
    # 3. Error Relativo Máximo (%)
    denom = np.abs(v_analytical)
    denom = np.where(denom < 1e-6, 1e-6, denom)
    rel_err_max = float(np.max(np.abs(v_sim - v_analytical) / denom) * 100.0)
    
    # 4. R² (Coeficiente de Determinación Teórico)
    ss_res = np.sum((v_sim - v_analytical) ** 2)
    ss_tot = np.sum((v_sim - np.mean(v_sim)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 1.0
    r2 = max(0.0, min(1.0, r2))
    
    return {
        "mae": mae,
        "rmse": rmse,
        "rel_err_max": rel_err_max,
        "r2": r2,
        "v_analytical": v_analytical
    }
