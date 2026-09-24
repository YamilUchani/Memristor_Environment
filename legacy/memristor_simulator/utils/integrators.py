"""
utils/integrators.py
=====================
Métodos de integración numérica y generadores de formas de onda.

Incluye:
  - Euler explícito (paso único)
  - Runge-Kutta de 4to orden (RK4) — mayor precisión para dt grande
  - Generadores vectoriales de señales de voltaje

Los integradores operan sobre las funciones puras de strukov_model.py,
por lo que son independientes del estado y testeables de forma aislada.
"""

import numpy as np
from enum import Enum, auto
from typing import Callable

from memristor_simulator.config.simulation_settings import WaveformType


# ─────────────────────────────────────────────────────────────────────────────
# Generadores de formas de onda
# ─────────────────────────────────────────────────────────────────────────────

def generate_waveform(t: np.ndarray, waveform: WaveformType,
                      amplitude: float = 1.0, frequency: float = 0.5) -> np.ndarray:
    """
    Genera un vector de voltaje para el array de tiempos `t`.

    Parámetros
    ----------
    t : np.ndarray
        Array de instantes de tiempo (s).
    waveform : WaveformType
        Tipo de forma de onda.
    amplitude : float
        Amplitud v₀ (V).
    frequency : float
        Frecuencia f₀ (Hz).

    Retorna
    -------
    np.ndarray : Voltaje v(t) (V).
    """
    w = 2.0 * np.pi * frequency * t

    if waveform == WaveformType.SINE:
        # v₀·sin(ω₀t) — Figura 2b del paper
        return amplitude * np.sin(w)

    elif waveform == WaveformType.SINE_SQUARED:
        # 6·v₀·sin²(ω₀t) → aquí amplitude ya incluye el factor
        # (el paper usa 6v₀ pero la amplitud real es ~4V para v₀=1V)
        return amplitude * (np.sin(w) ** 2)

    elif waveform == WaveformType.TRIANGLE:
        # Onda triangular usando la serie de Fourier (aprox. 20 términos)
        v = np.zeros_like(t)
        for k in range(20):
            n = 2 * k + 1
            v += ((-1) ** k / n ** 2) * np.sin(n * w)
        return amplitude * (8.0 / np.pi ** 2) * v

    elif waveform == WaveformType.SQUARE:
        return amplitude * np.sign(np.sin(w))

    else:
        raise ValueError(f"Forma de onda no soportada: {waveform}")


def generate_waveform_scalar(t: float, waveform: WaveformType,
                              amplitude: float = 1.0,
                              frequency: float = 0.5) -> float:
    """Versión escalar de generate_waveform para bucles de integración."""
    w = 2.0 * np.pi * frequency * t

    if waveform == WaveformType.SINE:
        return amplitude * np.sin(w)
    elif waveform == WaveformType.SINE_SQUARED:
        return amplitude * (np.sin(w) ** 2)
    elif waveform == WaveformType.TRIANGLE:
        v = sum(((-1) ** k / (2 * k + 1) ** 2) * np.sin((2 * k + 1) * w)
                for k in range(20))
        return amplitude * (8.0 / np.pi ** 2) * v
    elif waveform == WaveformType.SQUARE:
        return amplitude * np.sign(np.sin(w))
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Integradores numéricos
# ─────────────────────────────────────────────────────────────────────────────

def euler_step(x: float, dxdt_fn: Callable[[float], float],
               voltage: float, dt: float) -> float:
    """
    Método de Euler explícito para dx/dt = f(x, v, t).

    x_{n+1} = x_n + dt · f(x_n, v_n)

    Parámetros
    ----------
    x : float
        Estado actual.
    dxdt_fn : Callable
        Función que calcula dx/dt dado el voltaje.
    voltage : float
        Voltaje aplicado en este paso.
    dt : float
        Paso de tiempo.

    Retorna
    -------
    float : Estado actualizado.
    """
    return np.clip(x + dt * dxdt_fn(voltage), 0.0, 1.0)


def rk4_step(x: float, r_on: float, r_off: float,
             d: float, mu_v: float, voltage: float, dt: float,
             window_fn: Callable = None) -> tuple[float, float]:
    """
    Integración Runge-Kutta de 4to orden para el modelo Strukov.

    Ofrece mayor precisión que Euler para el mismo dt, especialmente
    útil cuando se analizan transitorios rápidos o se usa dt > 1e-3 s.

    Sistema de ecuaciones:
        I(t) = V(t) / R(x)
        dx/dt = β · I(t) · f(x)   con β = μ_v·R_on/D²

    Parámetros
    ----------
    x : float
        Variable de estado actual x = w/D.
    r_on, r_off, d, mu_v : float
        Parámetros físicos del dispositivo.
    voltage : float
        Voltaje aplicado en este paso (constante dentro del paso).
    dt : float
        Paso de tiempo.
    window_fn : Callable | None
        Función ventana f(x). Si None, usa f(x) = 1.

    Retorna
    -------
    tuple[float, float] : (x_{n+1}, corriente_media)
    """
    from .strukov_model import calculate_resistance, dxdt_strukov, window_biolek

    f = window_fn if window_fn is not None else (lambda xi: 1.0)

    def state_deriv(xi: float) -> float:
        r = max(calculate_resistance(xi, r_on, r_off), 1.0)
        i = voltage / r
        return dxdt_strukov(i, r_on, d, mu_v) * f(xi)

    # Corriente media (evaluada en x actual, representativa del paso)
    r_mid = max(calculate_resistance(x, r_on, r_off), 1.0)
    i_mean = voltage / r_mid

    k1 = state_deriv(x)
    k2 = state_deriv(np.clip(x + 0.5 * dt * k1, 0.0, 1.0))
    k3 = state_deriv(np.clip(x + 0.5 * dt * k2, 0.0, 1.0))
    k4 = state_deriv(np.clip(x + dt * k3, 0.0, 1.0))

    x_new = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.clip(x_new, 0.0, 1.0), i_mean
