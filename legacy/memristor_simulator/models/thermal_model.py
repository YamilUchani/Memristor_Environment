"""
models/thermal_model.py
========================
Modelo térmico de Joule Heating para el dispositivo memristivo TiO₂.

El calentamiento interno modifica la movilidad iónica y la resistencia,
produciendo efectos de histéresis adicionales a temperaturas elevadas.

Ecuación diferencial de temperatura:
    C_th · dT/dt = P(t) − (T(t) − T_amb) / R_th

donde P(t) = V²(t) / R(t) es la potencia disipada instantánea (W).
"""

import numpy as np
from dataclasses import dataclass
from ..config.parameters import StrukovParameters


@dataclass
class ThermalState:
    """Estado térmico del dispositivo en un instante dado."""
    temperature: float   # K
    power: float         # W (potencia disipada)
    heat_loss: float     # W (pérdida por conducción)
    dTdt: float          # K/s (tasa de cambio de temperatura)


class ThermalModel:
    """
    Modelo de temperatura de orden 1 para calentamiento por efecto Joule.

    Permite acoplar los efectos térmicos al modelo eléctrico de Strukov:
        - La resistencia se modifica como R(T) = R(x) · [1 + α_T·(T − T_amb)]
        - La movilidad iónica aumenta con temperatura (opcional)

    Parámetros (desde StrukovParameters)
    ------------------------------------
    R_th : Resistencia térmica (K/W). Controla la temperatura estacionaria.
    C_th : Capacidad térmica (J/K). Controla la inercia térmica.
    alpha_T : Coeficiente TCR de resistencia (1/K).
    T_amb : Temperatura ambiente (K).
    """

    def __init__(self, params: StrukovParameters):
        self.params = params
        self.T = params.T_amb   # Temperatura actual (K)
        self._history_T: list = []
        self._history_P: list = []

    @property
    def temperature(self) -> float:
        """Temperatura actual del dispositivo (K)."""
        return self.T

    @property
    def temperature_celsius(self) -> float:
        """Temperatura actual en grados Celsius."""
        return self.T - 273.15

    @property
    def delta_T(self) -> float:
        """Aumento de temperatura sobre el ambiente (K)."""
        return self.T - self.params.T_amb

    def step(self, voltage: float, resistance: float, dt: float) -> ThermalState:
        """
        Avanza un paso de tiempo Δt calculando la nueva temperatura.

        Parámetros
        ----------
        voltage : float
            Voltaje aplicado (V).
        resistance : float
            Resistencia eléctrica actual (Ω).
        dt : float
            Paso de tiempo (s).

        Retorna
        -------
        ThermalState : Estado térmico después del paso.
        """
        r = max(resistance, 1.0)
        power = (voltage ** 2) / r
        heat_loss = (self.T - self.params.T_amb) / self.params.R_th
        dTdt = (power - heat_loss) / self.params.C_th

        self.T += dTdt * dt
        self.T = max(self.T, self.params.T_amb)  # Temperatura mínima = ambiente

        self._history_T.append(self.T)
        self._history_P.append(power)

        return ThermalState(
            temperature=self.T,
            power=power,
            heat_loss=heat_loss,
            dTdt=dTdt,
        )

    def resistance_correction(self, r_base: float) -> float:
        """
        Aplica la corrección térmica de resistencia (TCR):
            R(T) = R_base · [1 + α_T · (T − T_amb)]

        Parámetros
        ----------
        r_base : float
            Resistencia eléctrica sin corrección térmica (Ω).

        Retorna
        -------
        float : Resistencia corregida por temperatura (Ω).
        """
        return r_base * (1.0 + self.params.alpha_T * self.delta_T)

    def steady_state_temperature(self, power_w: float) -> float:
        """
        Temperatura estacionaria para una potencia constante P (W):
            T_ss = T_amb + P · R_th

        Útil para estimar el rango operativo del dispositivo.
        """
        return self.params.T_amb + power_w * self.params.R_th

    def reset(self):
        """Reinicia la temperatura al valor ambiente."""
        self.T = self.params.T_amb
        self._history_T.clear()
        self._history_P.clear()

    def get_history(self) -> dict:
        """Devuelve el historial térmico registrado."""
        return {
            "temperature": np.array(self._history_T),
            "power": np.array(self._history_P),
        }
