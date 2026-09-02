"""
neurolab.devices.realism.d2d
==============================
Modificador Device-to-Device (D2D) — Variabilidad estática de fabricación.

Fuente de ABCs: neurolab.core.base_device

CORRECCIÓN (bug anterior): El modificador D2D solo tenía `apply_to_electrical()`
que nadie en el pipeline de Memristor.step() llamaba. Ahora también implementa
`modify_dxdt()` aplicando un factor de escalado fijo calculado una sola vez
al momento de inicializar el dispositivo, modelando la variabilidad "quemada"
en la fabricación del chip (no varía ciclo a ciclo, sino dispositivo a dispositivo).
"""
import numpy as np
from neurolab.core.base_device import BaseRealismModifier
from neurolab.core.config import ElectricalConfig
from typing import Any


class D2DVariabilityModifier(BaseRealismModifier):
    """
    Modificador D2D (Device-to-Device).

    Simula la variabilidad de fabricación entre muestras:
    - Un factor escalar fijo `_d2d_factor` se calcula **una sola vez** al inicializar.
    - Este factor modula dx/dt en cada paso, escalando la velocidad de conmutación.
    - También puede modificar R_on/R_off vía `apply_to_electrical()`.

    Parámetros:
        variability_std : Desviación estándar de la distribución Gaussiana de variabilidad.
        seed            : Semilla para reproducibilidad estricta.
    """

    def __init__(self, variability_std: float = 0.05, seed: int = 42):
        self.variability_std = variability_std
        self.rng = np.random.default_rng(seed)

        # Factor D2D fijo: calculado una sola vez (fabricación estática)
        if variability_std > 0:
            self._d2d_factor = float(max(0.5, 1.0 + self.rng.normal(0.0, variability_std)))
        else:
            self._d2d_factor = 1.0

    def modify_dxdt(self, dxdt: float, state: float, voltage: float, current: float,
                    electrical: ElectricalConfig, model_config: Any) -> float:
        """
        Aplica el factor D2D fijo (variabilidad de fabricación) a la derivada de estado.
        El factor es constante durante toda la vida del dispositivo, modelando dispersión
        de fabricación (no varía por ciclo).
        """
        return float(dxdt * self._d2d_factor)

    def apply_to_electrical(self, electrical: ElectricalConfig) -> ElectricalConfig:
        """
        Aplica dispersión estocástica a R_on y R_off (variabilidad de fabricación D2D).
        Útil para caracterización multi-dispositivo fuera del loop de simulación.
        """
        if self.variability_std <= 0:
            return electrical

        factor_on = float(max(0.5, 1.0 + self.rng.normal(0.0, self.variability_std)))
        factor_off = float(max(0.5, 1.0 + self.rng.normal(0.0, self.variability_std)))

        new_r_on = electrical.r_on * factor_on
        new_r_off = max(new_r_on * 1.5, electrical.r_off * factor_off)

        return ElectricalConfig(
            r_on=new_r_on,
            r_off=new_r_off,
            initial_state=electrical.initial_state
        )
