"""
neurolab.devices.realism.c2c
==============================
Modificador Ciclo-a-Ciclo (C2C) basado en el proceso de Ornstein-Uhlenbeck.

Fuente de ABCs: neurolab.core.base_device
"""
import numpy as np
from neurolab.core.base_device import BaseRealismModifier
from neurolab.core.config import ElectricalConfig
from typing import Any


class C2CVariabilityModifier(BaseRealismModifier):
    """
    Modificador C2C (Cycle-to-Cycle) — Proceso de Ornstein-Uhlenbeck discreto.

    Ecuación de actualización:
        deta = -theta * eta * dt_ou + sigma * sqrt(dt_ou) * N(0, 1)
        factor_multiplicativo = max(0.1, 1.0 + eta)

    Parámetros:
        sigma  : Volatilidad del ruido (intensidad de las fluctuaciones).
        theta  : Tasa de retorno a la media (mayor = más rápida reversión).
        seed   : Semilla aleatoria para reproducibilidad científica.
    """

    def __init__(self, sigma: float = 0.05, theta: float = 1.0,
                 seed: int = 42, relative_std: float = None):
        self.sigma = relative_std if relative_std is not None else sigma
        self.theta = theta
        self.rng = np.random.default_rng(seed)
        self.eta: float = 0.0

    def modify_dxdt(self, dxdt: float, state: float, voltage: float, current: float,
                    electrical: ElectricalConfig, model_config: Any) -> float:
        if self.sigma <= 0:
            return dxdt

        dt_ou = 0.01  # Paso de integración OU (normalizado, independiente del dt de simulación)
        d_eta = -self.theta * self.eta * dt_ou + self.sigma * np.sqrt(dt_ou) * self.rng.normal(0.0, 1.0)
        self.eta += float(d_eta)

        factor = max(0.1, 1.0 + self.eta)
        return float(dxdt * factor)
