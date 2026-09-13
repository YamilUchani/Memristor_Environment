"""
neurolab.devices.realism.c2c
==============================
Modificador Ciclo-a-Ciclo (C2C) basado en el proceso de Ornstein-Uhlenbeck.

Fuente de ABCs: neurolab.core.base_device

Cambios (v2026.09.11):
  - Bug #2 fix: dt_ou ya NO está hardcodeado a 0.02 s.
    El bucle de simulación debe llamar a `set_simulation_dt(dt)` antes de correr,
    o pasar dt_simulation al constructor. Si no se especifica, se mantiene 0.02 s
    como fallback para retrocompatibilidad.
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
        sigma           : Volatilidad del ruido (intensidad de las fluctuaciones).
        theta           : Tasa de retorno a la media (mayor = más rápida reversión).
        seed            : Semilla aleatoria para reproducibilidad científica.
        dt_simulation   : Paso temporal real de la simulación (s). Si se especifica,
                          el proceso OU queda calibrado a la escala temporal correcta.
                          Se puede actualizar en tiempo de ejecución via set_simulation_dt().

    Bug #2 fix (v2026.09.11): dt_ou ya no es hardcodeado. Escala con el dt real.
    """

    # Factor de escala del dt para el proceso OU respecto al dt de simulación.
    # OU opera a una escala temporal más lenta que el paso de integración del memristor.
    # Un factor de 20 significa que el ruido evoluciona ~20x más lento que la dinámica base.
    _OU_DT_SCALE: float = 20.0

    def __init__(self, sigma: float = 0.05, theta: float = 1.0,
                 seed: int = 42, relative_std: float = None,
                 dt_simulation: float = None):
        self.sigma = relative_std if relative_std is not None else sigma
        self.theta = theta
        self.rng = np.random.default_rng(seed)
        self.eta: float = 0.0
        # Bug #2 fix: dt real de la simulación. Fallback a 0.02 s para retrocompat.
        self._dt_sim: float = dt_simulation if dt_simulation is not None else 0.02

    def set_simulation_dt(self, dt: float) -> None:
        """
        Actualiza el paso temporal de la simulación para calibrar el proceso OU.
        Debe llamarse desde el bucle de simulación antes de empezar a iterar.

        Args:
            dt: Paso de integración real de la simulación en segundos.
        """
        self._dt_sim = dt

    def modify_dxdt(self, dxdt: float, state: float, voltage: float, current: float,
                    electrical: ElectricalConfig, model_config: Any) -> float:
        if self.sigma <= 0:
            return dxdt

        # Bug #2 fix: dt_ou escala con el dt real de simulación (no hardcodeado).
        # Se usa un factor de escala para que el OU opere más lento que la dinámica base.
        dt_ou = self._dt_sim * self._OU_DT_SCALE
        d_eta = -self.theta * self.eta * dt_ou + self.sigma * np.sqrt(dt_ou) * self.rng.normal(0.0, 1.0)
        self.eta += float(d_eta)

        factor = max(0.1, 1.0 + self.eta)
        return float(dxdt * factor)
