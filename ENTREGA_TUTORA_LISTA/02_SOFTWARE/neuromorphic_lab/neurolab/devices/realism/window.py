"""
neurolab.devices.realism.window
================================
Modificadores de Funciones de Ventana (Boundary Effects).

Fuente de ABCs: neurolab.core.base_device
"""
import numpy as np
from neurolab.core.base_device import BaseRealismModifier
from neurolab.core.config import ElectricalConfig
from typing import Any


class BiolekWindowModifier(BaseRealismModifier):
    """
    Función de Ventana de Biolek: f(x, i) = 1 - (x - stp(-i))^(2p)

    stp(-i) = 1 si i < 0 (RESET → atenúa al acercarse a x=0)
              0 si i >= 0 (SET  → atenúa al acercarse a x=1)

    Referencia: Biolek et al., RADIOENGINEERING, 2009.
    """

    def __init__(self, p: int = 5):
        self.p = p

    def modify_dxdt(self, dxdt: float, state: float, voltage: float, current: float,
                    electrical: ElectricalConfig, model_config: Any) -> float:
        x = state
        i = current
        stp = 1.0 if i < 0.0 else 0.0
        f_win = 1.0 - np.power(x - stp, 2 * self.p)
        return float(dxdt * f_win)


class JoglekarWindowModifier(BaseRealismModifier):
    """
    Función de Ventana de Joglekar: f(x) = 1 - (2x - 1)^(2p)

    Referencia: Joglekar & Wolf, Eur. Phys. J. B, 2009.
    """

    def __init__(self, p: int = 5):
        self.p = p

    def modify_dxdt(self, dxdt: float, state: float, voltage: float, current: float,
                    electrical: ElectricalConfig, model_config: Any) -> float:
        x = state
        f_win = 1.0 - np.power(2.0 * x - 1.0, 2 * self.p)
        return float(dxdt * f_win)
