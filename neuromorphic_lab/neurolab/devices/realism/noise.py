"""
neurolab.devices.realism.noise
================================
Modificador de Ruido Térmico de Lectura.

Fuente de ABCs: neurolab.core.base_device
"""
import numpy as np
from neurolab.core.base_device import BaseRealismModifier


class ThermalNoiseModifier(BaseRealismModifier):
    """
    Ruido Térmico de lectura modelado como N(0, noise_std).

    Se aplica a la corriente de salida en cada paso de integración,
    simulando el ruido de Johnson-Nyquist en la medición.

    Parámetros:
        noise_std : Desviación estándar del ruido (A). Típico: 1e-6 A.
        seed      : Semilla para reproducibilidad científica.
    """

    def __init__(self, noise_std: float = 1e-6, seed: int = 42):
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def modify_current(self, current: float, voltage: float, state: float) -> float:
        if self.noise_std <= 0:
            return current
        noise = float(self.rng.normal(0.0, self.noise_std))
        return float(current + noise)
