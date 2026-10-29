"""
neurolab.synapses.base
======================
Clase base abstracta para sinapsis.
Define la interfaz común para todas las sinapsis (memristivas o no).
"""

from abc import ABC, abstractmethod
import numpy as np


class Synapse(ABC):
    """Clase base abstracta para una sinapsis."""

    def __init__(self, name: str = "synapse"):
        self.name = name
        self._weight: float = 1.0       # Peso sináptico normalizado [0, 1]
        self._conductance: float = 1e-6 # Conductancia (S)
        self._trace_pre: float = 0.0    # Traza presináptica (para STDP)
        self._trace_post: float = 0.0   # Traza postsináptica (para STDP)

    @property
    def weight(self) -> float:
        """Peso sináptico normalizado [0, 1]."""
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        """Establece el peso y actualiza la conductancia."""
        self._weight = float(np.clip(value, 0.0, 1.0))
        self._update_conductance_from_weight()

    @property
    def conductance(self) -> float:
        """Conductancia sináptica (S)."""
        return self._conductance

    @conductance.setter
    def conductance(self, value: float) -> None:
        """Establece la conductancia y actualiza el peso."""
        self._conductance = max(float(value), 1e-9)
        self._update_weight_from_conductance()

    def _update_weight_from_conductance(self) -> None:
        """Actualiza w a partir de G. Mapeo lineal."""
        Gmin, Gmax = self.G_min, self.G_max
        if Gmax > Gmin:
            self._weight = float(np.clip((self._conductance - Gmin) / (Gmax - Gmin), 0.0, 1.0))
        else:
            self._weight = 0.5

    def _update_conductance_from_weight(self) -> None:
        """Actualiza G a partir de w. Mapeo lineal."""
        self._conductance = self.G_min + self._weight * (self.G_max - self.G_min)

    @abstractmethod
    def update(self, pre_spike: bool, post_spike: bool, dt: float, V_applied: float = None) -> None:
        """Actualiza el peso sináptico. Debe ser implementado por subclases."""
        pass

    def reset(self) -> None:
        """Reinicia la sinapsis a su estado inicial."""
        self._weight = 1.0
        self._conductance = 1e-6
        self._trace_pre = 0.0
        self._trace_post = 0.0

    @property
    def G_min(self) -> float:
        """Conductancia mínima (HRS). Por defecto: 1e-7 S."""
        return 1e-7

    @property
    def G_max(self) -> float:
        """Conductancia máxima (LRS). Por defecto: 1e-3 S."""
        return 1e-3
