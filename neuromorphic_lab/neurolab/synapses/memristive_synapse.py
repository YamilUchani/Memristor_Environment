"""
neurolab.synapses.memristive_synapse
===================================
Sinapsis memristiva: el peso sináptico es proporcional a la conductancia.
Integra un modelo memristivo (Strukov, Yakopcic, etc.) como elemento de peso.
"""

from typing import Optional
import numpy as np
from neurolab.synapses.base import Synapse


class MemristiveSynapse(Synapse):
    """
    Sinapsis basada en memristor.

    El peso w se mapea linealmente a la conductancia G:
        w = (G - G_min) / (G_max - G_min)

    La dinámica del memristor actualiza G en cada paso de simulación.
    """

    def __init__(self, memristor, name: str = "memristive_synapse"):
        """
        Parameters
        ----------
        memristor : Memristor
            Objeto memristor que define la dinámica de G.
        """
        super().__init__(name)
        self.memristor = memristor
        self._update_from_memristor()

    @property
    def r_off(self) -> float:
        """Resistencia máxima (OFF) del memristor en Ohmios."""
        if hasattr(self.memristor, "electrical"):
            return float(self.memristor.electrical.r_off)
        return float(getattr(self.memristor, "ROFF", getattr(self.memristor, "r_off", 16000.0)))

    @property
    def r_on(self) -> float:
        """Resistencia mínima (ON) del memristor en Ohmios."""
        if hasattr(self.memristor, "electrical"):
            return float(self.memristor.electrical.r_on)
        return float(getattr(self.memristor, "RON", getattr(self.memristor, "r_on", 100.0)))

    @property
    def G_min(self) -> float:
        """Conductancia mínima (Siemens)."""
        return 1.0 / self.r_off

    @property
    def G_max(self) -> float:
        """Conductancia máxima (Siemens)."""
        return 1.0 / self.r_on

    @property
    def conductance(self) -> float:
        """Conductancia sináptica (S)."""
        return self._conductance

    @conductance.setter
    def conductance(self, value: float) -> None:
        """Establece la conductancia y sincroniza el peso y el estado del memristor."""
        G_val = float(np.clip(value, self.G_min, self.G_max))
        self._conductance = G_val
        self._update_weight_from_conductance()

        # Sincronizar el estado físico x del memristor
        if hasattr(self.memristor, "x"):
            R_new = 1.0 / G_val
            r_off = self.r_off
            r_on = self.r_on
            if r_off != r_on:
                x_new = (r_off - R_new) / (r_off - r_on)
                self.memristor.x = float(np.clip(x_new, 0.0, 1.0))

    def _update_from_memristor(self) -> None:
        """Sincroniza G y w con el estado actual del memristor."""
        self._conductance = float(self.memristor.conductance)
        self._update_weight_from_conductance()

    def update(self, pre_spike: bool, post_spike: bool, dt: float, V_applied: Optional[float] = None) -> None:
        """
        Actualiza el memristor (y por tanto el peso) en función de los spikes.

        Parameters
        ----------
        pre_spike : bool
            True si hay spike presináptico en este instante.
        post_spike : bool
            True si hay spike postsináptico en este instante.
        dt : float
            Paso temporal (s).
        V_applied : float, optional
            Voltaje aplicado al memristor. Si es None, se calcula de los spikes.
        """
        if V_applied is None:
            V_applied = self._compute_voltage(pre_spike, post_spike)

        self.memristor.update(V_applied, dt)
        self._update_from_memristor()

    def _compute_voltage(self, pre_spike: bool, post_spike: bool) -> float:
        """
        Calcula el voltaje efectivo sobre el memristor.
        - Solo pre_spike: +V_pulse (potenciación)
        - Solo post_spike: -V_pulse (depresión)
        - Ambos / ninguno: 0.0 V
        """
        V_pulse = 0.5  # Voltaje de pulso por defecto (V)
        if pre_spike and not post_spike:
            return +V_pulse
        elif post_spike and not pre_spike:
            return -V_pulse
        else:
            return 0.0

    def reset(self) -> None:
        """Reinicia la sinapsis y el memristor."""
        if hasattr(self.memristor, "reset"):
            self.memristor.reset()
        self._update_from_memristor()
