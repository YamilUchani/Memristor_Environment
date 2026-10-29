"""
neurolab.synapses.stdp
======================
STDP con normalización porcentual.

Cada evento STDP modifica la conductancia un porcentaje del
valor actual, garantizando cambios físicamente realistas.
"""

import numpy as np
from neurolab.synapses.base import Synapse


class STDPRule:
    """
    STDP Hebbiano con normalización porcentual.

    Parámetros
    ----------
    A_plus, A_minus : float
        Amplitudes del aprendizaje (en [0, 1]).
    tau_plus, tau_minus : float
        Constantes de tiempo (s).
    max_percent_change : float
        Cambio máximo por evento (% del valor actual).
        Recomendado: 0.1 a 1.0
    w_min, w_max, w_0 : float
        Límites y offset del peso sináptico.
    """

    def __init__(
        self,
        A_plus: float = 0.05,
        A_minus: float = -0.025,
        tau_plus: float = 17e-3,
        tau_minus: float = 34e-3,
        max_percent_change: float = 0.5,
        w_min: float = 0.0,
        w_max: float = 1.0,
        w_0: float = 0.0
    ):
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.max_percent_change = max_percent_change
        self.w_min = w_min
        self.w_max = w_max
        self.w_0 = w_0

    def delta_w(self, dt: float) -> float:
        """ΔW normalizado en [0, 1]."""
        if dt > 0:
            return self.A_plus * np.exp(-dt / self.tau_plus) + self.w_0
        elif dt < 0:
            return self.A_minus * np.exp(dt / self.tau_minus) + self.w_0
        else:
            return 0.0

    def apply(self, synapse: Synapse, dt: float) -> float:
        """
        Aplica STDP con normalización porcentual a una sinapsis.

        Returns
        -------
        dW : float
            Cambio normalizado calculado por delta_w(dt).
        """
        dW = self.delta_w(dt)

        # Conductancia actual
        G_actual = synapse.conductance

        # Máximo cambio permitido (% de la conductancia actual)
        max_dG = G_actual * self.max_percent_change / 100.0

        # Escalado proporcional (conserva la curva exponencial)
        if dt > 0:
            fraction = dW / self.A_plus if self.A_plus != 0 else 0.0
        elif dt < 0:
            fraction = dW / abs(self.A_minus) if self.A_minus != 0 else 0.0
        else:
            fraction = 0.0

        dG = max_dG * fraction

        # Aplicar a G con límites físicos
        new_G = float(np.clip(G_actual + dG, synapse.G_min, synapse.G_max))
        synapse.conductance = new_G

        return dW


class AntiSTDPRule(STDPRule):
    """STDP Anti-Hebbiano con normalización porcentual."""

    def delta_w(self, dt: float) -> float:
        if dt > 0:
            return -self.A_plus * np.exp(-dt / self.tau_plus) + self.w_0
        elif dt < 0:
            return -self.A_minus * np.exp(dt / self.tau_minus) + self.w_0
        else:
            return 0.0

