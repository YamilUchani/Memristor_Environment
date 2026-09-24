"""
neurolab.synapses.stdp
======================
Implementación de STDP (Spike-Timing-Dependent Plasticity).
Incluye variantes Hebbiana y anti-Hebbiana.
"""

import numpy as np
from neurolab.synapses.base import Synapse


class STDPRule:
    """
    Regla STDP con ventana exponencial asimétrica.

    ΔW = A+ * exp(-Δt/τ+)   si Δt > 0 (pre antes que post)
    ΔW = A- * exp(-|Δt|/τ-) si Δt < 0 (post antes que pre)

    donde Δt = t_post - t_pre.
    """

    def __init__(
        self,
        A_plus: float = 0.05,
        A_minus: float = -0.025,
        tau_plus: float = 17e-3,
        tau_minus: float = 34e-3,
        w_min: float = 0.0,
        w_max: float = 1.0,
        w_0: float = 0.0
    ):
        """
        Parameters
        ----------
        A_plus : float
            Amplitud máxima de potenciación.
        A_minus : float
            Amplitud máxima de depresión (negativa).
        tau_plus : float
            Constante de tiempo de potenciación (s).
        tau_minus : float
            Constante de tiempo de depresión (s).
        w_min, w_max : float
            Límites del peso sináptico.
        w_0 : float
            Offset del peso.
        """
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_min = w_min
        self.w_max = w_max
        self.w_0 = w_0

    def delta_w(self, dt: float) -> float:
        """
        Calcula el cambio de peso ΔW para una diferencia temporal Δt.

        Parameters
        ----------
        dt : float
            Δt = t_post - t_pre (s).

        Returns
        -------
        dW : float
            Cambio de peso sináptico.
        """
        if dt > 0:
            return self.A_plus * np.exp(-dt / self.tau_plus) + self.w_0
        elif dt < 0:
            return self.A_minus * np.exp(dt / self.tau_minus) + self.w_0
        else:
            return 0.0

    def apply(self, synapse: Synapse, dt: float) -> float:
        """
        Aplica la regla STDP a una sinapsis.

        Parameters
        ----------
        synapse : Synapse
            Sinapsis a modificar.
        dt : float
            Δt = t_post - t_pre (s).

        Returns
        -------
        dW : float
            Cambio de peso aplicado.
        """
        dW = self.delta_w(dt)
        new_weight = float(np.clip(synapse.weight + dW, self.w_min, self.w_max))
        synapse.weight = new_weight
        return dW


class AntiSTDPRule(STDPRule):
    """
    STDP anti-Hebbiana: regla invertida.

    ΔW = -A+ * exp(-Δt/τ+)   si Δt > 0
    ΔW = -A- * exp(-|Δt|/τ-) si Δt < 0
    """

    def delta_w(self, dt: float) -> float:
        if dt > 0:
            return -self.A_plus * np.exp(-dt / self.tau_plus) + self.w_0
        elif dt < 0:
            return -self.A_minus * np.exp(dt / self.tau_minus) + self.w_0
        else:
            return 0.0
