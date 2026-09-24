"""
neurolab.synapses.plasticity
============================
Reglas de plasticidad sináptica: LTP y LTD.
"""

import numpy as np
from neurolab.synapses.base import Synapse


class LTPRule:
    """Regla de potenciación a largo plazo (Long-Term Potentiation)."""

    def __init__(self, n_pulses: int = 40, V_pulse: float = +1.0, pulse_width: float = 100e-3):
        self.n_pulses = n_pulses
        self.V_pulse = V_pulse
        self.pulse_width = pulse_width

    def apply(self, synapse: Synapse, dt: float = 1e-3) -> np.ndarray:
        """Aplica n pulsos positivos y registra la evolución de G."""
        G_history = [synapse.conductance]
        for _ in range(self.n_pulses):
            synapse.update(pre_spike=True, post_spike=False, dt=dt, V_applied=self.V_pulse)
            G_history.append(synapse.conductance)
        return np.array(G_history)


class LTDRule:
    """Regla de depresión a largo plazo (Long-Term Depression)."""

    def __init__(self, n_pulses: int = 40, V_pulse: float = -1.0, pulse_width: float = 100e-3):
        self.n_pulses = n_pulses
        self.V_pulse = V_pulse
        self.pulse_width = pulse_width

    def apply(self, synapse: Synapse, dt: float = 1e-3) -> np.ndarray:
        """Aplica n pulsos negativos y registra la evolución de G."""
        G_history = [synapse.conductance]
        for _ in range(self.n_pulses):
            synapse.update(pre_spike=False, post_spike=True, dt=dt, V_applied=self.V_pulse)
            G_history.append(synapse.conductance)
        return np.array(G_history)
