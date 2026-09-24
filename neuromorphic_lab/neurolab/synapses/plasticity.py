"""
neurolab.synapses.plasticity
============================
Reglas de plasticidad sináptica: LTP y LTD.
"""

import numpy as np
from neurolab.synapses.base import Synapse


class LTPRule:
    """Regla de potenciación a largo plazo (Long-Term Potentiation) con opción de saturación no lineal y periodo de relajación."""

    def __init__(self, n_pulses: int = 40, V_pulse: float = +1.0, pulse_width: float = 100e-3,
                 saturation: bool = False, tau_sat: float = 30.0, t_off: float = 0.02):
        self.n_pulses = n_pulses
        self.V_pulse = V_pulse
        self.pulse_width = pulse_width
        self.saturation = saturation
        self.tau_sat = tau_sat
        self.t_off = t_off

    def apply(self, synapse: Synapse, dt: float = 1e-3) -> np.ndarray:
        """Aplica n pulsos positivos y registra la evolución de G."""
        G_history = [synapse.conductance]
        for n in range(self.n_pulses):
            v_app = self.V_pulse * np.exp(-n / self.tau_sat) if self.saturation else self.V_pulse
            synapse.update(pre_spike=True, post_spike=False, dt=dt, V_applied=v_app)
            if self.t_off > 0:
                synapse.update(pre_spike=False, post_spike=False, dt=self.t_off, V_applied=0.0)
            G_history.append(synapse.conductance)
        return np.array(G_history)


class LTDRule:
    """Regla de depresión a largo plazo (Long-Term Depression) con opción de saturación no lineal y periodo de relajación."""

    def __init__(self, n_pulses: int = 40, V_pulse: float = -1.0, pulse_width: float = 100e-3,
                 saturation: bool = False, tau_sat: float = 30.0, t_off: float = 0.02):
        self.n_pulses = n_pulses
        self.V_pulse = V_pulse
        self.pulse_width = pulse_width
        self.saturation = saturation
        self.tau_sat = tau_sat
        self.t_off = t_off

    def apply(self, synapse: Synapse, dt: float = 1e-3) -> np.ndarray:
        """Aplica n pulsos negativos y registra la evolución de G."""
        G_history = [synapse.conductance]
        for n in range(self.n_pulses):
            v_app = self.V_pulse * np.exp(-n / self.tau_sat) if self.saturation else self.V_pulse
            synapse.update(pre_spike=False, post_spike=True, dt=dt, V_applied=v_app)
            if self.t_off > 0:
                synapse.update(pre_spike=False, post_spike=False, dt=self.t_off, V_applied=0.0)
            G_history.append(synapse.conductance)
        return np.array(G_history)
