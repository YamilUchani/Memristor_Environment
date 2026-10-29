"""
neurolab.synapses.plasticity
============================
Reglas de plasticidad sináptica: LTP y LTD.
"""

import numpy as np
from neurolab.synapses.base import Synapse


def _apply_pulse_train(
    synapse, n_pulses, voltage, pulse_width, dt, saturation, tau_sat,
    t_off, pre_spike, post_spike,
):
    if not isinstance(n_pulses, (int, np.integer)):
        raise ValueError("n_pulses must be an integer")
    if n_pulses < 0:
        raise ValueError("n_pulses must be non-negative")
    values = (voltage, pulse_width, dt, t_off)
    if not np.all(np.isfinite(values)):
        raise ValueError("voltage, pulse_width, dt, and t_off must be finite")
    if pulse_width <= 0.0:
        raise ValueError("pulse_width must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if t_off < 0.0:
        raise ValueError("t_off must be non-negative")
    if saturation and (not np.isfinite(tau_sat) or tau_sat <= 0.0):
        raise ValueError("tau_sat must be finite and positive when saturation is enabled")

    G_history = [synapse.conductance]
    n_steps = max(1, int(np.ceil(pulse_width / dt)))
    step_dt = pulse_width / n_steps
    off_steps = int(np.ceil(t_off / dt)) if t_off > 0.0 else 0
    off_dt = t_off / off_steps if off_steps else 0.0
    for pulse in range(n_pulses):
        v_app = voltage * np.exp(-pulse / tau_sat) if saturation else voltage
        for _ in range(n_steps):
            synapse.update(pre_spike, post_spike, dt=step_dt, V_applied=v_app)
        for _ in range(off_steps):
            synapse.update(False, False, dt=off_dt, V_applied=0.0)
        G_history.append(synapse.conductance)
    return np.asarray(G_history)


class LTPRule:
    """LTP por pulsos; pulse_width y t_off se expresan en segundos."""

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
        return _apply_pulse_train(
            synapse, self.n_pulses, self.V_pulse, self.pulse_width, dt,
            self.saturation, self.tau_sat, self.t_off, True, False,
        )


class LTDRule:
    """LTD por pulsos; pulse_width y t_off se expresan en segundos."""

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
        return _apply_pulse_train(
            synapse, self.n_pulses, self.V_pulse, self.pulse_width, dt,
            self.saturation, self.tau_sat, self.t_off, False, True,
        )
