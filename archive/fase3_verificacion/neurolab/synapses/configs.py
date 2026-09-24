"""
neurolab.synapses.configs
=========================
Configuraciones de plasticidad sináptica.
"""

from dataclasses import dataclass

@dataclass
class PlasticityConfig:
    """Configuración de parámetros para reglas de plasticidad (LTP, LTD, STDP)."""
    # LTP / LTD
    n_pulses_ltp: int = 40
    n_pulses_ltd: int = 40
    v_pulse_ltp: float = 1.0
    v_pulse_ltd: float = -1.0
    pulse_width: float = 100e-3

    # STDP
    A_plus: float = 0.05
    A_minus: float = -0.025
    tau_plus: float = 17e-3
    tau_minus: float = 34e-3
    w_min: float = 0.0
    w_max: float = 1.0
    w_0: float = 0.0
