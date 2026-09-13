"""
neurolab.neurons.config
======================
Contiene los parámetros configurables de la neurona Leaky Integrate-and-Fire (LIF).
"""
from dataclasses import dataclass
import warnings

@dataclass
class LIFConfig:
    """
    Configuración de parámetros físicos y eléctricos para la neurona LIF.
    """
    c_m: float = 100e-9       # Capacitancia de membrana (Faradios) -> 100 nF
    r_leak: float = 1e6       # Resistencia de fuga (Ohmios) -> 1 MΩ
    r_series: float = 100e3   # Resistencia en serie de entrada (Ohmios) -> 100 kΩ
    v_rest: float = 0.0       # Potencial de reposo (Voltios)
    v_th: float = 1.0         # Potencial de umbral (Voltios)
    v_reset: float = 0.0      # Potencial de reinicio (Voltios)
    t_ref: float = 0.0        # Período refractario (Segundos)

    def __post_init__(self):
        """Valida que los parámetros físicos sean coherentes."""
        if self.c_m <= 0:
            raise ValueError("La capacitancia de membrana (c_m) debe ser estrictamente mayor que cero.")
        if self.r_leak <= 0:
            raise ValueError("La resistencia de fuga (r_leak) debe ser estrictamente mayor que cero.")
        if self.r_series <= 0:
            raise ValueError("La resistencia en serie (r_series) debe ser estrictamente mayor que cero.")
        if self.v_th <= self.v_reset:
            raise ValueError("El potencial de umbral (v_th) debe ser superior al potencial de reinicio (v_reset).")
        if self.t_ref < 0:
            raise ValueError("El período refractario no puede ser negativo.")

    @property
    def r_eq(self) -> float:
        """Resistencia equivalente R_eq = R_series || R_leak."""
        return (self.r_series * self.r_leak) / (self.r_series + self.r_leak)

    @property
    def tau(self) -> float:
        """Calcula la constante de tiempo equivalente tau = R_eq * C_m."""
        return self.r_eq * self.c_m

    @property
    def has_afterhyperpolarization(self) -> bool:
        """True si V_reset < V_rest (hiperpolarización post-spike activa)."""
        return self.v_reset < self.v_rest
