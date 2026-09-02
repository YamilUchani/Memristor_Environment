"""
neurolab.neurons.config
=======================
Contiene los parámetros configurables de las neuronas.
"""
from dataclasses import dataclass

@dataclass
class LIFConfig:
    """Configuración de parámetros físicos y eléctricos para la neurona LIF."""
    c_m: float = 500e-9       # Capacitancia de membrana (Faradios) -> 500 nF
    r_leak: float = 100e3     # Resistencia de fuga (Ohmios) -> 100 kΩ
    v_rest: float = 0.0       # Potencial de reposo (Voltios)
    v_th: float = 0.95        # Potencial de umbral (Voltios)
    v_reset: float = 0.15     # Potencial de reinicio (Voltios)
    t_ref: float = 0.002      # Período refractario (Segundos)

    def __post_init__(self):
        """Valida que los parámetros físicos sean coherentes."""
        if self.c_m <= 0:
            raise ValueError("La capacitancia de membrana (c_m) debe ser estrictamente mayor que cero.")
        if self.r_leak <= 0:
            raise ValueError("La resistencia de fuga (r_leak) debe ser estrictamente mayor que cero.")
        if self.v_th <= self.v_reset:
            raise ValueError("El potencial de umbral (v_th) debe ser superior al potencial de reinicio (v_reset).")
        if self.t_ref < 0:
            raise ValueError("El período refractario no puede ser negativo.")

    @property
    def tau(self) -> float:
        """
        Calcula la constante de tiempo tau = R * C.
        Con R = 100 kOhm y C = 500 nF, tau = 0.05 segundos (50 ms).
        """
        return self.r_leak * self.c_m
