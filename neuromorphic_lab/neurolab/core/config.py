from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class DeviceIdentity:
    """NIVEL A: Identidad y metadatos del dispositivo."""
    device_name: str = "Strukov TiO2"
    device_family: str = "oxide_memristor"
    model_name: str = "strukov"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ElectricalConfig:
    """NIVEL B: Parámetros eléctricos universales del dispositivo."""
    r_on: float = 100.0        # R_min = 100 Ω (Strukov 2008)
    r_off: float = 16_000.0    # R_max = 16 kΩ (Strukov 2008, Ratio 160)
    initial_state: float = 0.10 # x_0 = w_0/D = 0.10 (w_0 = 1 nm)

    def __post_init__(self):
        if self.r_on <= 0 or self.r_off <= 0:
            raise ValueError("R_on y R_off deben ser estrictamente positivos.")
        if self.r_on >= self.r_off:
            raise ValueError("R_on debe ser menor que R_off.")
        if not (0.0 <= self.initial_state <= 1.0):
            raise ValueError("initial_state debe estar acotado en el rango [0.0, 1.0].")

    @property
    def g_on(self) -> float:
        """Conductancia máxima en Siemens (1 / R_on)."""
        return 1.0 / self.r_on

    @property
    def g_off(self) -> float:
        """Conductancia mínima en Siemens (1 / R_off)."""
        return 1.0 / self.r_off

@dataclass
class StrukovConfig:
    """NIVEL E: Parámetros físicos específicos del modelo de Strukov (2008)."""
    D: float = 10e-9        # Espesor físico de la capa activa en metros (10 nm)
    mu_v: float = 1e-14     # Movilidad de vacancias de oxígeno en m^2 / (V * s)

    def __post_init__(self):
        if self.D <= 0:
            raise ValueError("El grosor D debe ser estrictamente positivo.")
        if self.mu_v <= 0:
            raise ValueError("La movilidad mu_v debe ser estrictamente positiva.")
