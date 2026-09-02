"""
neurolab.devices.config
=======================
Configuraciones estructurales del dispositivo memristivo.

- `ElectricalConfig` y `StrukovConfig` se importan desde `core.config`
  (fuente de verdad única).
- `DeviceConfig`, `StateConfig`, `WindowConfig`, `StochasticConfig`,
  `RealismConfig` se definen aquí como parte de la capa `devices`.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# ── Reexports desde core (fuente de verdad) ─────────────────────────────────
from neurolab.core.config import ElectricalConfig, StrukovConfig  # noqa: F401

# ── Configuraciones propias de la capa devices ───────────────────────────────

@dataclass
class DeviceConfig:
    """A. Configuración General de Identidad del Dispositivo."""
    name: str = "Strukov 2008 - Figure 2b"
    family: str = "oxide_memristor"
    model: str = "strukov"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def device_name(self) -> str:
        return self.name

    @property
    def device_family(self) -> str:
        return self.family

    @property
    def model_name(self) -> str:
        return self.model


@dataclass
class StateConfig:
    """C. Configuración del Estado Interno Normalizado."""
    x_init: float = 0.1
    x_min: float = 0.0
    x_max: float = 1.0
    normalized: bool = True

    def __post_init__(self):
        if not (self.x_min <= self.x_init <= self.x_max):
            raise ValueError("x_init debe estar acotado entre x_min y x_max.")


@dataclass
class WindowConfig:
    """Configuración de Ventanas y Efectos de Frontera."""
    enabled: bool = False
    window_type: str = "biolek"   # "none" | "biolek" | "joglekar"
    p: int = 5


@dataclass
class StochasticConfig:
    """Configuración Universal de Estocasticidad."""
    enabled: bool = False

    # D2D — Device-to-Device (Variabilidad de Fabricación)
    enable_d2d: bool = False
    d2d_distribution: str = "gaussian"
    d2d_ron_sigma: float = 0.05
    d2d_roff_sigma: float = 0.05
    d2d_mu_sigma: float = 0.05

    # C2C — Cycle-to-Cycle (Proceso Ornstein-Uhlenbeck)
    enable_c2c: bool = False
    c2c_model: str = "ornstein_uhlenbeck"
    c2c_sigma: float = 0.05
    c2c_theta: float = 1.0

    # Ruido Dinámico de Lectura
    enable_noise: bool = False
    noise_std: float = 1e-6
    noise_type: str = "gaussian"

    # Reproducibilidad Científica
    seed: Optional[int] = 42


@dataclass
class RealismConfig:
    """D. Configuración de Realismo agrupada."""
    window: WindowConfig = field(default_factory=WindowConfig)
    stochastic: StochasticConfig = field(default_factory=StochasticConfig)
