"""
neurolab.devices.models.strukov
=================================
Modelo matemático determinista de Strukov et al. (Nature 2008).

Importa la interfaz base desde `core.base_device` (fuente de verdad).
Importa los dataclasses de configuración desde `core.config`.
"""
from neurolab.core.base_device import BaseMathModel
from neurolab.core.config import ElectricalConfig, StrukovConfig
from typing import Optional


class StrukovMathModel(BaseMathModel):
    """
    Modelo Matemático Determinista de Strukov et al. (Nature 2008).

    Ecuación diferencial de deriva iónica normalizada:
        dx/dt = (mu_v * R_on / D²) * I(t)

    donde:
        x     = w/D  fracción normalizada de región dopada [0.0, 1.0]
        mu_v  = movilidad de vacancias de oxígeno (m² / V·s)
        D     = espesor físico de la capa activa (m)
        R_on  = resistencia mínima (estado ON) en Ω
        I(t)  = corriente instantánea (A)

    Referencia: Strukov et al., "The missing memristor found", Nature 453, 2008.
    """

    def compute_dxdt(
        self,
        state: float,
        voltage: float,
        current: float,
        electrical: ElectricalConfig,
        model_config: Optional[StrukovConfig] = None
    ) -> float:
        """Calcula la velocidad de deriva iónica pura sin perturbaciones ni ventanas."""
        cfg = model_config if model_config is not None else StrukovConfig()
        # dx/dt = (mu_v * R_on / D²) * I(t)
        return float((cfg.mu_v * electrical.r_on / (cfg.D ** 2)) * current)
