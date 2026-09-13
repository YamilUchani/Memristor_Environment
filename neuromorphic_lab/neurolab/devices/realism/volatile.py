"""
neurolab.devices.realism.volatile
===================================
Modificador de Decaimiento Volátil (Memristor Difusivo).

Implementa el término de relajación espontánea que convierte el modelo de Strukov
(no volátil, memoria perfecta) en un modelo de memristor VOLÁTIL (difusivo),
análogo a los dispositivos Ag/SiO₂, NbOx o VO₂ usados en sensores neuromórficos reales.

Referencia física:
  Wang et al. (2025), "Memristive Approaches to Biologically Plausible Spiking Neural
  Networks", Sección 4.1-4.2 — "Diffusive TSMs (Threshold Switching Memristors)":
  estos dispositivos tienen relajación espontánea: cuando se retira o disminuye el estímulo,
  el filamento conductor se disuelve y el estado x decae libremente hacia x_eq por sí solo.

Ecuación diferencial modificada (Strukov + Relajación Volátil):
─────────────────────────────────────────────────────────────────
  dx/dt = [μᵥ·R_ON/D²·I(t)·f(x)] + [-(x - x_eq) / τ_relax]
            ↑ Deriva iónica           ↑ Relajación volátil
            (Strukov + Ventana)       (Decaimiento espontáneo)

Donde:
  x_eq    : Estado de equilibrio bajo relajación (= x0_override o electrical.initial_state)
  τ_relax : Tiempo de relajación (s). Cuánto tarda x en volver a x_eq tras un estímulo.
"""
import math
import numpy as np
from neurolab.core.base_device import BaseRealismModifier
from neurolab.core.config import ElectricalConfig
from typing import Any


class VolatileDecayModifier(BaseRealismModifier):
    """
    Modificador de Relajación Volátil (Memristor Difusivo).

    Añade al pipeline de dx/dt el término:
        dx/dt_relax = -(x - x_eq) / τ_relax

    Esto simula la relajación espontánea del filamento conductor en
    memristores difusivos (Ag/SiO₂, NbOx, VO₂) al retirar el campo eléctrico.

    Permite que el memristor en serie vuelva a R_OFF tras cada estímulo,
    habilitando operación como SENSOR repetible sin necesidad de borrado (RESET) externo.

    Parámetros:
        tau_relax  : Tiempo de relajación (s). Cuánto tarda x en volver a x_eq.
        x0_override: Si se especifica, usa este valor como x_eq (estado de equilibrio
                     en relajación) en lugar del electrical.initial_state.
    """

    def __init__(self, tau_relax: float = 0.05, x0_override: float = None):
        if tau_relax <= 0:
            raise ValueError(f"tau_relax debe ser positivo (recibido: {tau_relax})")
        self.tau_relax = tau_relax
        self.x0_override = x0_override

    def modify_dxdt(self, dxdt: float, state: float, voltage: float, current: float,
                    electrical: ElectricalConfig, model_config: Any) -> float:
        """
        Suma el término de relajación volátil al dx/dt existente.

        dx/dt_total = dx/dt_strukov + (-(x - x_eq) / τ_relax)
        """
        x_eq = self.x0_override if self.x0_override is not None else electrical.initial_state

        # Término de relajación: tira x hacia x_eq con constante de tiempo τ_relax
        dx_dt_relax = -(state - x_eq) / self.tau_relax

        return float(dxdt + dx_dt_relax)
