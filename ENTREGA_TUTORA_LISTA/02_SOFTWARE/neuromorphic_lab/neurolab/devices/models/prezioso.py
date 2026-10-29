"""
neurolab.devices.models.prezioso
==================================
Modelo Yakopcic (2011) para dispositivos de óxido metálico con
conmutación asimétrica. Reproduce la Fig 1b de Prezioso 2014.

Referencias:
  - C. Yakopcic et al., "A Memristor Device Model",
    IEEE Electron Device Letters 32(10), 1436-1438, 2011.
  - M. Prezioso et al., "Training and operation of an integrated
    neuromorphic network based on metal-oxide memristors",
    Nature Communications 5:4801, 2014.
"""
import numpy as np
from neurolab.core.base_device import BaseMathModel
from neurolab.core.config import ElectricalConfig, PreziosoConfig
from typing import Optional


class PreziosoMathModel(BaseMathModel):
    """
    Modelo de Yakopcic con conmutación SET/RESET asimétrica.

    Ecuación de estado:
        dx/dt = g(V) · f(x)

    donde:
        g(V) = A_p · (exp(V)   - exp(V_p))    si V >  V_p   (SET)
               -A_n · (exp(-V) - exp(V_n))    si V < -V_n   (RESET)
                0                              en otro caso

        f(x) = exp(-α_p · (x - x_p))          si x > x_p   (SET)
               exp(-α_n · ((1-x) - x_n))      si x < 1-x_n (RESET)
               1                              en otro caso

    La corriente se calcula como I = V/R(x) con R(x) lineal entre
    R_ON y R_OFF (heredado del modelo de Strukov).
    """

    def compute_dxdt(
        self,
        state: float,
        voltage: float,
        current: float,
        electrical: ElectricalConfig,
        model_config: Optional[PreziosoConfig] = None,
    ) -> float:
        cfg = model_config if isinstance(model_config, PreziosoConfig) else PreziosoConfig()

        # --- g(V): tasa dependiente de voltaje (asimétrica) ---
        V_0 = float(getattr(cfg, 'V_0', 1.0))
        if voltage > cfg.V_p:
            g = cfg.A_p * (np.exp(voltage / V_0) - np.exp(cfg.V_p / V_0))
        elif voltage < -cfg.V_n:
            g = -cfg.A_n * (np.exp(-voltage / V_0) - np.exp(cfg.V_n / V_0))
        else:
            g = 0.0

        # --- f(x): ventana asimétrica ---
        if voltage >= 0.0:  # SET
            if state > cfg.x_p:
                f = float(np.clip(np.exp(-cfg.alpha_p * (state - cfg.x_p)), 0.0, 1.0))
            else:
                f = 1.0
        else:  # RESET
            if state < 1.0 - cfg.x_n:
                f = float(np.clip(np.exp(-cfg.alpha_n * ((1.0 - state) - cfg.x_n)), 0.0, 1.0))
            else:
                f = 1.0

        return float(np.clip(g * f, -1000.0, 1000.0))

    def compute_resistance(
        self,
        state: float,
        electrical: ElectricalConfig,
        model_config: Optional[PreziosoConfig] = None,
        voltage: float = 0.0,
    ) -> float:
        cfg = model_config if isinstance(model_config, PreziosoConfig) else PreziosoConfig()
        r_on = cfg.r_on_reset if voltage < 0.0 else electrical.r_on
        return float(r_on * state + electrical.r_off * (1.0 - state))
