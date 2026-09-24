"""
neurolab.devices.yakopcic
=========================
Exporta MemristorYakopcic y YakopcicVirginConfig.
"""
import numpy as np
from typing import Optional
from neurolab.core.memristor import Memristor
from neurolab.core.config import DeviceIdentity, ElectricalConfig, PreziosoVirginConfig, PreziosoConfig
from neurolab.devices.models.prezioso import PreziosoMathModel

YakopcicVirginConfig = PreziosoVirginConfig
YakopcicConfig = PreziosoConfig


class MemristorYakopcic(Memristor):
    """
    Abstracción de dispositivo MemristorYakopcic (Prezioso 2014) para el módulo Crossbar.
    """
    def __init__(self, config: Optional[PreziosoVirginConfig] = None, G_0: Optional[float] = None, **kwargs):
        if config is None:
            config = PreziosoVirginConfig()
        if G_0 is not None:
            config.G_initial = G_0

        r_on = getattr(config, 'R_on_reset', 1750.0)
        r_off = getattr(config, 'R_off', 1.0e6)

        if G_0 is not None and G_0 > 1e-9:
            r_init = 1.0 / G_0
            x0 = (r_init - r_off) / (r_on - r_off) if r_off != r_on else 0.02
            x0 = float(np.clip(x0, 0.01, 0.99))
        else:
            x0 = float(np.clip(getattr(config, 'x0', 0.02), 0.01, 0.99))

        elec = ElectricalConfig(
            r_on=r_on,
            r_off=r_off,
            initial_state=x0
        )
        identity = DeviceIdentity(
            device_name="Yakopcic Prezioso 2014",
            device_family="oxide_memristor",
            model_name="prezioso"
        )
        modifiers = kwargs.get('modifiers', [])
        super().__init__(
            math_model=PreziosoMathModel(),
            electrical=elec,
            identity=identity,
            model_config=config,
            modifiers=modifiers,
            clip_x=True
        )

    def update(self, V_applied: float, dt: float) -> None:
        super().update(voltage=V_applied, dt=dt)


__all__ = ["MemristorYakopcic", "YakopcicVirginConfig", "YakopcicConfig"]
