"""
neurolab.devices.strukov
=========================
Exporta StrukovMathModel y MemristorStrukov.
"""
from typing import Optional
from neurolab.core.memristor import Memristor
from neurolab.core.config import DeviceIdentity, ElectricalConfig, StrukovConfig
from neurolab.devices.models.strukov import StrukovMathModel

class MemristorStrukov(Memristor):
    """
    Abstracción de dispositivo MemristorStrukov para fácil instanciación.
    """
    def __init__(self, config: Optional[StrukovConfig] = None, **kwargs):
        if config is None:
            config = StrukovConfig()
        r_on = getattr(config, 'RON', None)
        if r_on is None:
            r_on = getattr(config, 'r_on', 100.0)
        r_off = getattr(config, 'ROFF', None)
        if r_off is None:
            r_off = getattr(config, 'r_off', 16_000.0)
        x0 = getattr(config, 'x0', None)
        if x0 is None:
            x0 = getattr(config, 'initial_state', 0.10)
        
        elec = ElectricalConfig(
            r_on=r_on,
            r_off=r_off,
            initial_state=x0
        )
        identity = DeviceIdentity(
            device_name="Strukov TiO2",
            device_family="oxide_memristor",
            model_name="strukov"
        )
        clip_x = getattr(config, 'clip_x', True)
        super().__init__(
            math_model=StrukovMathModel(),
            electrical=elec,
            identity=identity,
            model_config=config,
            clip_x=clip_x
        )

__all__ = ["StrukovMathModel", "MemristorStrukov"]
