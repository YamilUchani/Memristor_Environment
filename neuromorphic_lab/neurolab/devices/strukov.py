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
        modifiers = []
        enable_volatile = getattr(config, 'enable_volatile', kwargs.get('enable_volatile', kwargs.get('is_volatile', False)))
        if enable_volatile:
            from neurolab.devices.realism.volatile import VolatileDecayModifier
            tau_relax = float(getattr(config, 'tau_relax', kwargs.get('tau_relax', kwargs.get('volatile_tau_relax', 0.5))))
            x_eq = float(getattr(config, 'x_eq', kwargs.get('x_eq', 0.05)))
            modifiers.append(VolatileDecayModifier(tau_relax=tau_relax, x0_override=x_eq))

        super().__init__(
            math_model=StrukovMathModel(),
            electrical=elec,
            identity=identity,
            model_config=config,
            modifiers=modifiers,
            clip_x=clip_x
        )

__all__ = ["StrukovMathModel", "MemristorStrukov"]
