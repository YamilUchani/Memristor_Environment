"""
neurolab.configs
================
Paquete de alias de configuraciones para compatibilidad y simplicidad en importaciones.
"""
from neurolab.core.config import (
    DeviceIdentity,
    ElectricalConfig,
    StrukovConfig,
    PreziosoConfig,
    PreziosoVirginConfig as YakopcicVirginConfig,
)

__all__ = [
    "DeviceIdentity",
    "ElectricalConfig",
    "StrukovConfig",
    "PreziosoConfig",
    "YakopcicVirginConfig",
]
