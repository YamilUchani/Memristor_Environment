"""
Módulo core de neurolab.
Contiene la abstracción universal del Memristor, estructuras de configuración y clases base.
"""

from neurolab.core.config import DeviceIdentity, ElectricalConfig, StrukovConfig
from neurolab.core.base_device import BaseMathModel, BaseRealismModifier
from neurolab.core.memristor import Memristor

__all__ = [
    "DeviceIdentity",
    "ElectricalConfig",
    "StrukovConfig",
    "BaseMathModel",
    "BaseRealismModifier",
    "Memristor",
]
