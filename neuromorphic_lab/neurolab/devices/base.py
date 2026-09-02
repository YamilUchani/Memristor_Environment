"""
neurolab.devices.base
=====================
Reexporta las clases base abstractas desde `core.base_device` para mantener
compatibilidad de imports en todo el paquete `devices`.

Fuente de verdad: neurolab/core/base_device.py
"""
from neurolab.core.base_device import BaseMathModel, BaseRealismModifier  # noqa: F401

# Alias de compatibilidad legacy
BaseDevice = None  # No utilizado en la arquitectura actual

__all__ = ["BaseMathModel", "BaseRealismModifier"]
