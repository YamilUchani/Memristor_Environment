"""
Modificadores de realismo y dinámica estocástica.
"""
from neurolab.devices.realism.window import BiolekWindowModifier, JoglekarWindowModifier
from neurolab.devices.realism.d2d import D2DVariabilityModifier
from neurolab.devices.realism.c2c import C2CVariabilityModifier
from neurolab.devices.realism.noise import ThermalNoiseModifier

__all__ = [
    "BiolekWindowModifier",
    "JoglekarWindowModifier",
    "D2DVariabilityModifier",
    "C2CVariabilityModifier",
    "ThermalNoiseModifier"
]
