from typing import List, Optional
from neurolab.core.memristor import Memristor
from neurolab.devices.base import BaseRealismModifier
from neurolab.devices.presets.strukov_2008 import create_strukov_2008_fig2b_device

def create_strukov_paper_device(
    modifiers: Optional[List[BaseRealismModifier]] = None,
    initial_state: float = 0.1
) -> Memristor:
    """Alias de compatibilidad para create_strukov_2008_fig2b_device."""
    dev = create_strukov_2008_fig2b_device(modifiers=modifiers)
    dev.electrical.initial_state = initial_state
    dev.x = initial_state
    return dev
