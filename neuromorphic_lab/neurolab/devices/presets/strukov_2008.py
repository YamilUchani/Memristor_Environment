from neurolab.core.memristor import Memristor
from neurolab.devices.config import DeviceConfig, ElectricalConfig, StrukovConfig
from neurolab.devices.models.strukov import StrukovMathModel
from neurolab.devices.realism import (
    BiolekWindowModifier, D2DVariabilityModifier,
    C2CVariabilityModifier, ThermalNoiseModifier
)
from typing import List, Optional, Any

def create_strukov_2008_fig2b_device(modifiers: List[Any] = None) -> Memristor:
    """
    Preset 1: STRUKOV_2008_IDEAL
    Perfil científico reproducible puro (sin modificadores por defecto).
    """
    identity = DeviceConfig(
        name="Strukov 2008 - Figure 2b (Ideal)",
        family="oxide_memristor",
        model="strukov"
    )

    electrical = ElectricalConfig(
        r_on=100.0,
        r_off=16_000.0,
        initial_state=0.1
    )

    strukov_config = StrukovConfig(
        D=10e-9,
        mu_v=1e-14
    )

    return Memristor(
        math_model=StrukovMathModel(),
        electrical=electrical,
        identity=identity,
        model_config=strukov_config,
        modifiers=modifiers or []
    )

def create_strukov_normalized_preset(p: int = 5) -> Memristor:
    """
    Preset 2: STRUKOV_NORMALIZED
    Perfil extendido determinista no lineal con ventana de Biolek activa.
    """
    dev = create_strukov_2008_fig2b_device(modifiers=[BiolekWindowModifier(p=p)])
    dev.identity.name = "Strukov TiO2 (Normalizado - Ventana Biolek)"
    return dev

def create_strukov_stochastic_preset(seed: int = 42) -> Memristor:
    """
    Preset 3: STRUKOV_STOCHASTIC
    Perfil realista completo con D2D, C2C (Ornstein-Uhlenbeck) y Ruido Térmico (seed = 42).
    """
    modifiers = [
        BiolekWindowModifier(p=5),
        D2DVariabilityModifier(variability_std=0.05, seed=seed),
        C2CVariabilityModifier(sigma=0.05, theta=1.0, seed=seed),
        ThermalNoiseModifier(noise_std=1e-6, seed=seed)
    ]
    dev = create_strukov_2008_fig2b_device(modifiers=modifiers)
    dev.identity.name = "Strukov TiO2 (Estocástico Realista)"
    return dev

def create_strukov_paper_device(
    modifiers: Optional[List[Any]] = None,
    initial_state: float = 0.1
) -> Memristor:
    """Alias de compatibilidad para la suite de pruebas."""
    dev = create_strukov_2008_fig2b_device(modifiers=modifiers)
    dev.identity.name = "Strukov TiO2 (Paper Fig 2b)"
    dev.electrical.initial_state = initial_state
    dev.x = initial_state
    return dev
