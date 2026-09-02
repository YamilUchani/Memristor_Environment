import pytest
import numpy as np
from neurolab.devices.presets import (
    create_strukov_paper_device,
    create_strukov_2008_fig2b_device,
    create_strukov_normalized_preset,
    create_strukov_stochastic_preset
)
from neurolab.devices.realism import (
    BiolekWindowModifier,
    C2CVariabilityModifier,
    ThermalNoiseModifier,
)

def test_strukov_paper_preset_initial_state():
    """Verifica que el preset de la Fig 2b de Strukov se inicialice con los valores correctos."""
    mem = create_strukov_paper_device(initial_state=0.1)
    
    assert mem.identity.device_name == "Strukov TiO2 (Paper Fig 2b)"
    assert mem.electrical.r_on == 100.0
    assert mem.electrical.r_off == 16_000.0
    assert mem.x == 0.1
    
    # R(0.1) = 100 * 0.1 + 16000 * 0.9 = 10 + 14400 = 14410 Ohm
    expected_resistance = 14410.0
    assert pytest.approx(mem.resistance, rel=1e-5) == expected_resistance
    assert pytest.approx(mem.conductance, rel=1e-5) == 1.0 / expected_resistance

def test_strukov_official_presets():
    """Verifica la instanciación de los 3 perfiles oficiales."""
    dev_ideal = create_strukov_2008_fig2b_device()
    assert dev_ideal.identity.name == "Strukov 2008 - Figure 2b (Ideal)"
    assert len(dev_ideal.modifiers) == 0

    dev_norm = create_strukov_normalized_preset(p=5)
    assert dev_norm.identity.name == "Strukov TiO2 (Normalizado - Ventana Biolek)"
    assert len(dev_norm.modifiers) == 1

    dev_stoch = create_strukov_stochastic_preset(seed=42)
    assert dev_stoch.identity.name == "Strukov TiO2 (Estocástico Realista)"
    assert len(dev_stoch.modifiers) == 4

def test_stochastic_reproducibility():
    """Verifica que dos simulaciones con la misma semilla seed=42 entreguen vectores idénticos."""
    dev1 = create_strukov_stochastic_preset(seed=42)
    dev2 = create_strukov_stochastic_preset(seed=42)

    v0, f0, dt, steps = 1.0, 0.5, 0.0001, 1000
    i_1, i_2 = [], []

    for idx in range(steps):
        t = idx * dt
        v = v0 * np.sin(2.0 * np.pi * f0 * t)
        i_1.append(dev1.step(voltage=v, dt=dt))
        i_2.append(dev2.step(voltage=v, dt=dt))

    # Reproducibilidad científica estricta
    np.testing.assert_array_equal(np.array(i_1), np.array(i_2))

def test_strukov_modo_1_ideal_simulation():
    """Modo 1: Strukov puro sin modificadores."""
    mem = create_strukov_paper_device(initial_state=0.1)
    
    v0 = 1.0
    f0 = 0.5
    dt = 0.0001
    total_time = 4.0 / f0  # 4 ciclos completos
    steps = int(total_time / dt)
    
    states = []
    currents = []
    voltages = []
    
    for i in range(steps):
        t = i * dt
        v = v0 * np.sin(2.0 * np.pi * f0 * t)
        i_out = mem.step(voltage=v, dt=dt)
        
        states.append(mem.x)
        currents.append(i_out)
        voltages.append(v)
    
    # 1. Verificar acotamiento de estado
    assert all(0.0 <= x <= 1.0 for x in states)
    
    # 2. Propiedad fundamental: "Pinched Hysteresis Loop" (I = 0 cuando V = 0)
    for v, i_out in zip(voltages, currents):
        if abs(v) < 1e-9:
            assert abs(i_out) < 1e-9

def test_strukov_modo_2_biolek_window():
    """Modo 2: Modelo con ventana no lineal de Biolek."""
    mem = create_strukov_paper_device(
        modifiers=[BiolekWindowModifier(p=5)],
        initial_state=0.1
    )
    
    v0 = 1.5
    f0 = 0.5
    dt = 0.0001
    steps = 1000
    
    states = []
    for i in range(steps):
        t = i * dt
        v = v0 * np.sin(2.0 * np.pi * f0 * t)
        mem.step(voltage=v, dt=dt)
        states.append(mem.x)
        
    assert all(0.0 <= x <= 1.0 for x in states)

def test_strukov_modo_3_realista_estocastico():
    """Modo 3: Dispositivo estocástico con Biolek, C2C y ruido térmico."""
    mem = create_strukov_paper_device(
        modifiers=[
            BiolekWindowModifier(p=5),
            C2CVariabilityModifier(relative_std=0.05, seed=42),
            ThermalNoiseModifier(noise_std=1e-6, seed=42)
        ],
        initial_state=0.1
    )
    
    v0 = 1.0
    f0 = 0.5
    dt = 0.0001
    steps = 1000
    
    currents = []
    for i in range(steps):
        t = i * dt
        v = v0 * np.sin(2.0 * np.pi * f0 * t)
        i_out = mem.step(voltage=v, dt=dt)
        currents.append(i_out)
        
    assert len(currents) == steps
    assert all(0.0 <= mem.x <= 1.0 for _ in [0])
