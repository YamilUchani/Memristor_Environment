import sys
from pathlib import Path

# Add neuromorphic_lab to sys.path
lab_dir = Path(r"g:\Github\Software de simulacion\Memristor_Environment\neuromorphic_lab")
sys.path.insert(0, str(lab_dir))

import numpy as np
from neurolab.core.memristor import Memristor
from neurolab.core.config import ElectricalConfig, StrukovConfig, DeviceIdentity
from neurolab.devices.models.strukov import StrukovMathModel
from neurolab.devices.realism.volatile import VolatileDecayModifier
from neurolab.neurons.lif import LIFNeuron, LIFConfig

def test_volatile_hybrid():
    # 1. Non-volatile memristor
    mem_non_volatile = Memristor(
        math_model=StrukovMathModel(),
        electrical=ElectricalConfig(r_on=1000.0, r_off=500000.0, initial_state=0.05),
        identity=DeviceIdentity(device_name="Strukov Non-Volatile"),
        model_config=StrukovConfig(D=10e-9, mu_v=1e-14),
        modifiers=[]
    )

    # 2. Volatile memristor (with VolatileDecayModifier)
    mem_volatile = Memristor(
        math_model=StrukovMathModel(),
        electrical=ElectricalConfig(r_on=1000.0, r_off=500000.0, initial_state=0.05),
        identity=DeviceIdentity(device_name="HfO2 Volatile"),
        model_config=StrukovConfig(D=10e-9, mu_v=1e-14),
        modifiers=[VolatileDecayModifier(tau_relax=0.05, x0_override=0.05)]
    )


    dt = 1e-4
    # Apply 5.0 V pulse for 0.1 s, then 0.0 V for 0.2 s
    t_pulse = np.arange(0, 0.1, dt)
    t_silence = np.arange(0.1, 0.3, dt)

    print("--- NON-VOLATILE MEMRISTOR ---")
    r_start = mem_non_volatile.resistance
    for t in t_pulse:
        mem_non_volatile.step(5.0, dt)
    r_after_pulse = mem_non_volatile.resistance
    for t in t_silence:
        mem_non_volatile.step(0.0, dt)
    r_after_silence = mem_non_volatile.resistance
    print(f"R_start: {r_start/1e3:.2f} kOhm -> After pulse: {r_after_pulse/1e3:.2f} kOhm -> After silence: {r_after_silence/1e3:.2f} kOhm")

    print("\n--- VOLATILE MEMRISTOR (VolatileDecayModifier) ---")
    r_start_v = mem_volatile.resistance
    for t in t_pulse:
        mem_volatile.step(5.0, dt)
    r_after_pulse_v = mem_volatile.resistance
    for t in t_silence:
        mem_volatile.step(0.0, dt)
    r_after_silence_v = mem_volatile.resistance
    print(f"R_start: {r_start_v/1e3:.2f} kOhm -> After pulse: {r_after_pulse_v/1e3:.2f} kOhm -> After silence: {r_after_silence_v/1e3:.2f} kOhm")

if __name__ == "__main__":
    test_volatile_hybrid()
