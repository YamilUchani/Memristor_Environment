"""
Determina empíricamente las unidades de t_ref y la sensibilidad de la LIF
a corriente realista del crossbar.
"""
import numpy as np
from neurobot import lab_bridge
from neurolab.neurons.config import LIFConfig
from neurolab.neurons.lif import LIFNeuron

# --- config base (desde el JSON del lab) ---
base = lab_bridge.get_neuron_config()
print(f"config cruda del lab: t_ref={base.t_ref}  c_m={base.c_m:.2e}  r_leak={base.r_leak:.2e}")
print(f"                      v_th={base.v_th}  v_rest={base.v_rest}  v_reset={base.v_reset}")

def count_spikes(t_ref: float, I_in: float, dt: float, n_steps: int = 500):
    cfg = LIFConfig(
        c_m=base.c_m, r_leak=base.r_leak, r_series=base.r_series,
        v_rest=base.v_rest, v_th=base.v_th, v_reset=base.v_reset,
        t_ref=t_ref,
    )
    n = LIFNeuron(config=cfg)
    spikes = 0
    for _ in range(n_steps):
        if n.step(current_input=I_in, dt=dt):
            spikes += 1
    return spikes

dt = 1e-4
I_realista = 1e-3   # ~1 mA, lo que vimos en verify_brain

print("\n--- t_ref con corriente realista (1 mA) ---")
for t_ref in [0.0, 2e-3, 2e-2, 0.2, 2.0]:
    s = count_spikes(t_ref, I_realista, dt, n_steps=500)
    print(f"  t_ref={t_ref:>8}  ->  {s:4d} spikes / 500 pasos   ({s/500/dt/1000:.1f} kHz)")

print("\n--- corrientes para t_ref=0 ---")
for I in [1e-6, 1e-5, 1e-4, 1e-3]:
    s = count_spikes(0.0, I, dt, n_steps=500)
    print(f"  I={I:.1e} A  ->  {s:4d} spikes / 500 pasos  ({s/500/dt:.1f} Hz)")
