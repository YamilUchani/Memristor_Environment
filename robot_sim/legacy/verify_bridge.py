from neurobot.lab_bridge import (
    get_electrical_config, get_device_config,
    make_memristor, get_neuron_config, make_lif, solve_mna,
)
import numpy as np

# 1) ElectricalConfig
elec = get_electrical_config()
print("[1] ElectricalConfig:", elec)

# 2) StrukovConfig
cfg = get_device_config("strukov")
print("[2] StrukovConfig:", cfg)

# 3) Memristor instanciado
m = make_memristor()
print("[3] Memristor ok:", type(m).__name__)

nc = get_neuron_config()
print(f"[4] LIFConfig: c_m={nc.c_m:.2e} F  r_leak={nc.r_leak:.2e} Ohm  r_series={nc.r_series:.2e} Ohm")

# 5) LIF instanciada
n = make_lif()
spiked = n.step(current_input=1e-6, dt=1e-4)
print(f"[5] LIFNeuron step -> spiked={spiked}, v_m={getattr(n, 'v_membrane', '?')}")

# 6) Solver MNA con G trivial
G = np.full((4, 4), 1e-4)   # 100 µS por celda
V = np.array([0.1, 0.1, 0.0, 0.0])
I = solve_mna(G, V)
print(f"[6] solve_mna -> I_col={I}  shape={I.shape}")
