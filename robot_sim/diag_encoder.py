import numpy as np
from neurobot.encoder import SensorEncoder

enc = SensorEncoder(obs_dim=2, N=8, V_read=0.1, pre_topk=0.4, seed=0)

for s in (0, 1):
    obs = np.array([1.0, float(s)])
    V = enc.encode(obs)
    pre = enc.last_pre_spikes()
    print(f"\n--- s = {s} ---")
    print(f"obs          = {obs}")
    print(f"V_rows       = {V.round(4)}")
    print(f"pre          = {pre.astype(int)}")
    print(f"sum(pre)     = {int(pre.sum())}")

# Diferencia entre contextos
pre0 = enc.encode(np.array([1.0, 0.0]))
pre0_mask = enc.last_pre_spikes()
pre1 = enc.encode(np.array([1.0, 1.0]))
pre1_mask = enc.last_pre_spikes()

print("\n--- Comparación ---")
print("pre0      :", pre0_mask.astype(int))
print("pre1      :", pre1_mask.astype(int))
print("XOR       :", (pre0_mask ^ pre1_mask).astype(int), " (esperado: ≥1)")
print("sum(pre0) :", int(pre0_mask.sum()))
print("sum(pre1) :", int(pre1_mask.sum()))
