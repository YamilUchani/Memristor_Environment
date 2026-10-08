import numpy as np
from neurobot.encoder import SensorEncoder

enc = SensorEncoder(obs_dim=4, N=8, V_read=0.1, pre_threshold=0.3, seed=1)

for _ in range(3):
    obs = np.random.randn(4)
    V = enc.encode(obs)
    pre = enc.last_pre_spikes()
    print(f"obs={obs.round(3)}  V={V.round(4)}  pre={pre.astype(int)}")
