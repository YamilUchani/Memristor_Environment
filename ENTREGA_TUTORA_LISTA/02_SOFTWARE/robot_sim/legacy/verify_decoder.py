import numpy as np
from neurobot.decoder import ActionDecoder

dec = ActionDecoder(n_actions=2, M=4)

# caso 1: ganador col=1
print(dec.decode(winners=[1], I_col=[0.1, 0.5, 0.3, 0.2]))  # -> 1

# caso 2: ganadores múltiples, gana col=3
print(dec.decode(winners=[0, 3], I_col=[0.1, 0.2, 0.3, 0.9]))  # -> 1

# caso 3: sin ganadores, fallback argmax
print(dec.decode(winners=[], I_col=[0.1, 0.2, 0.9, 0.3]))  # -> 0 (col 2 % 2)
