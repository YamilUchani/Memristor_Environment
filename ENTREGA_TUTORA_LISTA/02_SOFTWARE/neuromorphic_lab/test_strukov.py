import numpy as np
import matplotlib.pyplot as plt

def test_loop(mu, RON, ROFF, V0, freq, x0):
    t = np.linspace(0, 1.0/freq, 2000)
    dt = t[1] - t[0]
    V = V0 * np.sin(2 * np.pi * freq * t)
    x = x0
    I = np.zeros_like(V)
    D = 10e-9
    for k, v in enumerate(V):
        R = RON * x + ROFF * (1 - x)
        i_curr = v / R
        I[k] = i_curr
        
        # Joglekar window with p=1
        w = 1 - (2*x - 1)**2
        dx = (mu * RON / D**2) * i_curr * w * dt
        x += dx
        x = np.clip(x, 0, 1)
        
    plt.figure()
    plt.plot(V, I*1e3)
    plt.title(f"mu={mu:.1e}, RON={RON}, ROFF={ROFF}, V0={V0}, f={freq}, x0={x0}")
    plt.savefig(f"loop_{mu:.1e}_{RON}_{ROFF}_{V0}.png")
    plt.close()
    
    print(f"mu={mu:.1e} RON={RON} ROFF={ROFF} | Imax={I.max()*1e3:.2f} Imin={I.min()*1e3:.2f} ratio={abs(I.max()/I.min()):.2f}")

test_loop(1e-14, 100, 16000, 1.0, 1.0, 0.1)
test_loop(1e-14, 100, 16000, 1.0, 1.0, 0.5)
test_loop(1e-14, 100, 16000, 2.0, 1.0, 0.5)
test_loop(5e-14, 100, 16000, 1.0, 1.0, 0.5)
test_loop(1e-13, 100, 16000, 1.0, 1.0, 0.5)
test_loop(1e-14, 1000, 16000, 1.0, 1.0, 0.5)
test_loop(5e-14, 1000, 16000, 1.0, 1.0, 0.5)
test_loop(1e-13, 1000, 16000, 1.0, 1.0, 0.5)
test_loop(5e-14, 100, 16000, 0.5, 1.0, 0.5)
