import numpy as np
from neurobot.crossbar_brain import CrossbarBrain

# 1) Instanciar
brain = CrossbarBrain(N=8, M=4, seed=42)
print(f"[1] G_min={brain.G_min:.3e}  G_max={brain.G_max:.3e}")
print(f"    G initial: mean={brain.G.mean():.3e}  std={brain.G.std():.3e}")

# 2) Lectura con V uniforme
V = np.full(8, 0.1)
I_col, sneak = brain.read(V)
print(f"\n[2] I_col={I_col}")
print(f"    sneak_ratio={sneak:.4f}  (esperado: pequeño con G casi uniforme)")

# 3) Paso con LIF + WTA
brain.reset()
for _ in range(200):
    spikes, I_col, winners, sneak = brain.step(V, dt=1e-4)

rate_hz = sum(1 for s in brain.history["spikes"] if s.any()) / 200.0 / 1e-4
print(f"\n[3] Tras 200 pasos:")
print(f"    tasa disparo total={rate_hz:.1f} Hz")
print(f"    winners últimos: {winners}")

# 4) Sneak vs N (esto es figura de tesis)
print("\n[4] sneak_ratio vs N (G heterogénea):")
for N in (4, 8, 16, 32):
    b = CrossbarBrain(N=N, M=4, seed=0)
    # Hacer G muy heterogénea para exagerar sneak
    b.X = np.random.default_rng(1).uniform(0.05, 0.95, (N, 4))
    b.read(np.full(N, 0.1))
    print(f"    N={N:3d}  sneak={b.history['sneak_ratio'][-1]:.4f}")

# 5) Guardar estado
brain.save_state("crossbar_state_test.json")
print("\n[5] Estado guardado en robot_sim/outputs/crossbar_state_test.json")
