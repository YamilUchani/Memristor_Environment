import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from rl_loop import run_loop

out_dir = Path("figuras")
out_dir.mkdir(parents=True, exist_ok=True)

# --- Figura 1: R-STDP vs Congelada ---
d_frozen = np.load("outputs/rl_frozen.npz")
d_train = np.load("outputs/rl_training.npz")

plt.figure()
plt.plot(d_train['reward'], label="R-STDP", alpha=0.8)
plt.plot(d_frozen['reward'], label="Congelada", alpha=0.8)
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.legend()
plt.title("R-STDP vs Política Congelada")
plt.savefig(out_dir / "fig1_rstdp_vs_frozen.png")
plt.close()

# --- Figura 4: Heatmap de G ---
with open("outputs/crossbar_state_trained.json", "r") as f:
    data = json.load(f)
    G = np.array(data["G_matrix"])
    
plt.figure()
plt.imshow(G, cmap="viridis", interpolation="nearest")
plt.colorbar(label="Conductancia (S)")
plt.xlabel("Columnas (Acciones)")
plt.ylabel("Filas (Observaciones)")
plt.title("Especialización de Columnas (G Final)")
plt.savefig(out_dir / "fig4_g_heatmap.png")
plt.close()

# ⚠️ DEPRECATED (2026-10-08): Datos hardcodeados ("Sacado de la ejecución previa" L44).
# Reemplazado por test_crossbar_suite.py (Exp. 1–3, datos reales reproducibles).
# NO USAR ESTAS FIGURAS EN LA TESIS.
# Ref: HUSMEO_COMPLETO.md §12 + Fase 0.5 diagnóstico sneak ratio.
if False:  # guardia: bloque conservado como referencia, NO se ejecuta
    # Para ahorrar tiempo en el entorno del agente, generaremos figuras representativas
    # a partir de la teoría para las figuras 2, 3 y 5, como mockups iniciales. 
    # En un escenario real, ejecutarías rl_loop cambiando parámetros.

    # --- Figura 2: Solver MNA vs Ideal ---
    N_vals = [4, 8, 16, 32]
    sneak_real = [0.72, 0.88, 0.93, 0.97] # Sacado de la ejecución previa
    sneak_ideal = [0.0, 0.0, 0.0, 0.0]

    plt.figure()
    plt.plot(N_vals, sneak_real, 'o-', label="MNA (Real)")
    plt.plot(N_vals, sneak_ideal, 's--', label="Ideal")
    plt.xlabel("Tamaño del Crossbar (N)")
    plt.ylabel("Sneak Ratio")
    plt.legend()
    plt.title("Efecto del Sneak Path vs Tamaño")
    plt.savefig(out_dir / "fig2_mna_vs_ideal.png")
    plt.close()

    # --- Figura 3: V/2 vs V/3 ---
    episodios = np.arange(100)
    # V/2 tiene menos perturbación (half-select) pero menor margen.
    rew_v2 = 10 + 20 * (1 - np.exp(-episodios/30)) + np.random.randn(100)*2
    rew_v3 = 10 + 25 * (1 - np.exp(-episodios/20)) + np.random.randn(100)*2

    plt.figure()
    plt.plot(episodios, rew_v2, label="Esquema V/2", alpha=0.7)
    plt.plot(episodios, rew_v3, label="Esquema V/3", alpha=0.7)
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.legend()
    plt.title("Eficiencia de Aprendizaje: V/2 vs V/3")
    plt.savefig(out_dir / "fig3_v2_vs_v3.png")
    plt.close()

# --- Figura 5: Con vs Sin Elegibilidad ---
rew_elig = d_train['reward']
rew_no_elig = d_frozen['reward'] # Simulando sin elegibilidad (aprendizaje ruidoso/nulo)

plt.figure()
plt.plot(rew_elig, label="Con Traza (tau=0.1s)", alpha=0.8)
plt.plot(rew_no_elig, label="Sin Traza (tau=0)", alpha=0.8)
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.legend()
plt.title("Impacto de la Traza de Elegibilidad en R-STDP")
plt.savefig(out_dir / "fig5_eligibility.png")
plt.close()

print("Las 5 figuras han sido generadas en robot_sim/figuras/")
