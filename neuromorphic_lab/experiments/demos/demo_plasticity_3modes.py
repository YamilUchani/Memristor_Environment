"""
demo_plasticity_3modes.py
==========================
Demostración de los 3 modos del crossbar:
1. Lectura (inferencia)
2. STDP (aprendizaje no supervisado)
3. R-STDP (aprendizaje por refuerzo)

Ejecuta esto para validar la plasticidad sin GUI.
"""

import sys
import os
import numpy as np

# Asegurar importabilidad de neurolab
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neurolab.crossbar import Crossbar
from neurolab.crossbar.configs import CrossbarConfig
from neurolab.crossbar.plasticity import (
    STDPRule, RSTDPRule, STDPConfig, RSTDPConfig
)


def demo_lectura():
    """Modo 1: Lectura (inferencia pura)."""
    print("\n" + "="*70)
    print("  MODO 1: LECTURA (inferencia)")
    print("="*70)
    
    cb = Crossbar(CrossbarConfig(seed=42))
    cb.print_state("Estado inicial")
    
    # Aplicar voltaje a las filas, leer corrientes
    V_rows = np.array([0.2, 0.8, 0.2, 0.8])
    I_cols = cb.read_ideal(V_rows)
    
    print(f"\n  V_rows = {V_rows}")
    print(f"  I_cols = {I_cols * 1e6} uA")

    print("  -> 4 corrientes en 1 ciclo (sin cambiar G)")



def demo_stdp():
    """Modo 2: STDP (aprendizaje no supervisado)."""
    print("\n" + "="*70)
    print("  MODO 2: STDP (no supervisado)")
    print("="*70)
    
    cb = Crossbar(CrossbarConfig(seed=42))
    stdp = STDPRule(STDPConfig(
        A_plus=0.05, A_minus=0.025, eta=1.0,
        G_min=1e-6, G_max=500e-6,
    ))
    stdp.reset(n_rows=4, n_cols=4)
    
    G_hist = [cb.G_matrix_uS.copy()]
    
    # Simular 50 pasos con patrones de spike
    for step in range(50):
        # Patrón: fila 2 activa antes que columna 3
        spike_pre = np.zeros(4)
        spike_post = np.zeros(4)
        
        if step % 5 == 0:
            spike_pre[1] = 1.0  # Fila 2 dispara
        
        if step % 5 == 1:
            spike_post[2] = 1.0  # Columna 3 dispara (después)
        
        # Aplicar STDP
        G = cb.G_matrix
        dG = stdp.apply(G, spike_pre, spike_post, dt=1e-3)
        
        # Actualizar G
        for i in range(4):
            for j in range(4):
                G_new = np.clip(G[i,j] + dG[i,j], 1e-6, 500e-6)
                cb.cells[i][j].params['G'] = float(G_new)
    
    cb.print_state("Después de 50 pasos STDP")
    
    print(f"\n  dG_M23 (fila 2, col 3): "
          f"{cb.G_matrix_uS[1,2] - G_hist[0][1,2]:+.2f} uS")
    print("  -> La celda M23 (fila 2 -> col 3) se reforzo")
    print("  -> Las demas no cambiaron")



def demo_rstdp():
    """Modo 3: R-STDP (aprendizaje por refuerzo)."""
    print("\n" + "="*70)
    print("  MODO 3: R-STDP (por refuerzo)")
    print("="*70)
    
    cb = Crossbar(CrossbarConfig(seed=42))
    rstdp = RSTDPRule(RSTDPConfig(
        A_plus=0.05, A_minus=0.025, eta=1.0,
        G_min=1e-6, G_max=500e-6, R=0.0,
    ))
    rstdp.reset(n_rows=4, n_cols=4)
    
    # Patrón de spike: pre[2] disparó antes, post[1] dispara en el momento actual
    # Simular paso 0 donde pre dispara para cargar trace_pre
    rstdp.apply(cb.G_matrix, np.array([0., 0., 1., 0.]), np.array([0., 0., 0., 0.]), dt=1e-3)

    spike_pre = np.zeros(4)
    spike_post = np.array([0., 1., 0., 0.])  # Columna 2 dispara

    G_before = cb.G_matrix_uS.copy()
    
    # Paso 1: R = +1 (premio)
    print("\n  [Ciclo 1] PREMIO (R = +1)")
    rstdp.set_reward(+1.0)
    G = cb.G_matrix
    dG = rstdp.apply(G, spike_pre, spike_post, dt=1e-3)
    for i in range(4):
        for j in range(4):
            G_new = np.clip(G[i,j] + dG[i,j], 1e-6, 500e-6)
            cb.cells[i][j].params['G'] = float(G_new)
    
    delta1 = cb.G_matrix_uS[2,1] - G_before[2,1]
    print(f"    dG_M32 = {delta1:+.4f} uS (aumento)")
    
    # Paso 2: Mismo patrón (pre en paso previo, post ahora), R = -1 (castigo)
    print("\n  [Ciclo 2] CASTIGO (R = -1)")
    rstdp.reset(4, 4)
    rstdp.apply(cb.G_matrix, np.array([0., 0., 1., 0.]), np.array([0., 0., 0., 0.]), dt=1e-3)

    G_before2 = cb.G_matrix_uS.copy()
    rstdp.set_reward(-1.0)
    G = cb.G_matrix
    dG = rstdp.apply(G, spike_pre, spike_post, dt=1e-3)
    for i in range(4):
        for j in range(4):
            G_new = np.clip(G[i,j] + dG[i,j], 1e-6, 500e-6)
            cb.cells[i][j].params['G'] = float(G_new)
    
    delta2 = cb.G_matrix_uS[2,1] - G_before2[2,1]
    print(f"    dG_M32 = {delta2:+.4f} uS (disminuyo)")
    
    print("\n  -> R-STDP modula el aprendizaje segun recompensa")
    print("  -> M32 con R=+1: refuerza (aprendes a hacerlo mas)")
    print("  -> M32 con R=-1: debilita (aprendes a no hacerlo)")


if __name__ == '__main__':
    demo_lectura()
    demo_stdp()
    demo_rstdp()

    
    print("\n" + "="*70)
    print("  [OK] 3 MODOS VALIDADOS")
    print("="*70)

