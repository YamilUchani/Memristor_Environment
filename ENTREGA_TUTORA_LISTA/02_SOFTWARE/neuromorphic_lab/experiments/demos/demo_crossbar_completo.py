"""
demo_crossbar_completo.py
==========================
Demostración de TODOS los componentes funcionales del módulo neurolab.crossbar.
"""

import numpy as np
import matplotlib.pyplot as plt

from neurolab.crossbar import (
    Crossbar, CrossbarConfig,
    validate_read_operation,
    validate_programming_selectivity,
    validate_line_resistance,
    validate_scalability,
)
from neurolab.crossbar.configs import ProgrammingMode
from neurolab.crossbar.programming import classify_cells


def demo_basic():
    """Demo 1: Crossbar básico con D2D."""
    print("\n" + "="*70)
    print("  DEMO 1: CROSSBAR BÁSICO CON D2D")
    print("="*70)
    
    cb = Crossbar(CrossbarConfig(n_rows=4, n_cols=4, seed=42))
    cb.print_state("ESTADO INICIAL")
    cb.print_summary()
    
    # Lectura
    V_rows = np.array([0.1, 0.2, 0.1, 0.2])
    I_cols = cb.read(V_rows)
    
    print(f"\n  V_rows = {V_rows}")
    print(f"  I_cols = {I_cols * 1e6} uA")
    
    return cb


def demo_programming_modes():
    """Demo 2: Los 3 modos de programación."""
    print("\n\n" + "="*70)
    print("  DEMO 2: MODOS DE PROGRAMACIÓN")
    print("="*70)
    
    for mode in ['1T1R', 'V2', 'V3']:
        cb = Crossbar(CrossbarConfig(seed=42))
        G_before = cb.G_matrix.copy()
        
        if mode == '1T1R':
            cb.program_1T1R(1, 1, V_program=2.0, dt=1e-3)
        elif mode == 'V2':
            cb.program_V2(1, 1, V_program=2.0, dt=1e-3)
        elif mode == 'V3':
            cb.program_V3(1, 1, V_program=2.0, dt=1e-3)
        
        delta = (cb.G_matrix - G_before) * 1e6
        
        print(f"\n  Modo: {mode}")
        print("  dG (uS):")
        for i in range(4):
            row_str = f"    "
            for j in range(4):
                row_str += f"{delta[i,j]:+7.3f} "
            print(row_str)
        
        target = delta[1, 1]
        print(f"  dG objetivo: {target:+.3f} uS")


def demo_visualization():
    """Demo 3: Visualización con heatmap."""
    print("\n\n" + "="*70)
    print("  DEMO 3: VISUALIZACIÓN")
    print("="*70)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # --- Antes ---
    cb = Crossbar(CrossbarConfig(seed=42))
    G_before = cb.G_matrix_uS
    
    im0 = axes[0].imshow(G_before, cmap='YlOrRd')
    axes[0].set_title('Antes (D2D)')
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f'{G_before[i,j]:.1f}',
                         ha='center', va='center', fontsize=9)
    plt.colorbar(im0, ax=axes[0])
    
    # --- Programar V/2 ---
    cb.program_V2(1, 1, V_program=2.0, dt=1e-3)
    G_after = cb.G_matrix_uS
    
    im1 = axes[1].imshow(G_after, cmap='YlOrRd')
    axes[1].set_title('Después de V/2 en (1,1)')
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, f'{G_after[i,j]:.1f}',
                         ha='center', va='center', fontsize=9)
    plt.colorbar(im1, ax=axes[1])
    
    # --- Delta ---
    delta = G_after - G_before
    im2 = axes[2].imshow(delta, cmap='RdBu_r')
    axes[2].set_title('dG (uS) - Cruz de programación')
    for i in range(4):
        for j in range(4):
            axes[2].text(j, i, f'{delta[i,j]:+.1f}',
                         ha='center', va='center', fontsize=9)
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('crossbar_demo.png', dpi=150)
    plt.close()
    print("\n  [OK] Figura guardada: crossbar_demo.png")


def demo_ltp_ltd():
    """Demo 4: Pulsos LTP/LTD secuenciales."""
    print("\n\n" + "="*70)
    print("  DEMO 4: PULSOS LTP/LTD")
    print("="*70)
    
    cb = Crossbar(CrossbarConfig(seed=42))
    G_history = [cb.G_matrix_uS[1, 1]]
    
    print("\n  Aplicando 20 pulsos LTP (+2V) en M22...")
    for k in range(20):
        cb.program_V2(1, 1, V_program=+2.0, dt=1e-3)
        G_history.append(cb.G_matrix_uS[1, 1])
    
    print(f"  G después LTP: {G_history[-1]:.2f} uS")
    
    print("\n  Aplicando 20 pulsos LTD (-2V) en M22...")
    for k in range(20):
        cb.program_V2(1, 1, V_program=-2.0, dt=1e-3)
        G_history.append(cb.G_matrix_uS[1, 1])
    
    print(f"  G después LTD: {G_history[-1]:.2f} uS")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(G_history, 'b-o', ms=4, lw=1.5)
    ax.axvline(20, color='r', ls='--', label='Cambio LTP->LTD')
    ax.set_xlabel('Número de pulso')
    ax.set_ylabel('Conductancia (uS)')
    ax.set_title('Evolución de G en M22 con pulsos LTP/LTD')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig('crossbar_ltp_ltd.png', dpi=150)
    plt.close()
    print("\n  [OK] Figura guardada: crossbar_ltp_ltd.png")


def demo_validation():
    """Demo 5: Todas las validaciones."""
    print("\n\n" + "="*70)
    print("  DEMO 5: VALIDACIONES")
    print("="*70)
    
    cb = Crossbar(CrossbarConfig(seed=42))
    
    validate_read_operation(cb, n_tests=100)
    
    for mode in ['1T1R', 'V2', 'V3']:
        cb.reset()
        validate_programming_selectivity(cb, mode=mode)
    
    V_rows = np.array([0.1, 0.2, 0.1, 0.2])
    validate_line_resistance(
        cb, V_rows,
        R_H_values=[0.0, 0.1, 1.0, 5.0, 10.0, 50.0],
    )
    
    validate_scalability()


def main():
    """Ejecuta todas las demos."""
    print("\n" + "="*70)
    print("  MÓDULO CROSSBAR - DEMOSTRACIÓN COMPLETA")
    print("="*70)
    
    demo_basic()
    demo_programming_modes()
    demo_visualization()
    demo_ltp_ltd()
    demo_validation()
    
    print("\n\n" + "="*70)
    print("  DEMOSTRACIÓN COMPLETADA")
    print("="*70)


if __name__ == '__main__':
    main()
