"""
demo_crossbar_completo.py
==========================
Demostración de TODOS los componentes del crossbar.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Asegurar import de neurolab si se ejecuta desde raíz
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'neuromorphic_lab'))

from neurolab.crossbar import Crossbar, CrossbarConfig
from neurolab.crossbar.validation import (
    validate_read_operation,
    validate_programming_selectivity,
    validate_sneak_paths,
    validate_line_resistance,
    validate_scalability,
    validate_d2d_c2c_variability,
)


def main():
    print("\n" + "="*70)
    print("  DEMOSTRACION CROSSBAR COMPLETO -- Fase 4")
    print("="*70)
    
    # ============================================================
    # 1. CREAR CROSSBAR CON D2D Y C2C
    # ============================================================
    print("\n\n[1] CREACIÓN DEL CROSSBAR CON D2D Y C2C")
    print("-"*70)
    
    cfg = CrossbarConfig(
        n_rows=4, n_cols=4,
        G_mean=0.45e-6, G_sigma=0.08e-6,
        enable_d2d=True, d2d_sigma=0.08, d2d_range=(0.2e-6, 0.7e-6),
        enable_c2c=True, c2c_sigma=0.05, c2c_range=(0.9, 1.1),
        programming_mode='V2',
    )
    cb = Crossbar(cfg)
    
    cb.print_state("ESTADO INICIAL (D2D + C2C)")
    cb.summary()
    
    # ============================================================
    # 2. VALIDAR VARIABILIDAD D2D Y C2C
    # ============================================================
    print("\n\n[2] VALIDACIÓN DE VARIABILIDAD D2D Y C2C")
    validate_d2d_c2c_variability(cb)

    # ============================================================
    # 3. VALIDAR LECTURA I = G^T · V
    # ============================================================
    print("\n\n[3] VALIDACIÓN DE LECTURA")
    validate_read_operation(cb, n_tests=100)
    
    # ============================================================
    # 4. VALIDAR PROGRAMACIÓN SELECTIVA
    # ============================================================
    print("\n\n[4] VALIDACIÓN DE PROGRAMACIÓN V/2")
    validate_programming_selectivity(cb)
    
    # ============================================================
    # 5. VALIDAR SNEAK PATHS
    # ============================================================
    print("\n\n[5] VALIDACIÓN DE SNEAK PATHS")
    V_rows = np.array([0.5, 1.0, 0.5, 1.0])
    validate_sneak_paths(cb, V_rows)
    
    # ============================================================
    # 6. VALIDAR RESISTENCIAS DE LÍNEA
    # ============================================================
    print("\n\n[6] VALIDACIÓN DE RESISTENCIAS DE LÍNEA")
    validate_line_resistance(cb, V_rows,
                              R_H_values=[0.0, 0.1, 1.0, 5.0, 10.0])
    
    # ============================================================
    # 7. VALIDAR ESCALABILIDAD
    # ============================================================
    print("\n\n[7] VALIDACIÓN DE ESCALABILIDAD")
    validate_scalability()
    
    # ============================================================
    # 8. GENERAR FIGURAS
    # ============================================================
    print("\n\n[8] GENERANDO FIGURAS")
    generate_figures(cb)
    
    print("\n\n" + "="*70)
    print("  DEMOSTRACION COMPLETADA")
    print("="*70)


def generate_figures(cb):
    """Genera figuras comparativas."""
    
    # --- Figura 1: Crossbar heatmap ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    G = cb.G_matrix * 1e6
    im0 = axes[0].imshow(G, cmap='YlOrRd', aspect='auto')
    axes[0].set_title('G matrix (μS)')
    axes[0].set_xlabel('Columna')
    axes[0].set_ylabel('Fila')
    plt.colorbar(im0, ax=axes[0])
    for i in range(cb.n_rows):
        for j in range(cb.n_cols):
            axes[0].text(j, i, f'{G[i,j]:.1f}', ha='center', va='center',
                         fontsize=8)
    
    # --- Figura 2: Ideal vs sneak paths ---
    V_rows = np.linspace(-0.5, 0.5, cb.n_rows)
    I_ideal = cb.read_ideal(V_rows) * 1e6
    
    cb.cfg.enable_sneak_paths = True
    I_sneak = cb.read(V_rows) * 1e6
    cb.cfg.enable_sneak_paths = False
    
    x = np.arange(cb.n_cols)
    axes[1].bar(x - 0.2, I_ideal, 0.4, label='Ideal', color='steelblue')
    axes[1].bar(x + 0.2, I_sneak, 0.4, label='Con sneak', color='indianred')
    axes[1].set_xlabel('Columna')
    axes[1].set_ylabel('Corriente (μA)')
    axes[1].set_title('Ideal vs Sneak paths')
    axes[1].legend()
    
    # --- Figura 3: Escalabilidad ---
    sizes = [4, 8, 16, 32, 64]
    errors = []
    for n in sizes:
        cb_tmp = Crossbar(n_rows=n, n_cols=n)
        cb_tmp.cfg.enable_sneak_paths = True
        cb_tmp.cfg.R_sneak_factor = 0.05
        V = np.ones(n) * 0.1
        I_id = cb_tmp.read_ideal(V)
        I_sn = cb_tmp.read(V)
        err = np.mean(np.abs(I_sn - I_id) / np.abs(I_id) * 100)
        errors.append(err)
    
    axes[2].plot(sizes, errors, 'o-', color='darkgreen', lw=2)
    axes[2].set_xlabel('Tamaño (N×N)')
    axes[2].set_ylabel('Error sneak (%)')
    axes[2].set_title('Error vs tamaño')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_img = 'crossbar_validacion_completa.png'
    plt.savefig(output_img, dpi=150)
    plt.close()
    
    print(f"  [OK] Figura guardada: {output_img}")


if __name__ == '__main__':
    main()
