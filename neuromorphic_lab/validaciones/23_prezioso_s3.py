"""
validaciones/23_prezioso_s3.py
==============================
Genera las 3 figuras de Prezioso 2015 (Fig S3a, S3b, S3c) con formato blanco para la tesis.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from neurolab.validation.prezioso2015_s3 import validate_prezioso2015_s3

def main():
    print("Generando figuras para Prezioso 2015 (S3a, S3b, S3c) con formato de tesis...")
    
    # 1. Obtener datos de simulacion
    results = validate_prezioso2015_s3()
    s3a = results['s3a']
    s3b = results['s3b']
    s3c = results['s3c']
    
    out_dir = Path(__file__).resolve().parent / "figuras"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # FIGURA S3a: Curva S Asimetrica
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('black')
        
    for I_curve in s3a['I_curves']:
        ax.plot(s3a['V_sweep'], I_curve * 1e6, alpha=0.6)
    
    ax.plot(s3a['V_sweep'], s3a['I_ref'] * 1e6, 'k--', linewidth=2, label="Media Ajustada")
    
    ax.set_title("Prezioso 2015 Fig S3a - Curva I-V Virgen (Curva S)", color='black')
    ax.set_xlabel("Voltaje (V)", color='black')
    ax.set_ylabel("Corriente (μA)", color='black')
    ax.grid(True, color='#e5e5e5', linestyle='--')
    ax.legend(facecolor='white', edgecolor='black', labelcolor='black')
    plt.tight_layout()
    plt.savefig(out_dir / "23a_prezioso_s3a.png", dpi=300)
    plt.close()
    
    # FIGURA S3b: Mapa de Conductancias 4x4
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
    ax.set_facecolor('white')
    G_matrix = s3b['G_matrix'] * 1e6 # a uS
    
    cax = ax.matshow(G_matrix, cmap='YlOrRd')
    fig.colorbar(cax, label='Conductancia (μS)')
    
    for i in range(G_matrix.shape[0]):
        for j in range(G_matrix.shape[1]):
            ax.text(j, i, f"{G_matrix[i, j]:.1f}\nμS", va='center', ha='center', color='black', fontweight='bold')
            
    ax.set_title("Prezioso 2015 Fig S3b - Mapa de Conductancias 4x4", pad=20, color='black')
    ax.set_xticks(range(s3b['n_cols']))
    ax.set_yticks(range(s3b['n_rows']))
    ax.set_xticklabels([f"C{j+1}" for j in range(s3b['n_cols'])])
    ax.set_yticklabels([f"W{i+1}" for i in range(s3b['n_rows'])])
    plt.tight_layout()
    plt.savefig(out_dir / "23b_prezioso_s3b.png", dpi=300)
    plt.close()
    
    # FIGURA S3c: Histograma de G
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('black')
        
    G_flat = s3b['G_matrix'].flatten() * 1e6
    ax.hist(G_flat, bins=5, color='salmon', edgecolor='black', density=True, alpha=0.7)
    
    x_norm = np.linspace(G_flat.min() - 0.2, G_flat.max() + 0.2, 100)
    y_norm = norm.pdf(x_norm, s3c['G_mean'], s3c['G_std'])
    ax.plot(x_norm, y_norm, 'k--', linewidth=2, label=f"Normal (μ={s3c['G_mean']:.2f}, σ={s3c['G_std']:.2f} μS)")
    
    ax.set_title("Prezioso 2015 Fig S3c - Histograma de Conductancia 4x4", color='black')
    ax.set_xlabel("Conductancia (μS)", color='black')
    ax.set_ylabel("Densidad de Probabilidad", color='black')
    ax.grid(True, color='#e5e5e5', linestyle='--')
    ax.legend(facecolor='white', edgecolor='black', labelcolor='black')
    plt.tight_layout()
    plt.savefig(out_dir / "23c_prezioso_s3c.png", dpi=300)
    plt.close()

    print("Figuras generadas exitosamente en neuromorphic_lab/validaciones/figuras/")

if __name__ == "__main__":
    main()
