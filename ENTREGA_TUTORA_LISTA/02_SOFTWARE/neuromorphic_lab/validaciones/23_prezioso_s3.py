"""
validaciones/23_prezioso_s3.py
==============================
Genera ejemplos sintéticos inspirados en la descripción pre-forming de Prezioso et al. (2015).
No carga series digitalizadas del artículo.
"""

import argparse
import csv
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thesis-provenance",
        action="store_true",
        help="Create separately named model-example figures and export their generated data.",
    )
    args = parser.parse_args()
    print("Generando ejemplos sintéticos del modelo pre-forming...")
    
    # 1. Obtener datos de simulacion
    results = validate_prezioso2015_s3()
    s3a = results['s3a']
    s3b = results['s3b']
    s3c = results['s3c']
    
    out_dir = Path(__file__).resolve().parent / "figuras"
    out_dir.mkdir(parents=True, exist_ok=True)
    thesis_dir = None
    if args.thesis_provenance:
        thesis_dir = ROOT_DIR.parent / "PLANTILLA_PDF_FINAL" / "figuras"
        thesis_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(__file__).resolve().parent / "datos"
    data_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_modelo_ejemplo"

    with (data_dir / "prezioso_s3a_modelo_ejemplo.csv").open(
        "w", newline="", encoding="utf-8"
    ) as data_file:
        writer = csv.writer(data_file)
        writer.writerow(
            [
                "voltage_V",
                *[f"simulated_device_{i + 1}_current_A" for i in range(len(s3a["I_curves"]))],
                "model_mean_current_A",
            ]
        )
        writer.writerows(zip(s3a["V_sweep"], *s3a["I_curves"], s3a["I_ref"]))

    with (data_dir / "prezioso_s3b_modelo_ejemplo.csv").open(
        "w", newline="", encoding="utf-8"
    ) as data_file:
        writer = csv.writer(data_file)
        writer.writerow(["row", "column", "simulated_conductance_S"])
        for row_index, row in enumerate(s3b["G_matrix"], start=1):
            for column_index, conductance in enumerate(row, start=1):
                writer.writerow([row_index, column_index, f"{conductance:.12g}"])

    with (data_dir / "prezioso_s3c_modelo_ejemplo.csv").open(
        "w", newline="", encoding="utf-8"
    ) as data_file:
        writer = csv.writer(data_file)
        writer.writerow(["bin_left_uS", "bin_right_uS", "count"])
        counts, edges = np.histogram(s3b["G_matrix"].flatten() * 1e6, bins=5)
        writer.writerows(zip(edges[:-1], edges[1:], counts))
    
    # FIGURA S3a: Curva S Asimetrica
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('black')
        
    for I_curve in s3a['I_curves']:
        ax.plot(s3a['V_sweep'], I_curve * 1e6, alpha=0.6)
    
    ax.plot(s3a['V_sweep'], s3a['I_ref'] * 1e6, 'k--', linewidth=2, label="Curva media del modelo")
    
    ax.set_title("Modelo paramétrico: curva I-V asimétrica", color='black')
    ax.set_xlabel("Voltaje (V)", color='black')
    ax.set_ylabel("Corriente (μA)", color='black')
    ax.grid(True, color='#e5e5e5', linestyle='--')
    ax.legend(facecolor='white', edgecolor='black', labelcolor='black')
    plt.tight_layout()
    plt.savefig(out_dir / f"23a_prezioso_s3a{suffix}.png", dpi=300)
    if thesis_dir is not None:
        plt.savefig(thesis_dir / f"23a_prezioso_s3a{suffix}.pdf")
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
            
    ax.set_title("Mapa sintético 4x4 de conductancias", pad=20, color='black')
    ax.set_xticks(range(s3b['n_cols']))
    ax.set_yticks(range(s3b['n_rows']))
    ax.set_xticklabels([f"C{j+1}" for j in range(s3b['n_cols'])])
    ax.set_yticklabels([f"W{i+1}" for i in range(s3b['n_rows'])])
    plt.tight_layout()
    plt.savefig(out_dir / f"23b_prezioso_s3b{suffix}.png", dpi=300)
    if thesis_dir is not None:
        plt.savefig(thesis_dir / f"23b_prezioso_s3b{suffix}.pdf")
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
    
    ax.set_title("Distribución sintética de conductancias 4x4", color='black')
    ax.set_xlabel("Conductancia (μS)", color='black')
    ax.set_ylabel("Densidad de Probabilidad", color='black')
    ax.grid(True, color='#e5e5e5', linestyle='--')
    ax.legend(facecolor='white', edgecolor='black', labelcolor='black')
    plt.tight_layout()
    plt.savefig(out_dir / f"23c_prezioso_s3c{suffix}.png", dpi=300)
    if thesis_dir is not None:
        plt.savefig(thesis_dir / f"23c_prezioso_s3c{suffix}.pdf")
    plt.close()

    print(f"Figuras generadas en {out_dir}")
    if thesis_dir is not None:
        print(f"Variantes para tesis generadas en {thesis_dir}")

if __name__ == "__main__":
    main()
