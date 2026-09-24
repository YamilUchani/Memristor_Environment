"""
validaciones/validacion_ruido.py
=================================
Validación del efecto del ruido controlado en el modelo memristor (Etapa 1.2, Actividad 3).
Compara: Sin Ruido (determinista) vs. Ruido Gaussiano (σ_rel = 5%) en R_ON, R_OFF y x0.

Salidas:
  - neuromorphic_lab/validaciones/validacion_ruido.png
  - neuromorphic_lab/validaciones/validacion_ruido.csv
"""

import sys
from pathlib import Path

# Reconfigurar encoding para consola Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# Garantizar que el directorio raíz de neuromorphic_lab esté en sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from neurolab.devices import MemristorStrukov
from neurolab.configs import StrukovConfig


def simulate_with_noise(noise_enabled: bool, sigma_rel: float = 0.05, n_samples: int = 50, seed: int = 42):
    """
    Simula N dispositivos con/sin ruido en parámetros nominales.
    """
    rng = np.random.default_rng(seed)
    results = []

    # Señal de entrada (senoidal)
    T, dt = 6.0, 1e-4
    t = np.arange(0, T, dt)
    V = np.sin(2.0 * np.pi * 0.5 * t)

    for i in range(n_samples):
        # Parámetros nominales
        RON = 100.0
        ROFF = 16_000.0
        x0 = 0.10

        # Aplicar ruido si está habilitado
        if noise_enabled:
            RON  *= (1.0 + sigma_rel * rng.standard_normal())
            ROFF *= (1.0 + sigma_rel * rng.standard_normal())
            x0    = float(np.clip(x0 + sigma_rel * rng.standard_normal(), 0.01, 0.99))

        cfg = StrukovConfig(RON=RON, ROFF=ROFF, x0=x0, D=10e-9, mu_v=1e-14, clip_x=True)
        mem = MemristorStrukov(cfg)

        I = np.zeros_like(t)
        for k in range(len(t)):
            I[k] = mem.current(V[k])
            mem.update(V[k], dt)

        results.append({
            'RON': RON,
            'ROFF': ROFF,
            'x0': x0,
            'I_max': float(I.max()),
            'I_min': float(I.min()),
            'I_pp': float(I.max() - I.min()),
        })

    return results


def compute_stats(results):
    """Calcula μ, σ y CV para cada magnitud."""
    stats = {}
    for key in ['RON', 'ROFF', 'x0', 'I_max', 'I_min', 'I_pp']:
        vals = np.array([r[key] for r in results])
        mu = float(vals.mean())
        sigma = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        cv = (sigma / mu * 100.0) if mu != 0 else 0.0
        stats[key] = {'mu': mu, 'sigma': sigma, 'cv': cv}
    return stats


def main():
    print("=" * 80)
    print(" VALIDACIÓN DEL EFECTO DEL RUIDO CONTROLADO EN EL MODELO MEMRISTOR")
    print("=" * 80)

    output_dir = Path(__file__).resolve().parent
    fig_subfolder = output_dir.parent / "figuras"
    fig_subfolder.mkdir(parents=True, exist_ok=True)

    # --- Simulaciones ---
    res_sin = simulate_with_noise(noise_enabled=False, n_samples=50)
    res_con = simulate_with_noise(noise_enabled=True, sigma_rel=0.05, n_samples=50)

    stats_sin = compute_stats(res_sin)
    stats_con = compute_stats(res_con)

    # --- Tabla comparativa en consola ---
    print("\n" + "=" * 78)
    print(" ANÁLISIS ESTADÍSTICO: SIN RUIDO (DETERMINISTA) vs. CON RUIDO (σ_rel = 5%)")
    print("=" * 78)
    print(f"{'Magnitud':<10} {'μ (sin)':>12} {'σ (sin)':>12} {'CV% (sin)':>10}"
          f" {'μ (con)':>12} {'σ (con)':>12} {'CV% (con)':>10}")
    print("-" * 78)
    for key in ['RON', 'ROFF', 'x0', 'I_pp']:
        s = stats_sin[key]
        c = stats_con[key]
        print(f"{key:<10} {s['mu']:>12.4f} {s['sigma']:>12.4f} {s['cv']:>10.2f}"
              f" {c['mu']:>12.4f} {c['sigma']:>12.4f} {c['cv']:>10.2f}")
    print("=" * 78)

    # --- Figura en tema blanco académico ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor='#ffffff')
    fig.suptitle("Efecto del Ruido Controlado (5%) en Parámetros Nominales y Respuesta Memristiva", color='#0f172a', fontsize=13, fontweight='bold')

    for ax in axes.flat:
        ax.set_facecolor('#ffffff')
        ax.tick_params(colors='#000000', labelcolor='#000000')
        ax.grid(True, color='#e2e8f0', linestyle='--', linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_color('#1e293b')
            spine.set_linewidth(1.1)

    # Histograma RON
    axes[0, 0].hist([r['RON'] for r in res_sin], bins=15, alpha=0.6, label='Sin ruido', color='#2563eb', edgecolor='#ffffff')
    axes[0, 0].hist([r['RON'] for r in res_con], bins=15, alpha=0.6, label='Con ruido (5%)', color='#dc2626', edgecolor='#ffffff')
    axes[0, 0].set_xlabel('R_ON (Ohm)', fontweight='bold')
    axes[0, 0].set_ylabel('Frecuencia', fontweight='bold')
    axes[0, 0].set_title('① Distribución de R_ON', fontweight='bold')
    axes[0, 0].legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a')

    # Histograma ROFF
    axes[0, 1].hist([r['ROFF'] for r in res_sin], bins=15, alpha=0.6, label='Sin ruido', color='#2563eb', edgecolor='#ffffff')
    axes[0, 1].hist([r['ROFF'] for r in res_con], bins=15, alpha=0.6, label='Con ruido (5%)', color='#dc2626', edgecolor='#ffffff')
    axes[0, 1].set_xlabel('R_OFF (Ohm)', fontweight='bold')
    axes[0, 1].set_ylabel('Frecuencia', fontweight='bold')
    axes[0, 1].set_title('② Distribución de R_OFF', fontweight='bold')
    axes[0, 1].legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a')

    # Histograma I_pp
    axes[1, 0].hist([r['I_pp'] * 1e3 for r in res_sin], bins=15, alpha=0.6, label='Sin ruido', color='#2563eb', edgecolor='#ffffff')
    axes[1, 0].hist([r['I_pp'] * 1e3 for r in res_con], bins=15, alpha=0.6, label='Con ruido (5%)', color='#dc2626', edgecolor='#ffffff')
    axes[1, 0].set_xlabel('I_pp (mA)', fontweight='bold')
    axes[1, 0].set_ylabel('Frecuencia', fontweight='bold')
    axes[1, 0].set_title('③ Distribución de Corriente Pico-Pico', fontweight='bold')
    axes[1, 0].legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a')

    # Bar plot CV comparativo
    keys = ['RON', 'ROFF', 'x0', 'I_pp']
    cv_sin = [stats_sin[k]['cv'] for k in keys]
    cv_con = [stats_con[k]['cv'] for k in keys]
    x_pos = np.arange(len(keys))
    axes[1, 1].bar(x_pos - 0.2, cv_sin, 0.4, label='Sin ruido', color='#2563eb')
    axes[1, 1].bar(x_pos + 0.2, cv_con, 0.4, label='Con ruido (5%)', color='#dc2626')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(keys, fontweight='bold')
    axes[1, 1].set_ylabel('CV (%)', fontweight='bold')
    axes[1, 1].set_title('④ Coeficiente de Variación CV (%)', fontweight='bold')
    axes[1, 1].legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a')

    plt.tight_layout()
    png_path = output_dir / "figuras" / "validacion_ruido.png"
    plt.savefig(png_path, dpi=300)
    plt.savefig(fig_subfolder / "validacion_ruido.png", dpi=300)
    plt.close()

    # --- Guardar CSV ---
    csv_path = output_dir / "datos" / "validacion_ruido.csv"
    with open(csv_path, "w", encoding="utf-8") as f_out:
        f_out.write("Modo,Magnitud,Mu,Sigma,CV_percent\n")
        for key in keys:
            s = stats_sin[key]
            c = stats_con[key]
            f_out.write(f"SinRuido,{key},{s['mu']:.6f},{s['sigma']:.6f},{s['cv']:.4f}\n")
            f_out.write(f"ConRuido,{key},{c['mu']:.6f},{c['sigma']:.6f},{c['cv']:.4f}\n")

    print(f"\n[OK] Archivos generados exitosamente:")
    print(f"   - {png_path}")
    print(f"   - {csv_path}")


if __name__ == '__main__':
    main()
