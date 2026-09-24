"""
validaciones/validacion_estadistica_completa.py
================================================
Validación estadística completa de Fase 1 (Etapa 1.2, Actividades 4-5).
Consolida variabilidad D2D, C2C y ruido controlado en un solo reporte riguroso.

Salidas:
  - neuromorphic_lab/validaciones/validacion_estadistica_completa.csv
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
from neurolab.validation.statistics import StatisticalAnalyzer


def simulate_d2d(n: int = 100, sigma_rel: float = 0.05, seed: int = 42):
    """Simula N dispositivos con variabilidad de fabricación D2D."""
    rng = np.random.default_rng(seed)
    RONs, ROFFs, x0s = [], [], []
    for _ in range(n):
        RON  = 100.0    * (1.0 + sigma_rel * rng.standard_normal())
        ROFF = 16_000.0 * (1.0 + sigma_rel * rng.standard_normal())
        x0   = float(np.clip(0.10 + sigma_rel * rng.standard_normal(), 0.01, 0.99))
        RONs.append(RON)
        ROFFs.append(ROFF)
        x0s.append(x0)
    return np.array(RONs), np.array(ROFFs), np.array(x0s)


def simulate_c2c(n_cycles: int = 50000, dt: float = 1e-3, seed: int = 42):
    """Simula N ciclos con variabilidad ciclo-a-ciclo C2C (Proceso Ornstein-Uhlenbeck en estado estacionario)."""
    rng = np.random.default_rng(seed)
    theta = 1.0        # tasa de relajación
    sigma = 0.05       # intensidad del ruido
    eta = np.zeros(n_cycles)
    for k in range(1, n_cycles):
        dW = rng.standard_normal() * np.sqrt(dt)
        eta[k] = eta[k-1] - theta * eta[k-1] * dt + sigma * dW
    
    # Descartar transitorio (primeros 10 segundos = 10,000 pasos)
    warmup = int(10.0 / dt)
    eta_ss = eta[warmup:] if n_cycles > warmup else eta
    return eta_ss, theta, sigma


def main():
    print("\n" + "=" * 80)
    print(" VALIDACIÓN ESTADÍSTICA COMPLETA — CIERRE DE FASE 1")
    print("=" * 80)

    output_dir = Path(__file__).resolve().parent

    # --- D2D ---
    RONs, ROFFs, x0s = simulate_d2d(n=100, sigma_rel=0.05)
    analyzers_d2d = [
        StatisticalAnalyzer(RONs, 'R_ON (Ohm)'),
        StatisticalAnalyzer(ROFFs, 'R_OFF (Ohm)'),
        StatisticalAnalyzer(x0s, 'x_0 (adim.)')
    ]

    print("\n--- 1. Variabilidad Device-to-Device D2D (n = 100) ---")
    for analyzer in analyzers_d2d:
        print(analyzer.report())
        print()

    # --- C2C ---
    eta, theta, sigma_c2c = simulate_c2c(n_cycles=500000, dt=1e-3)
    c2c_analyzer = StatisticalAnalyzer(eta, 'η(t) C2C')
    print(f"--- 2. Variabilidad Cycle-to-Cycle C2C (Proceso OU, n = {len(eta)}) ---")
    print(c2c_analyzer.report())

    # Varianza teórica vs. simulada
    var_teo = (sigma_c2c ** 2) / (2.0 * theta)
    var_sim = float(eta.var(ddof=1))
    err_rel = abs(var_sim - var_teo) / var_teo * 100.0

    print(f"\n  Varianza teórica OU:  {var_teo:.6e}")
    print(f"  Varianza simulada OU: {var_sim:.6e}")
    print(f"  Error relativo (%):   {err_rel:.4f} % ({'PASADO < 5%' if err_rel < 5.0 else 'REVISAR'})")
    print("=" * 80)

    # --- Guardar CSV ---
    csv_path = output_dir / "datos" / "validacion_estadistica_completa.csv"
    with open(csv_path, "w", encoding="utf-8") as f_out:
        f_out.write("Tipo,Magnitud,n,Media,Std,CV_percent,KS_p_value,Normalidad\n")
        for a in analyzers_d2d:
            s = a.summary()
            ks = a.ks_test_normal()
            norm_str = "RECHAZADA" if ks['reject_normal'] else "ACEPTADA"
            f_out.write(f"D2D,{s['name']},{s['n']},{s['mean']:.6f},{s['std']:.6f},{s['cv_percent']:.4f},{ks['p_value']:.4f},{norm_str}\n")
        
        s_c2c = c2c_analyzer.summary()
        ks_c2c = c2c_analyzer.ks_test_normal()
        norm_c2c = "RECHAZADA" if ks_c2c['reject_normal'] else "ACEPTADA"
        f_out.write(f"C2C,{s_c2c['name']},{s_c2c['n']},{s_c2c['mean']:.6f},{s_c2c['std']:.6f},{s_c2c['cv_percent']:.4f},{ks_c2c['p_value']:.4f},{norm_c2c}\n")
        f_out.write(f"C2C_Varianza,Theoretical_vs_Simulated,1000,{var_teo:.6e},{var_sim:.6e},{err_rel:.4f},N/A,Error_pct<{err_rel:.2f}%\n")

    # --- Figura en tema blanco académico ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor='#ffffff')
    fig.suptitle("Análisis Estadístico Completo de Variabilidad D2D y C2C (Fase 1)", color='#0f172a', fontsize=13, fontweight='bold')

    for ax in axes.flat:
        ax.set_facecolor('#ffffff')
        ax.tick_params(colors='#000000', labelcolor='#000000')
        ax.grid(True, color='#e2e8f0', linestyle='--', linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_color('#1e293b')
            spine.set_linewidth(1.1)

    # Panel 1: D2D R_ON vs Normal
    ax1 = axes[0, 0]
    count, bins, _ = ax1.hist(RONs, bins=15, density=True, color='#2563eb', alpha=0.7, edgecolor='#ffffff', label='D2D Muestras')
    x_grid = np.linspace(bins.min(), bins.max(), 100)
    pdf = stats.norm.pdf(x_grid, loc=RONs.mean(), scale=RONs.std(ddof=1))
    ax1.plot(x_grid, pdf, color='#dc2626', lw=2.0, label='Ajuste Normal Gaussiano')
    ax1.set_xlabel('R_ON (Ohm)', fontweight='bold')
    ax1.set_ylabel('Densidad', fontweight='bold')
    ax1.set_title(f'① R_ON D2D (KS p={analyzers_d2d[0].ks_test_normal()["p_value"]:.4f})', fontweight='bold')
    ax1.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a')

    # Panel 2: D2D R_OFF vs Normal
    ax2 = axes[0, 1]
    count, bins, _ = ax2.hist(ROFFs, bins=15, density=True, color='#059669', alpha=0.7, edgecolor='#ffffff', label='D2D Muestras')
    x_grid = np.linspace(bins.min(), bins.max(), 100)
    pdf = stats.norm.pdf(x_grid, loc=ROFFs.mean(), scale=ROFFs.std(ddof=1))
    ax2.plot(x_grid, pdf, color='#dc2626', lw=2.0, label='Ajuste Normal Gaussiano')
    ax2.set_xlabel('R_OFF (Ohm)', fontweight='bold')
    ax2.set_ylabel('Densidad', fontweight='bold')
    ax2.set_title(f'② R_OFF D2D (KS p={analyzers_d2d[1].ks_test_normal()["p_value"]:.4f})', fontweight='bold')
    ax2.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a')

    # Panel 3: D2D x0 vs Normal
    ax3 = axes[1, 0]
    count, bins, _ = ax3.hist(x0s, bins=15, density=True, color='#7c3aed', alpha=0.7, edgecolor='#ffffff', label='D2D Muestras')
    x_grid = np.linspace(bins.min(), bins.max(), 100)
    pdf = stats.norm.pdf(x_grid, loc=x0s.mean(), scale=x0s.std(ddof=1))
    ax3.plot(x_grid, pdf, color='#dc2626', lw=2.0, label='Ajuste Normal Gaussiano')
    ax3.set_xlabel('x_0 (adim.)', fontweight='bold')
    ax3.set_ylabel('Densidad', fontweight='bold')
    ax3.set_title(f'③ Estado Inicial x0 (KS p={analyzers_d2d[2].ks_test_normal()["p_value"]:.4f})', fontweight='bold')
    ax3.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a')

    # Panel 4: C2C Trayectoria y Varianza
    ax4 = axes[1, 1]
    t_c2c = np.arange(len(eta[:2000])) * 0.001
    ax4.plot(t_c2c, eta[:2000], color='#d97706', lw=1.2, label='Proceso OU η(t)')
    ax4.axhline(0, color='#64748b', linestyle='--', lw=0.8)
    ax4.set_xlabel('Tiempo (s)', fontweight='bold')
    ax4.set_ylabel('Fluctuación C2C η', fontweight='bold')
    ax4.set_title(f'④ C2C OU (Error Var = {err_rel:.2f}%)', fontweight='bold')
    ax4.legend(facecolor='#f8fafc', edgecolor='#cbd5e1', labelcolor='#0f172a')

    plt.tight_layout()
    png_path = output_dir / "figuras" / "validacion_estadistica.png"
    plt.savefig(png_path, dpi=300)
    fig_subfolder = output_dir.parent / "figuras"
    fig_subfolder.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_subfolder / "validacion_estadistica.png", dpi=300)
    plt.close()

    print(f"\n[OK] Reporte y figuras guardados en:")
    print(f"   - {csv_path}")
    print(f"   - {png_path}")


if __name__ == '__main__':
    main()
