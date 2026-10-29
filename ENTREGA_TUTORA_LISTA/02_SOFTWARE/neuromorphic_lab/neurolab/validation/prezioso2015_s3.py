"""
neurolab.validation.prezioso2015_s3
=====================================
Ejemplos de simulación inspirados en la descripción de Fig. S3 de Prezioso et al.:

    M. Prezioso, F. Merrikh-Bayat, B. D. Hoskins, G. C. Adam,
    K. K. Likharev & D. B. Strukov.
    "Training and operation of an integrated neuromorphic network
     based on metal-oxide memristors."
    Nature 521, 61–64 (2015). doi:10.1038/nature14441

La publicación describe la caracterización pre-forming de un crossbar 10×8.
Este módulo no carga datos digitalizados del artículo: S3a es un modelo
paramétrico con constantes locales; S3b y S3c usan muestras sintéticas
generadas con semilla fija sobre una matriz 4×4 por defecto. Por ello, estas
salidas ilustran comportamientos y no reproducen cuantitativamente Fig. S3.

Metas aproximadas anotadas durante el desarrollo (no verificadas contra CSV):
  - Curva asimétrica no lineal
  - Corrientes aproximadas tomadas de notas locales, no de datos importados
  - Modelo: I = G_0·V + a_p·(exp(b_p·V)−1)  para V > 0
             I = G_0·V − a_n·(exp(b_n·|V|)−1) para V < 0
  - Parámetros ajustados: a_p=0.02, b_p=5.0, a_n=0.03, b_n=4.0

Constantes sintéticas locales para S3b/S3c (no medidas del paper):
  - G_mean = 0.45 μS
  - G_sigma = 0.08 μS
"""

import numpy as np
from typing import Dict, Any


# --------------------------------------------------------------------------- #
# PARÁMETROS DEL EJEMPLO LOCAL (inspirados, no validados contra datos S3)
# --------------------------------------------------------------------------- #

PREZIOSO_VIRGIN_DEFAULTS = {
    'R_on_set':   4750.0,    # Ω — resistencia ON tras SET
    'R_on_reset': 1750.0,    # Ω — resistencia ON tras RESET
    'R_off':      1.0e6,     # Ω — resistencia OFF (estado virgen ≈ R_off)
    'V_p':        0.60,      # V — umbral SET (NO se alcanza en estado virgen)
    'V_n':        0.85,      # V — umbral RESET
    'A_p':        200.0,     # s⁻¹
    'A_n':        600.0,     # s⁻¹
    'x0':         0.02,      # adim. — estado inicial virgen
    'G_initial':  0.45e-6,   # S — conductancia media virgen @ 0.1 V
    'G_sigma':    0.08e-6,   # S — dispersión D2D
    'V_read':     0.1,       # V — voltaje de lectura
    'D':          30e-9,     # m — espesor Al₂O₃/TiO₂-x
    'mu_v':       1e-12,     # m²/(V·s)
    'dt':         1e-4,      # s
    # --- Conducción no lineal: ejemplo paramétrico local ---
    # Modelo: I = G_0·V + a_p·(exp(b_p·V)−1)  para V > 0
    #         I = G_0·V − a_n·(exp(b_n·|V|)−1) para V < 0
    # Constantes locales de forma; no se ajustan automáticamente a datos fuente.
    'a_p':        2.67e-7,     # A — amplitud exponencial rama positiva
    'b_p':        2.77,        # V⁻¹ — tasa exponencial positiva
    'a_n':        9.4e-8,      # A — amplitud exponencial rama negativa
    'b_n':        3.32,        # V⁻¹ — tasa exponencial negativa
}


# --------------------------------------------------------------------------- #
# S3a-like — Curvas I-V sintéticas para ilustrar no linealidad asimétrica
# --------------------------------------------------------------------------- #

def _current_virgin_iv(
    V: np.ndarray,
    G_0: float,
    a_p: float,
    b_p: float,
    a_n: float,
    b_n: float,
) -> np.ndarray:
    """
    Corriente del dispositivo virgen (pre-forming) con conducción no lineal.

    Modelo empírico paramétrico inspirado en la forma descrita para S3a:
      - Rama positiva (V > 0): I = G_0·V + a_p·(exp(b_p·V) − 1)
      - Rama negativa (V < 0): I = G_0·V − a_n·(exp(b_n·|V|) − 1)
      - Resultado: curva S-shaped asimétrica (NO lineal)

    No se carga ni se compara con una serie experimental digitalizada.

    Args:
        V:    Array de voltajes (V).
        G_0:  Conductancia lineal de fondo (S).
        a_p:  Amplitud exponencial rama positiva (A).
        b_p:  Tasa de crecimiento exponencial positiva (V⁻¹).
        a_n:  Amplitud exponencial rama negativa (A).
        b_n:  Tasa de crecimiento exponencial negativa (V⁻¹).

    Returns:
        Array de corrientes (A).
    """
    V = np.asarray(V, dtype=float)
    I = np.zeros_like(V)

    # Rama positiva: V > 0
    mask_p = V > 0
    bV_p = np.clip(b_p * V[mask_p], -20.0, 20.0)   # evitar overflow
    I[mask_p] = G_0 * V[mask_p] + a_p * (np.exp(bV_p) - 1.0)

    # Rama negativa: V < 0
    mask_n = V < 0
    bV_n = np.clip(b_n * np.abs(V[mask_n]), -20.0, 20.0)
    I[mask_n] = G_0 * V[mask_n] - a_n * (np.exp(bV_n) - 1.0)

    # V = 0: I = 0 (ya inicializado)
    return I


def simulate_iv_virgin(
    n_devices: int = 5,
    V_max: float = 1.0,
    n_points: int = 100,
    G_mean: float = 0.45e-6,
    G_sigma: float = 0.08e-6,
    a_p: float = 2.67e-7,
    b_p: float = 2.77,
    a_n: float = 9.4e-8,
    b_n: float = 3.32,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simula curvas I-V de N dispositivos en estado virgen (pre-forming).

    El modelo aplica una corriente no lineal asimétrica para ilustrar una
    curva S. No carga datos digitalizados ni certifica una reproducción de
    Fig. S3a. El estado x del memristor NO se actualiza.

    Args:
        n_devices:  Número de dispositivos a simular.
        V_max:      Amplitud máxima del barrido de voltaje (V). Default 1.0 V.
        n_points:   Número de puntos por sentido del barrido.
        G_mean:     Conductancia media de la población (S).
        G_sigma:    Desviación estándar D2D de conductancias (S).
        a_p:        Amplitud no lineal rama positiva (A).
        b_p:        Tasa exponencial rama positiva (V⁻¹).
        a_n:        Amplitud no lineal rama negativa (A).
        b_n:        Tasa exponencial rama negativa (V⁻¹).
        seed:       Semilla para reproducibilidad.

    Returns:
        dict con:
          'V_sweep'    : array de voltajes del barrido triangular
          'I_curves'   : lista de arrays de corriente (A) por dispositivo
          'G_devices'  : lista de conductancias G_0 (S) por dispositivo
          'params'     : dict de parámetros a_p, b_p, a_n, b_n usados
          'I_ref_pos'  : corriente @ V_max del dispositivo con G=G_mean
          'I_ref_neg'  : corriente @ -V_max del dispositivo con G=G_mean
          'asym_ratio' : |I_ref_pos / I_ref_neg| (asimetría)
    """
    rng = np.random.default_rng(seed)

    # Barrido triangular: subida (−V_max → +V_max) + bajada (+V_max → −V_max)
    V_up   = np.linspace(-V_max, V_max, n_points)
    V_down = np.linspace(V_max, -V_max, n_points)
    V_sweep = np.concatenate([V_up, V_down])

    # Generar conductancias base G_0 con variabilidad D2D gaussiana
    G_devices = []
    for _ in range(n_devices):
        G0 = G_mean + G_sigma * rng.standard_normal()
        G0 = float(np.clip(G0, 0.15e-6, 0.90e-6))
        G_devices.append(G0)

    # Variabilidad D2D también en los parámetros exponenciales (±10%)
    I_curves = []
    for G0 in G_devices:
        a_p_k = float(np.clip(a_p * (1.0 + 0.10 * rng.standard_normal()), 1e-8, 1e-5))
        b_p_k = float(np.clip(b_p * (1.0 + 0.05 * rng.standard_normal()), 0.5, 10.0))
        a_n_k = float(np.clip(a_n * (1.0 + 0.10 * rng.standard_normal()), 1e-8, 1e-5))
        b_n_k = float(np.clip(b_n * (1.0 + 0.05 * rng.standard_normal()), 0.5, 10.0))

        I = _current_virgin_iv(V_sweep, G0, a_p_k, b_p_k, a_n_k, b_n_k)
        I_curves.append(I)

    # Corriente de referencia (dispositivo con G_mean, sin variabilidad)
    I_ref = _current_virgin_iv(V_sweep, G_mean, a_p, b_p, a_n, b_n)
    I_ref_pos = float(_current_virgin_iv(np.array([V_max]),  G_mean, a_p, b_p, a_n, b_n)[0])
    I_ref_neg = float(_current_virgin_iv(np.array([-V_max]), G_mean, a_p, b_p, a_n, b_n)[0])
    asym = abs(I_ref_pos / I_ref_neg) if abs(I_ref_neg) > 1e-15 else float('inf')

    return {
        'V_sweep':    V_sweep,
        'I_curves':   I_curves,
        'I_ref':      I_ref,
        'G_devices':  G_devices,
        'params':     {'a_p': a_p, 'b_p': b_p, 'a_n': a_n, 'b_n': b_n},
        'I_ref_pos':  I_ref_pos,
        'I_ref_neg':  I_ref_neg,
        'asym_ratio': asym,
        'n_points_per_sweep': n_points,
    }


# --------------------------------------------------------------------------- #
# Fig. S3b — Mapa de conductancias N×M
# --------------------------------------------------------------------------- #

def simulate_conductance_map(
    n_rows: int = 4,
    n_cols: int = 4,
    G_mean: float = 0.45e-6,
    G_sigma: float = 0.08e-6,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Genera la matriz de conductancias del crossbar en estado virgen.

    Mapa de conductancias para la matriz crossbar 4×4.

    Args:
        n_rows:   Número de filas (wordlines).
        n_cols:   Número de columnas (bitlines).
        G_mean:   Conductancia media de la distribución (S).
        G_sigma:  Desviación estándar D2D (S).
        seed:     Semilla para reproducibilidad.

    Returns:
        dict con 'G_matrix' [n_rows × n_cols] en Siemens,
        'G_mean', 'G_std', 'G_min', 'G_max', 'CV_pct'.
    """
    rng = np.random.default_rng(seed)

    G_matrix = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            G = G_mean + G_sigma * rng.standard_normal()
            G = float(np.clip(G, 0.15e-6, 0.80e-6))
            G_matrix[i, j] = G

    G_flat = G_matrix.flatten()
    return {
        'G_matrix': G_matrix,
        'G_mean':   float(G_flat.mean()),
        'G_std':    float(G_flat.std()),
        'G_min':    float(G_flat.min()),
        'G_max':    float(G_flat.max()),
        'CV_pct':   float(G_flat.std() / G_flat.mean() * 100.0),
        'n_rows':   n_rows,
        'n_cols':   n_cols,
    }


# --------------------------------------------------------------------------- #
# Fig. S3c — Histograma de conductancias
# --------------------------------------------------------------------------- #

def compute_histogram_stats(
    G_matrix: np.ndarray,
    n_bins: int = 12,
) -> Dict[str, Any]:
    """
    Calcula estadísticas e histograma para Fig. S3c.

    Args:
        G_matrix:  Matriz de conductancias (S).
        n_bins:    Número de bins del histograma.

    Returns:
        dict con 'bins', 'counts', 'bin_centers', 'G_mean', 'G_std',
        'CV_pct', 'normal_x', 'normal_y' (curva normal ajustada).
    """
    G_flat = G_matrix.flatten() * 1e6   # μS

    counts, bin_edges = np.histogram(G_flat, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    G_mean_us = float(G_flat.mean())
    G_std_us  = float(G_flat.std())
    CV_pct    = G_std_us / G_mean_us * 100.0 if G_mean_us > 0 else 0.0

    # Curva normal ajustada (escala de densidad × N × bin_width)
    x_norm = np.linspace(G_flat.min() - G_std_us, G_flat.max() + G_std_us, 200)
    y_norm = (
        len(G_flat) * bin_width
        * (1.0 / (G_std_us * np.sqrt(2 * np.pi)))
        * np.exp(-0.5 * ((x_norm - G_mean_us) / G_std_us) ** 2)
    )

    return {
        'bins':        bin_edges,
        'counts':      counts,
        'bin_centers': bin_centers,
        'G_mean':      G_mean_us,
        'G_std':       G_std_us,
        'CV_pct':      CV_pct,
        'normal_x':    x_norm,
        'normal_y':    y_norm,
        'n_devices':   len(G_flat),
    }


# --------------------------------------------------------------------------- #
# Compatibilidad: función que evalúa criterios internos de los ejemplos sintéticos
# --------------------------------------------------------------------------- #

def validate_prezioso2015_s3(
    n_devices: int = 5,
    n_rows: int = 4,
    n_cols: int = 4,
    G_mean: float = 0.45e-6,
    G_sigma: float = 0.08e-6,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Ejecuta tres ejemplos locales de curvas y conductancias inspirados en
    Fig. S3. No los contrasta con datos digitalizados del artículo.

    Returns:
        dict con claves 's3a', 's3b', 's3c' y 'pass_all'.
    """
    # S3a
    s3a = simulate_iv_virgin(
        n_devices=n_devices, G_mean=G_mean, G_sigma=G_sigma, seed=seed
    )

    # S3b
    s3b = simulate_conductance_map(
        n_rows=n_rows, n_cols=n_cols,
        G_mean=G_mean, G_sigma=G_sigma, seed=seed
    )

    # S3c
    s3c = compute_histogram_stats(s3b['G_matrix'])

    # Criterios internos para el ejemplo sintético; no comparan con mediciones.
    g_range_ok    = 0.25 < s3b['G_mean'] * 1e6 < 0.65
    cv_ok         = 10.0 < s3c['CV_pct'] < 25.0
    # Comprueba la asimetría definida por los parámetros locales.
    params = s3a['params']
    _V075p = np.array([0.75])
    _V075n = np.array([-0.75])
    I_075p = abs(float(_current_virgin_iv(_V075p, 0.45e-6, params['a_p'], params['b_p'], params['a_n'], params['b_n'])[0]))
    I_075n = abs(float(_current_virgin_iv(_V075n, 0.45e-6, params['a_p'], params['b_p'], params['a_n'], params['b_n'])[0]))
    asym_ok       = I_075p > I_075n
    i_pos_ok      = I_075p * 1e6 > 1.0

    return {
        's3a':       s3a,
        's3b':       s3b,
        's3c':       s3c,
        'pass_all':  g_range_ok and cv_ok and asym_ok and i_pos_ok,
        'criteria':  {
            'G_media en rango [0.25, 0.65] μS':    g_range_ok,
            'CV en rango [10, 25] %':               cv_ok,
            'Asimetria I-V (pos > neg)':            asym_ok,
            'I(+V_max) > 1 uA (curva S no lineal)': i_pos_ok,
        }
    }


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(errors='replace')

    print("=" * 60)
    print("  EJEMPLOS SINTETICOS INSPIRADOS EN PREZIOSO 2015 (NO VALIDACION EMPIRICA)")
    print("=" * 60)

    results = validate_prezioso2015_s3()

    print(f"\n  S3a — Curvas I-V (modelo no lineal asimetrico):")
    print(f"    I(+V_max):    {results['s3a']['I_ref_pos']*1e6:.4f} uA")
    print(f"    I(-V_max):    {results['s3a']['I_ref_neg']*1e6:.4f} uA")
    print(f"    Asimetria:    {results['s3a']['asym_ratio']:.4f}x")
    print(f"    N curvas:     {len(results['s3a']['I_curves'])}")

    print(f"\n  S3b — Mapa de conductancias {results['s3b']['n_rows']}x{results['s3b']['n_cols']}:")
    print(f"    G media:  {results['s3b']['G_mean']*1e6:.4f} uS")
    print(f"    G std:    {results['s3b']['G_std']*1e6:.4f} uS")
    print(f"    G min:    {results['s3b']['G_min']*1e6:.4f} uS")
    print(f"    G max:    {results['s3b']['G_max']*1e6:.4f} uS")
    print(f"    CV:       {results['s3b']['CV_pct']:.2f} %")

    print(f"\n  S3c — Histograma:")
    print(f"    G media: {results['s3c']['G_mean']:.4f} uS")
    print(f"    G std:   {results['s3c']['G_std']:.4f} uS")
    print(f"    CV:      {results['s3c']['CV_pct']:.2f} %")

    print(f"\n  Criterios internos del modelo local:")
    for k, v in results['criteria'].items():
        print(f"    {'[OK]' if v else '[FAIL]'} {k}")

    print(f"\n  RESULTADO GENERAL: {'PASS' if results['pass_all'] else 'FAIL'}")
    print("=" * 60)
