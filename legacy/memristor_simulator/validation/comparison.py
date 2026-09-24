"""
validation/comparison.py
=========================
Comparación cuantitativa con los resultados del paper Strukov (2008).

Dado que el paper no publica datos numéricos tabulados, la comparación
se realiza a nivel morfológico (forma de la curva) y a nivel de las
métricas físicas clave reportadas en el texto.

Métricas de referencia extraídas del paper (Figura 2b):
  - Doble lazo de histéresis simétrico alrededor del origen
  - i_max ≈ ±10 mA para v₀ = 1 V (tabla de unidades)
  - Δ(w/D) < 1 (el estado no alcanza los bordes)
  - Colapso de histéresis a línea recta al aumentar 10× la frecuencia

Referencias
-----------
  Strukov, D. B. et al. Nature 453, 80–83 (2008).
  DOI: 10.1038/nature06932
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List
from .metrics import (calculate_r_squared, rmse, relative_error,
                       hysteresis_area, state_excursion, hysteresis_collapse_index)


@dataclass
class PaperSpec:
    """Especificación cuantitativa de referencia del paper (Fig 2b)."""
    # Rango de corriente (A) — ajustado al valor físico real (la gráfica original escala i_0)
    i_max_expected: float = 0.12e-3      # A
    i_max_tolerance: float = 0.20        # ±20 % tolerancia

    # Rango de voltaje
    v_amplitude_expected: float = 1.0    # V
    v_amplitude_tolerance: float = 0.01  # ±1 %

    # Variable de estado: no debe alcanzar los bordes exactos
    x_min_threshold: float = 0.01        # x no debe llegar a 0 exacto
    x_max_threshold: float = 0.99        # x no debe llegar a 1 exacto

    # Histéresis visible a baja frecuencia
    min_hysteresis_area: float = 1e-5    # V·A mínimos

    # Colapso a ≥90 % al multiplicar frecuencia por 10
    frequency_collapse_threshold: float = 0.90


class PaperComparison:
    """
    Herramienta de validación morfológica y cuantitativa contra
    los resultados del paper Strukov (2008).

    Uso::

        comp = PaperComparison()
        report = comp.validate_fig2b(results_fig2b)
        comp.print_report(report)

        # Comparar colapso de frecuencia
        comp.validate_frequency_collapse(results_low_freq, results_high_freq)
    """

    def __init__(self, spec: PaperSpec = None):
        self.spec = spec or PaperSpec()

    # ── Validación de Figura 2b ───────────────────────────────────────────

    def validate_fig2b(self, results: Dict[str, np.ndarray]) -> dict:
        """
        Valida que los resultados de simulación reproduzcan la Fig 2b del paper.

        Criterios de APROBACIÓN:
          [P1] Rango de corriente dentro de ±20 % del valor teórico
          [P2] Histéresis visible (área > umbral mínimo)
          [P3] Estado x no toca los bordes exactos (0 o 1)
          [P4] Corriente pasa por el origen cuando V = 0 (característica memristiva)
          [P5] Curva I-V pinchada en el origen (pinched hysteresis loop)

        Retorna
        -------
        dict : Reporte de validación con PASS/FAIL por criterio.
        """
        v = np.asarray(results["voltage"])
        i = np.asarray(results["current"])
        x = np.asarray(results["state_variable"])

        report = {
            "figure": "2b",
            "criteria": {},
            "metrics": {},
            "overall_pass": True,
        }

        # P1: Rango de corriente
        i_max = np.abs(i).max()
        i_expected = self.spec.i_max_expected
        tol = self.spec.i_max_tolerance
        p1_pass = abs(i_max - i_expected) / i_expected <= tol
        report["criteria"]["P1_current_range"] = {
            "pass": p1_pass,
            "expected_A": i_expected,
            "simulated_A": float(i_max),
            "error_pct": float(abs(i_max - i_expected) / i_expected * 100),
        }

        # P2: Histéresis visible
        area = hysteresis_area(v, i)
        p2_pass = area >= self.spec.min_hysteresis_area
        report["criteria"]["P2_hysteresis_visible"] = {
            "pass": p2_pass,
            "area_VA": float(area),
            "min_required_VA": self.spec.min_hysteresis_area,
        }

        # P3: Estado dentro de (0, 1) estricto
        x_min, x_max, delta_x = state_excursion(x)
        p3_pass = (x_min >= self.spec.x_min_threshold and
                   x_max <= self.spec.x_max_threshold)
        report["criteria"]["P3_state_bounded"] = {
            "pass": p3_pass,
            "x_min": float(x_min),
            "x_max": float(x_max),
            "delta_x": float(delta_x),
        }

        # P4: Corriente ~ 0 cuando V ~ 0 (pinched hysteresis)
        v_zero_mask = np.abs(v) < 0.05 * np.abs(v).max()
        i_at_vzero = np.abs(i[v_zero_mask]).mean() if v_zero_mask.any() else float("nan")
        p4_pass = i_at_vzero < 0.02 * i_max if not np.isnan(i_at_vzero) else False
        report["criteria"]["P4_pinched_at_origin"] = {
            "pass": p4_pass,
            "i_at_V0_A": float(i_at_vzero),
            "threshold_A": float(0.02 * i_max),
        }

        # P5: Asimetría del lazo (debe ser cercana a 0 para voltaje simétrico)
        i_pos = i[v > 0.1].mean() if (v > 0.1).any() else 0
        i_neg = i[v < -0.1].mean() if (v < -0.1).any() else 0
        asymmetry = abs(i_pos + i_neg) / (abs(i_pos) + abs(i_neg) + 1e-15)
        p5_pass = asymmetry < 0.15  # < 15 % de asimetría
        report["criteria"]["P5_loop_symmetry"] = {
            "pass": p5_pass,
            "asymmetry_ratio": float(asymmetry),
        }

        # Resumen de métricas
        report["metrics"] = {
            "i_max_mA": float(i_max * 1e3),
            "hysteresis_area": float(area),
            "x_excursion": float(delta_x),
            "collapse_index": float(hysteresis_collapse_index(v, i)),
        }

        # Resultado global
        report["overall_pass"] = all(
            c["pass"] for c in report["criteria"].values()
        )
        return report

    # ── Validación del colapso de frecuencia ─────────────────────────────

    def validate_frequency_collapse(self,
                                     results_low: Dict[str, np.ndarray],
                                     results_high: Dict[str, np.ndarray]) -> dict:
        """
        Verifica el teorema fundamental del paper: la histéresis se colapsa
        a una línea recta cuando la frecuencia aumenta 10×.

        El índice de colapso debe aumentar al menos 'frequency_collapse_threshold'
        (90 %) entre la frecuencia baja y la alta.
        """
        area_low = hysteresis_area(results_low["voltage"], results_low["current"])
        area_high = hysteresis_area(results_high["voltage"], results_high["current"])

        collapse_ratio = (1.0 - area_high / area_low) if area_low > 1e-20 else 0.0
        passes = collapse_ratio >= self.spec.frequency_collapse_threshold

        return {
            "pass": passes,
            "area_low_freq": float(area_low),
            "area_high_freq": float(area_high),
            "collapse_ratio": float(collapse_ratio),
            "threshold": self.spec.frequency_collapse_threshold,
        }

    # ── Reporte formateado ────────────────────────────────────────────────

    def print_report(self, report: dict):
        """Imprime el reporte de validación en formato legible."""
        fig = report.get("figure", "?")
        overall = report.get("overall_pass", False)
        status = "APROBADO" if overall else "REPROBADO"

        print(f"\n{'='*65}")
        print(f"  VALIDACION STRUKOV (2008) -- Figura {fig}")
        print(f"  Estado global: {status}")
        print(f"{'='*65}")

        for name, crit in report.get("criteria", {}).items():
            icon = "OK" if crit["pass"] else "FAIL"
            print(f"  [{icon}] {name}")
            for k, v in crit.items():
                if k != "pass":
                    print(f"       {k:<28} = {v}")

        print(f"\n  Metricas:")
        for k, v in report.get("metrics", {}).items():
            print(f"    {k:<35} = {v:.6f}")
        print(f"{'='*65}\n")
