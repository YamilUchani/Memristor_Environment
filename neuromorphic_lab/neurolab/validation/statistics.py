"""
neurolab/validation/statistics.py
==================================
Módulo de análisis estadístico para validación de variabilidad y estocasticidad.
Calcula: μ, σ, CV%, percentiles, intervalo de confianza (Student's t), KS-test y Shapiro-Wilk.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple, Optional


class StatisticalAnalyzer:
    """Analizador estadístico de conjuntos de resultados unidimensionales."""

    def __init__(self, data: Any, name: str = "magnitud"):
        """
        Parameters
        ----------
        data : array-like
            Conjunto de valores (1D).
        name : str
            Nombre de la magnitud para reportes.
        """
        self.data = np.asarray(data, dtype=float).ravel()
        self.name = name
        self.n = len(self.data)
        if self.n == 0:
            raise ValueError("El conjunto de datos no puede estar vacío.")

    def summary(self) -> Dict[str, Any]:
        """Retorna un diccionario con los estadísticos fundamentales."""
        mean_val = float(np.mean(self.data))
        std_val = float(np.std(self.data, ddof=1)) if self.n > 1 else 0.0
        cv_val = (std_val / mean_val * 100.0) if mean_val != 0 else 0.0
        p25 = float(np.percentile(self.data, 25))
        p75 = float(np.percentile(self.data, 75))

        return {
            'name': self.name,
            'n': self.n,
            'mean': mean_val,
            'std': std_val,
            'cv_percent': cv_val,
            'min': float(np.min(self.data)),
            'max': float(np.max(self.data)),
            'median': float(np.median(self.data)),
            'p25': p25,
            'p75': p75,
            'iqr': p75 - p25,
        }

    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Intervalo de confianza para la media (t-Student)."""
        if self.n <= 1:
            val = float(self.data[0])
            return val, val

        mean_val = np.mean(self.data)
        se = stats.sem(self.data)
        h = se * stats.t.ppf((1.0 + confidence) / 2.0, self.n - 1)
        return float(mean_val - h), float(mean_val + h)

    def ks_test_normal(self, mu: Optional[float] = None, sigma: Optional[float] = None) -> Dict[str, Any]:
        """
        Test de Kolmogorov-Smirnov contra distribución normal.
        Si mu/sigma son None, se estiman a partir de los datos.
        """
        if mu is None:
            mu = float(np.mean(self.data))
        if sigma is None:
            sigma = float(np.std(self.data, ddof=1)) if self.n > 1 else 1.0

        if sigma == 0:
            return {'ks_stat': 0.0, 'p_value': 1.0, 'reject_normal': False}

        stat, p_value = stats.kstest(self.data, 'norm', args=(mu, sigma))
        return {
            'ks_stat': float(stat),
            'p_value': float(p_value),
            'reject_normal': bool(p_value < 0.05)
        }

    def shapiro_test(self) -> Dict[str, Any]:
        """Test de Shapiro-Wilk de normalidad (más potente para n < 50)."""
        if self.n < 3:
            return {'shapiro_stat': 0.0, 'p_value': 1.0, 'reject_normal': False}

        stat, p_value = stats.shapiro(self.data)
        return {
            'shapiro_stat': float(stat),
            'p_value': float(p_value),
            'reject_normal': bool(p_value < 0.05)
        }

    def report(self) -> str:
        """Genera un reporte formateado en texto estructurado."""
        s = self.summary()
        ci_lo, ci_hi = self.confidence_interval()
        ks = self.ks_test_normal()
        lines = [
            f"=== {s['name']} (n = {s['n']}) ===",
            f"  Media (μ):        {s['mean']:.6f}",
            f"  Desv. est. (σ):   {s['std']:.6f}",
            f"  CV:               {s['cv_percent']:.4f} %",
            f"  Mín / Máx:        {s['min']:.6f} / {s['max']:.6f}",
            f"  Mediana:          {s['median']:.6f}",
            f"  IQR:              {s['iqr']:.6f}",
            f"  IC 95% media:     [{ci_lo:.6f}, {ci_hi:.6f}]",
            f"  KS p-valor:       {ks['p_value']:.4f} ({'NO normal' if ks['reject_normal'] else 'normal'})",
        ]
        return "\n".join(lines)
