import pytest
import numpy as np
from neurolab.validation.statistics import StatisticalAnalyzer


def test_statistical_analyzer_basic_metrics():
    """Verifica el cálculo de media, desviación estándar, CV y percentiles."""
    data = [10.0, 12.0, 14.0, 16.0, 18.0]
    analyzer = StatisticalAnalyzer(data, name="TestVar")
    s = analyzer.summary()

    assert s['name'] == "TestVar"
    assert s['n'] == 5
    assert pytest.approx(s['mean']) == 14.0
    assert pytest.approx(s['min']) == 10.0
    assert pytest.approx(s['max']) == 18.0
    assert pytest.approx(s['median']) == 14.0
    assert pytest.approx(s['p25']) == 12.0
    assert pytest.approx(s['p75']) == 16.0
    assert pytest.approx(s['iqr']) == 4.0


def test_confidence_interval():
    """Verifica el cálculo del intervalo de confianza Student's t al 95%."""
    rng = np.random.default_rng(42)
    data = rng.normal(loc=100.0, scale=5.0, size=100)
    analyzer = StatisticalAnalyzer(data, name="GaussianData")

    ci_lo, ci_hi = analyzer.confidence_interval(confidence=0.95)
    mean_val = np.mean(data)

    assert ci_lo < mean_val < ci_hi
    assert (ci_hi - ci_lo) > 0


def test_ks_and_shapiro_tests():
    """Verifica las pruebas de normalidad de Kolmogorov-Smirnov y Shapiro-Wilk."""
    rng = np.random.default_rng(42)
    normal_data = rng.normal(loc=0.0, scale=1.0, size=50)

    analyzer = StatisticalAnalyzer(normal_data, name="NormalSample")
    ks_res = analyzer.ks_test_normal()
    shapiro_res = analyzer.shapiro_test()

    assert ks_res['p_value'] > 0.01
    assert not ks_res['reject_normal']

    assert shapiro_res['p_value'] > 0.01
    assert not shapiro_res['reject_normal']


def test_report_formatting():
    """Verifica que el reporte en texto contenga todas las métricas esperadas."""
    data = np.linspace(1.0, 10.0, 20)
    analyzer = StatisticalAnalyzer(data, name="LinearRange")
    rep = analyzer.report()

    assert "=== LinearRange (n = 20) ===" in rep
    assert "Media (μ):" in rep
    assert "Desv. est. (σ):" in rep
    assert "CV:" in rep
    assert "KS p-valor:" in rep
