"""
tests/test_crossbar_module.py
==============================
Tests unitarios del módulo neurolab.crossbar.
"""

import numpy as np
import pytest
from neurolab.crossbar import Crossbar, CrossbarConfig, ProgrammingMode


class TestCrossbarInit:
    """Tests de inicialización."""
    
    def test_default_config(self):
        cb = Crossbar()
        assert cb.n_rows == 4
        assert cb.n_cols == 4
        assert cb.G_matrix.shape == (4, 4)
    
    def test_custom_size(self):
        cb = Crossbar(CrossbarConfig(n_rows=8, n_cols=6))
        assert cb.G_matrix.shape == (8, 6)
    
    def test_d2d_variability(self):
        cb = Crossbar(CrossbarConfig(G_sigma=0.08e-6, seed=42))
        G = cb.G_matrix
        assert G.std() > 0
        assert G.min() != G.max()
    
    def test_reproducibility(self):
        cb1 = Crossbar(CrossbarConfig(seed=123))
        cb2 = Crossbar(CrossbarConfig(seed=123))
        np.testing.assert_array_almost_equal(cb1.G_matrix, cb2.G_matrix)


class TestRead:
    """Tests de lectura."""
    
    def test_read_ideal_consistency(self):
        cb = Crossbar()
        V = np.array([0.1, 0.2, 0.1, 0.2])
        I1 = cb.read_ideal(V)
        I2 = cb.G_matrix.T @ V
        np.testing.assert_array_almost_equal(I1, I2)
    
    def test_read_zero_voltage(self):
        cb = Crossbar()
        V = np.zeros(4)
        I = cb.read_ideal(V)
        np.testing.assert_array_almost_equal(I, np.zeros(4))
    
    def test_read_linearity(self):
        cb = Crossbar()
        V1 = np.array([0.1, 0.1, 0.1, 0.1])
        V2 = 2 * V1
        I1 = cb.read_ideal(V1)
        I2 = cb.read_ideal(V2)
        np.testing.assert_array_almost_equal(I2, 2 * I1)


class TestProgramming:
    """Tests de programación."""
    
    def test_program_row(self):
        cb = Crossbar(CrossbarConfig(seed=42))
        G_before = cb.G_matrix.copy()
        cb.program_row(0, V_program=2.0, dt=1e-3)
        G_after = cb.G_matrix
        assert np.all(np.abs(G_after[0, :] - G_before[0, :]) > 0)
    
    def test_program_1T1R(self):
        cb = Crossbar(CrossbarConfig(seed=42))
        G_before = cb.G_matrix.copy()
        cb.program_1T1R(1, 1, V_program=2.0, dt=1e-3)
        G_after = cb.G_matrix
        delta = np.abs(G_after - G_before)
        assert delta[1, 1] > 0
        assert delta.sum() - delta[1, 1] < 1e-12
    
    def test_program_V2_selectivity(self):
        cb = Crossbar(CrossbarConfig(seed=42))
        G_before = cb.G_matrix.copy()
        cb.program_V2(1, 1, V_program=2.0, dt=1e-3)
        G_after = cb.G_matrix
        delta = np.abs(G_after - G_before)
        assert delta[1, 1] > 0
        assert delta[0, 0] < 1e-12
    
    def test_program_V2_cross_pattern(self):
        cb = Crossbar(CrossbarConfig(seed=42))
        G_before = cb.G_matrix.copy()
        cb.program_V2(1, 1, V_program=2.0, dt=1e-3)
        delta = np.abs(cb.G_matrix - G_before)
        assert np.all(delta[1, :] > 0)
        assert np.all(delta[:, 1] > 0)
        assert delta[0, 0] < 1e-12


class TestReset:
    """Tests de reset."""
    
    def test_reset_restores_d2d(self):
        cb = Crossbar(CrossbarConfig(seed=42))
        G_initial = cb.G_matrix.copy()
        cb.program_V2(1, 1, V_program=2.0, dt=1e-3)
        cb.reset()
        G_reset = cb.G_matrix
        np.testing.assert_array_almost_equal(G_initial, G_reset)


class TestSummary:
    """Tests de resumen."""
    
    def test_summary_keys(self):
        cb = Crossbar()
        s = cb.summary()
        assert 'mean' in s
        assert 'std' in s
        assert 'cv_percent' in s
        assert s['n_cells'] == 16


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
