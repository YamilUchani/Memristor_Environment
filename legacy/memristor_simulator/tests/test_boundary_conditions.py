"""
tests/test_boundary_conditions.py
===================================
Pruebas de condiciones de frontera: x in [0, 1] bajo escenarios extremos.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from memristor_simulator.config.parameters import fig2b_params, StrukovParameters
from memristor_simulator.models.strukov_model import StrukovMemristor


class TestBoundaryConditions:
    def _make_device(self, w_init=0.5, nonlinear=True, hard=True):
        p = fig2b_params()
        p.w_init = w_init
        p.enable_nonlinear_drift = nonlinear
        p.enable_hard_switching = hard
        return StrukovMemristor(p)

    def test_no_negative_state(self):
        """x nunca baja de 0 bajo voltaje negativo fuerte."""
        device = self._make_device(w_init=0.1)
        for _ in range(5000):
            device.step(-5.0, 1e-3)
        assert device.x >= 0.0

    def test_no_state_above_one(self):
        """x nunca supera 1 bajo voltaje positivo fuerte."""
        device = self._make_device(w_init=0.9)
        for _ in range(5000):
            device.step(5.0, 1e-3)
        assert device.x <= 1.0

    def test_symmetry_ac_excitation(self):
        """Excitacion AC simetrica debe mantener x en [0, 1]."""
        device = self._make_device(w_init=0.5)
        for k in range(10000):
            t = k * 1e-4
            v = np.sin(2 * np.pi * 0.5 * t)
            device.step(float(v), 1e-4)
        # La ventana Biolek previene saturacion; x debe permanecer fisicamente valido
        assert 0.0 <= device.x <= 1.0

    def test_hard_switching_at_upper_boundary(self):
        """Con hard switching activado, x se fija en 1 al llegar al borde."""
        device = self._make_device(w_init=0.99, nonlinear=False, hard=True)
        for _ in range(1000):
            device.step(10.0, 1e-3)
        assert device.x == 1.0

    def test_resistance_never_zero(self):
        """La resistencia nunca debe ser 0 o negativa."""
        device = self._make_device()
        for _ in range(1000):
            device.step(5.0, 1e-3)
            assert device.resistance > 0

    def test_x_init_zero(self):
        """Dispositivo totalmente en OFF (x=0): R = R_off."""
        p = fig2b_params()
        p.w_init = 0.0
        d = StrukovMemristor(p)
        assert abs(d.resistance - p.R_off) < 1e-6

    def test_x_init_one(self):
        """Dispositivo totalmente en ON (x=1): R = R_on."""
        p = fig2b_params()
        p.w_init = 1.0
        d = StrukovMemristor(p)
        assert abs(d.resistance - p.R_on) < 1e-6


def run_all_tests():
    cls = TestBoundaryConditions()
    methods = [m for m in dir(cls) if m.startswith('test_')]
    total = len(methods)
    passed = 0
    for m in methods:
        try:
            getattr(cls, m)()
            print(f"  [OK] {m}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {m}: {e}")
    print(f"\n  Resultado: {passed}/{total} pruebas pasaron")
    return passed == total


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  TEST SUITE — Condiciones de Frontera")
    print("="*55)
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
