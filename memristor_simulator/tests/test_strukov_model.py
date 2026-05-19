"""
tests/test_strukov_model.py
============================
Pruebas unitarias del nucleo fisico del modelo Strukov (2008).

Ejecutar con: python -m pytest memristor_simulator/tests/ -v
"""

import numpy as np
import sys
import os

# Asegura que el paquete sea encontrado desde la raiz del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from memristor_simulator.config.parameters import (
    StrukovParameters, fig2b_params, fig2c_params)
from memristor_simulator.models.strukov_model import (
    StrukovMemristor, calculate_resistance, dxdt_strukov, window_biolek)


# ── Funciones puras ───────────────────────────────────────────────────────────

class TestCalculateResistance:
    def test_at_x0_equals_roff(self):
        """x=0 → solo region no dopada → R = R_off."""
        assert calculate_resistance(0.0, 100.0, 16000.0) == 16000.0

    def test_at_x1_equals_ron(self):
        """x=1 → totalmente dopado → R = R_on."""
        assert calculate_resistance(1.0, 100.0, 16000.0) == 100.0

    def test_at_x05_is_mean(self):
        """x=0.5 → promedio ponderado."""
        r = calculate_resistance(0.5, 100.0, 16000.0)
        assert abs(r - 8050.0) < 1e-9

    def test_monotonic(self):
        """R debe ser monotonamente decreciente con x."""
        xs = np.linspace(0, 1, 100)
        rs = [calculate_resistance(x, 100.0, 16000.0) for x in xs]
        assert all(rs[i] >= rs[i+1] for i in range(len(rs)-1))


class TestDxdtStrukov:
    def test_positive_current_increases_x(self):
        """Corriente positiva → dopantes se mueven → dx/dt > 0."""
        dxdt = dxdt_strukov(0.01, 100.0, 10e-9, 1e-14)
        assert dxdt > 0

    def test_zero_current_zero_dxdt(self):
        """Sin corriente no hay movimiento de dopantes."""
        dxdt = dxdt_strukov(0.0, 100.0, 10e-9, 1e-14)
        assert dxdt == 0.0

    def test_beta_from_paper(self):
        """
        Verifica beta = mu_v * R_on / D^2.
        Paper: mu_v=1e-14, R_on=100, D=10e-9
        beta = 1e-14 * 100 / (10e-9)^2 = 1e-12 / 1e-16 = 1e4 s^-1 A^-1
        """
        beta = (1e-14 * 100.0) / (10e-9)**2
        assert abs(beta - 1e4) < 1.0


class TestWindowBiolek:
    def test_at_center_is_one(self):
        """En el centro x=0.5 la ventana vale 1."""
        assert abs(window_biolek(0.5) - 1.0) < 1e-9

    def test_at_boundary_near_zero(self):
        """En los bordes la ventana tiende a 0."""
        assert window_biolek(0.0) < 0.01
        assert window_biolek(1.0) < 0.01

    def test_always_nonnegative(self):
        """La ventana nunca es negativa en [0,1]."""
        xs = np.linspace(0, 1, 200)
        assert all(window_biolek(x) >= 0 for x in xs)


# ── Clase StrukovMemristor ────────────────────────────────────────────────────

class TestStrukovMemristor:
    def setup_method(self):
        self.params = fig2b_params()
        self.device = StrukovMemristor(self.params)

    def test_initial_state(self):
        """Estado inicial = w_init."""
        assert self.device.x == self.params.w_init

    def test_initial_resistance(self):
        """Resistencia inicial consistente con x_0."""
        expected = calculate_resistance(
            self.params.w_init, self.params.R_on, self.params.R_off)
        assert abs(self.device.resistance - expected) < 1e-6

    def test_step_returns_current(self):
        """step() debe retornar corriente (float)."""
        i = self.device.step(1.0, 1e-4)
        assert isinstance(i, (float, np.floating))

    def test_current_sign_matches_voltage(self):
        """Corriente y voltaje deben tener el mismo signo."""
        i_pos = self.device.step(1.0, 1e-4)
        assert i_pos > 0
        self.device.reset()
        i_neg = self.device.step(-1.0, 1e-4)
        assert i_neg < 0

    def test_state_bounded(self):
        """x debe permanecer en [0, 1] bajo excitacion fuerte."""
        for _ in range(10000):
            self.device.step(10.0, 1e-3)  # voltaje grande
        assert 0.0 <= self.device.x <= 1.0

    def test_reset_restores_initial(self):
        """reset() debe restaurar el estado inicial."""
        for _ in range(100):
            self.device.step(1.0, 1e-4)
        self.device.reset()
        assert abs(self.device.x - self.params.w_init) < 1e-10

    def test_ohmic_limit_zero_voltage(self):
        """Con voltaje cero no debe fluir corriente."""
        i = self.device.step(0.0, 1e-4)
        assert abs(i) < 1e-15

    def test_memristance_between_ron_roff(self):
        """La resistencia debe estar entre R_on y R_off."""
        for v in np.sin(np.linspace(0, 2*np.pi, 100)):
            self.device.step(float(v), 1e-4)
        assert self.params.R_on <= self.device.resistance <= self.params.R_off


# ── Punto de entrada para ejecucion directa ──────────────────────────────────

def run_all_tests():
    """Ejecuta todas las pruebas sin pytest (para ejecucion directa)."""
    test_classes = [
        TestCalculateResistance,
        TestDxdtStrukov,
        TestWindowBiolek,
        TestStrukovMemristor,
    ]
    total = 0
    passed = 0
    failed = []

    for cls in test_classes:
        obj = cls()
        methods = [m for m in dir(obj) if m.startswith('test_')]
        for method in methods:
            total += 1
            try:
                if hasattr(obj, 'setup_method'):
                    obj.setup_method()
                getattr(obj, method)()
                passed += 1
                print(f"  [OK] {cls.__name__}::{method}")
            except Exception as e:
                failed.append(f"{cls.__name__}::{method}")
                print(f"  [FAIL] {cls.__name__}::{method}: {e}")

    print(f"\n  Resultado: {passed}/{total} pruebas pasaron")
    if failed:
        print(f"  Fallaron: {failed}")
    return len(failed) == 0


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  TEST SUITE — Modelo Strukov (2008)")
    print("="*55)
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
