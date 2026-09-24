"""
tests/test_lif_neuron.py
========================
Pruebas unitarias de la neurona LIF física basada en TSM.

Verifican que la neurona:
  - Inicializa sus estados físicos en 0 (Vc=0, w=0, Vout=0, R_tsm=R_off).
  - Carga el capacitor al recibir voltaje de entrada Vin.
  - Genera una transición de estado w y un spike de Vout cuando Vc supera V_th.
  - Descarga el capacitor de forma natural (física) tras conmutar a estado de baja resistencia.
  - Regresa al estado de alta resistencia cuando Vc cae por debajo de V_hold.
  - Se reinicia correctamente con reset().
"""

import numpy as np
import sys
import os

# Asegura que el paquete sea encontrado desde la raíz del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from memristor_simulator.models.lif_neuron import (
    LIFNeuron, LIFParameters, default_lif_params, fast_lif_params
)


# ── Tests de LIFParameters ────────────────────────────────────────────────────

class TestLIFParameters:
    def test_default_params_valid(self):
        """Los parámetros por defecto deben crearse sin excepción y ser coherentes."""
        p = default_lif_params()
        assert p.C > 0
        assert p.R_s > 0
        assert p.R_off > p.R_on
        assert p.V_th > p.V_hold

    def test_invalid_R_relation(self):
        """R_on >= R_off debe lanzar ValueError."""
        try:
            LIFParameters(R_on=1e6, R_off=1e3)
            assert False, "Debería haber lanzado ValueError"
        except ValueError:
            pass


# ── Tests de LIFNeuron — Estado inicial ──────────────────────────────────────

class TestLIFNeuronInitialState:
    def setup_method(self):
        self.params = default_lif_params()
        self.neuron = LIFNeuron(self.params)

    def test_initial_voltage_is_zero(self):
        """Vc inicial debe ser 0.0 V."""
        assert abs(self.neuron.V) < 1e-15

    def test_initial_w_is_zero(self):
        """w inicial del TSM debe ser 0.0 (totalmente OFF)."""
        assert abs(self.neuron.w) < 1e-15

    def test_initial_resistance_is_Roff(self):
        """La resistencia inicial de la neurona debe ser R_off."""
        assert abs(self.neuron.R_tsm - self.params.R_off) < 1e-15

    def test_initial_spike_count_zero(self):
        """La neurona comienza sin spikes."""
        assert self.neuron.spike_count == 0


# ── Tests de LIFNeuron — Dinámica de carga y descarga ─────────────────────────

class TestLIFNeuronDynamics:
    def setup_method(self):
        self.params = default_lif_params()
        self.neuron = LIFNeuron(self.params)
        self.dt = 0.05e-3  # 0.05 ms

    def test_vin_charges_capacitor(self):
        """Aplicar Vin = 5V debe incrementar Vc por encima de 0V."""
        _, V = self.neuron.step(5.0, self.dt)
        assert V > 0.0

    def test_subthreshold_no_switching(self):
        """Con Vin baja (p.ej. 0.5V), Vc se estabiliza por debajo de V_th y w no conmuta."""
        for _ in range(100):
            self.neuron.step(0.5, self.dt)
        assert self.neuron.V < self.params.V_th
        assert self.neuron.w < 0.1
        assert self.neuron.spike_count == 0

    def test_switching_occurs_with_high_vin(self):
        """Con Vin = 5.0 V constante, Vc cruza V_th, w conmuta a 1.0 y se detecta un spike."""
        fired_at_any_step = False
        w_max = 0.0
        for _ in range(5000):  # Simular suficiente tiempo
            fired, _ = self.neuron.step(5.0, self.dt)
            w_max = max(w_max, self.neuron.w)  # w oscila: sube a 1.0 al disparar y decae tras el reset
            if fired:
                fired_at_any_step = True
        assert fired_at_any_step
        assert self.neuron.spike_count >= 1
        assert w_max > 0.9

    def test_vout_peak_during_spike(self):
        """Durante la conmutación, Vout debe aproximarse a Vc * R0 / (Ron + R0)."""
        # Forzar conmutación activa
        self.neuron.V = 0.95
        self.neuron.w = 1.0
        self.neuron.step(5.0, self.dt)
        expected_vout = 0.95 * self.params.R_0 / (self.params.R_on + self.params.R_0)
        assert abs(self.neuron.Vout - expected_vout) < 0.05


# ── Tests de LIFNeuron — Reset ────────────────────────────────────────────────

class TestLIFNeuronReset:
    def setup_method(self):
        self.params = default_lif_params()
        self.neuron = LIFNeuron(self.params)
        self.dt = 0.05e-3

    def test_reset_restores_all_states(self):
        """reset() debe devolver Vc, w, R_tsm y spike_count a sus condiciones iniciales."""
        # Avanzar simulación hasta disparar
        for _ in range(3000):
            self.neuron.step(5.0, self.dt)
        
        assert self.neuron.V > 0.1
        self.neuron.reset()
        
        assert abs(self.neuron.V) < 1e-15
        assert abs(self.neuron.w) < 1e-15
        assert abs(self.neuron.R_tsm - self.params.R_off) < 1e-15
        assert self.neuron.spike_count == 0


# ── Ejecución Directa ─────────────────────────────────────────────────────────

def run_all_tests():
    test_classes = [
        TestLIFParameters,
        TestLIFNeuronInitialState,
        TestLIFNeuronDynamics,
        TestLIFNeuronReset,
    ]
    total = 0
    passed = 0
    failed = []

    for cls in test_classes:
        obj = cls()
        methods = sorted(m for m in dir(obj) if m.startswith('test_'))
        print(f"\n  [{cls.__name__}]")
        for method in methods:
            total += 1
            try:
                if hasattr(obj, 'setup_method'):
                    obj.setup_method()
                getattr(obj, method)()
                passed += 1
                print(f"    [OK]   {method}")
            except Exception as e:
                failed.append(f"{cls.__name__}::{method}")
                print(f"    [FAIL] {method}: {e}")

    print(f"\n{'='*55}")
    print(f"  Resultado: {passed}/{total} pruebas pasaron")
    if failed:
        print(f"  Fallaron:")
        for f in failed:
            print(f"    - {f}")
    else:
        print("  [PASS] Todas las pruebas pasaron correctamente.")
    print(f"{'='*55}")
    return len(failed) == 0


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  TEST SUITE — Neurona LIF Física (TSM)")
    print("="*55)
    ok = run_all_tests()
    sys.exit(0 if ok else 1)

