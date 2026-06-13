"""
tests/test_lif_neuron.py
========================
Pruebas unitarias de la neurona Leaky Integrate-and-Fire (LIF).

Verifican que la neurona:
  - Recibe corriente correctamente.
  - Acumula carga y sube el potencial.
  - Dispara cuando V supera V_th.
  - Reinicia a V_reset después del spike.
  - Respeta el período refractario.
  - No dispara con input insuficiente.
  - Mantiene V cerca de E_L sin input.

Ejecutar con:
    python -m pytest memristor_simulator/tests/test_lif_neuron.py -v
    python memristor_simulator/tests/test_lif_neuron.py
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
        """Los parámetros por defecto deben crear sin excepción."""
        p = default_lif_params()
        assert p.tau_m > 0
        assert p.R_m > 0
        assert p.V_th > p.E_L
        assert p.V_reset < p.V_th

    def test_C_m_derived(self):
        """C_m debe ser τ_m / R_m."""
        p = default_lif_params()
        expected = p.tau_m / p.R_m
        assert abs(p.C_m - expected) < 1e-20

    def test_invalid_Vreset_above_Vth(self):
        """V_reset >= V_th debe lanzar ValueError."""
        try:
            LIFParameters(V_reset=-40e-3, V_th=-50e-3)
            assert False, "Debería haber lanzado ValueError"
        except ValueError:
            pass

    def test_invalid_EL_above_Vth(self):
        """E_L >= V_th debe lanzar ValueError."""
        try:
            LIFParameters(E_L=-45e-3, V_th=-50e-3)
            assert False, "Debería haber lanzado ValueError"
        except ValueError:
            pass

    def test_fast_params_valid(self):
        """Los parámetros fast deben ser físicamente coherentes."""
        p = fast_lif_params()
        assert p.V_th > p.E_L
        assert p.V_reset < p.V_th
        assert p.t_ref >= 0


# ── Tests de LIFNeuron — Estado inicial ──────────────────────────────────────

class TestLIFNeuronInitialState:
    def setup_method(self):
        self.params = default_lif_params()
        self.neuron = LIFNeuron(self.params)

    def test_initial_voltage_is_EL(self):
        """Sin V_init, el potencial inicial debe ser E_L (reposo)."""
        assert abs(self.neuron.V - self.params.E_L) < 1e-15

    def test_initial_spike_count_zero(self):
        """La neurona comienza sin spikes."""
        assert self.neuron.spike_count == 0

    def test_initial_not_refractory(self):
        """La neurona empieza fuera del período refractario."""
        assert not self.neuron.is_refractory

    def test_custom_V_init(self):
        """V_init personalizado debe ser respetado."""
        V0 = -60e-3
        n = LIFNeuron(self.params, V_init=V0)
        assert abs(n.V - V0) < 1e-15

    def test_membrane_potential_mV(self):
        """La propiedad mV debe devolver V * 1000."""
        assert abs(self.neuron.membrane_potential_mV - self.params.E_L * 1e3) < 1e-9


# ── Tests de LIFNeuron — Integración (recepción de corriente) ─────────────────

class TestLIFNeuronIntegration:
    def setup_method(self):
        self.params = default_lif_params()
        self.neuron = LIFNeuron(self.params)
        self.dt = 0.1e-3  # 0.1 ms

    def test_positive_current_raises_voltage(self):
        """Corriente excitadora debe aumentar V por encima de E_L."""
        I_exc = 2e-9   # 2 nA
        _, V = self.neuron.step(I_exc, self.dt)
        assert V > self.params.E_L

    def test_zero_current_stays_near_EL(self):
        """Sin corriente, V debe permanecer en E_L (ya está en reposo)."""
        for _ in range(100):
            _, V = self.neuron.step(0.0, self.dt)
        assert abs(V - self.params.E_L) < 1e-9

    def test_negative_current_lowers_voltage(self):
        """Corriente inhibidora debe bajar V por debajo de E_L."""
        # Primero subir un poco
        self.neuron.V = -60e-3
        I_inh = -2e-9  # -2 nA
        _, V = self.neuron.step(I_inh, self.dt)
        assert V < -60e-3

    def test_step_returns_tuple(self):
        """step() debe retornar (bool, float)."""
        result = self.neuron.step(1e-9, self.dt)
        assert isinstance(result, tuple) and len(result) == 2
        fired, V = result
        assert isinstance(fired, bool)
        assert isinstance(V, float)

    def test_charge_accumulates(self):
        """La carga acumulada debe aumentar con cada paso."""
        I = 1e-9
        for _ in range(10):
            self.neuron.step(I, self.dt)
        assert self.neuron.charge_accumulated > 0

    def test_voltage_exponential_decay(self):
        """Sin input, V debe decaer hacia E_L exponencialmente."""
        self.neuron.V = -55e-3  # Por encima del reposo
        V_prev = self.neuron.V
        for _ in range(50):
            _, V = self.neuron.step(0.0, self.dt)
            # Cada paso debe acercarse más a E_L
        assert abs(V - self.params.E_L) < abs(V_prev - self.params.E_L)


# ── Tests de LIFNeuron — Disparo (spike) ─────────────────────────────────────

class TestLIFNeuronSpiking:
    def setup_method(self):
        self.params = default_lif_params()
        self.neuron = LIFNeuron(self.params)
        self.dt = 0.1e-3

    def test_fires_with_strong_current(self):
        """Con corriente suficiente, debe disparar eventualmente."""
        I_strong = 5e-9   # 5 nA — supera el umbral
        fired_at_any_step = False
        for _ in range(int(0.2 / self.dt)):  # 200 ms de simulación
            fired, _ = self.neuron.step(I_strong, self.dt)
            if fired:
                fired_at_any_step = True
                break
        assert fired_at_any_step, "La neurona debería disparar con 5 nA"

    def test_no_fire_with_weak_current(self):
        """Con corriente débil, no debe disparar en 500 ms."""
        I_weak = 0.1e-9   # 0.1 nA — insuficiente para llegar a V_th
        for _ in range(int(0.5 / self.dt)):
            fired, _ = self.neuron.step(I_weak, self.dt)
            assert not fired, "La neurona no debería disparar con 0.1 nA"

    def test_spike_resets_voltage(self):
        """Después del spike, V debe ser V_reset."""
        I_strong = 5e-9
        for _ in range(int(0.2 / self.dt)):
            fired, V = self.neuron.step(I_strong, self.dt)
            if fired:
                assert abs(V - self.params.V_reset) < 1e-15
                break

    def test_spike_increments_counter(self):
        """Cada spike debe incrementar spike_count en 1."""
        I_strong = 5e-9
        count_prev = self.neuron.spike_count
        for _ in range(int(0.2 / self.dt)):
            fired, _ = self.neuron.step(I_strong, self.dt)
            if fired:
                assert self.neuron.spike_count == count_prev + 1
                break

    def test_multiple_spikes_counted(self):
        """Con input sostenido, debe generar múltiples spikes."""
        I_strong = 5e-9
        for _ in range(int(1.0 / self.dt)):  # 1 segundo
            self.neuron.step(I_strong, self.dt)
        assert self.neuron.spike_count >= 3, (
            f"Con input sostenido esperamos ≥3 spikes, got {self.neuron.spike_count}")

    def test_voltage_never_exceeds_Vth_after_step(self):
        """El potencial nunca debe quedar por encima de V_th tras un step."""
        I_strong = 10e-9  # corriente muy grande
        for _ in range(int(0.5 / self.dt)):
            _, V = self.neuron.step(I_strong, self.dt)
            assert V <= self.params.V_th, (
                f"V={V*1e3:.2f} mV superó V_th={self.params.V_th*1e3:.2f} mV")


# ── Tests de LIFNeuron — Período refractario ─────────────────────────────────

class TestLIFNeuronRefractory:
    def setup_method(self):
        self.params = default_lif_params()
        self.neuron = LIFNeuron(self.params)
        self.dt = 0.1e-3

    def _force_spike(self):
        """Fuerza un spike aumentando V directamente."""
        self.neuron.V = self.params.V_th + 0.001  # justo sobre el umbral
        fired, _ = self.neuron.step(0.0, self.dt)
        assert fired, "El spike forzado debería ocurrir"

    def test_is_refractory_after_spike(self):
        """La neurona debe estar en refractario inmediatamente después del spike."""
        self._force_spike()
        assert self.neuron.is_refractory

    def test_voltage_clamped_during_refractory(self):
        """Durante el refractario, V se mantiene en V_reset."""
        self._force_spike()
        I_huge = 100e-9  # corriente enorme
        for _ in range(int(self.params.t_ref / self.dt) - 1):
            _, V = self.neuron.step(I_huge, self.dt)
            assert abs(V - self.params.V_reset) < 1e-15, (
                "V debe ser V_reset durante el período refractario")

    def test_no_spike_during_refractory(self):
        """No puede disparar durante el período refractario."""
        self._force_spike()
        I_huge = 100e-9
        n_ref_steps = int(self.params.t_ref / self.dt)
        for _ in range(n_ref_steps - 1):
            fired, _ = self.neuron.step(I_huge, self.dt)
            assert not fired, "No debe disparar durante el refractario"

    def test_not_refractory_after_t_ref(self):
        """Después de t_ref, la neurona debe poder disparar de nuevo."""
        self._force_spike()
        n_ref_steps = int(self.params.t_ref / self.dt) + 5  # +5 pasos de margen
        for _ in range(n_ref_steps):
            self.neuron.step(0.0, self.dt)
        assert not self.neuron.is_refractory


# ── Tests de LIFNeuron — Reset ────────────────────────────────────────────────

class TestLIFNeuronReset:
    def setup_method(self):
        self.params = default_lif_params()
        self.neuron = LIFNeuron(self.params)
        self.dt = 0.1e-3

    def test_reset_restores_voltage(self):
        """reset() debe devolver V a E_L."""
        # Excitar la neurona
        for _ in range(100):
            self.neuron.step(5e-9, self.dt)
        self.neuron.reset()
        assert abs(self.neuron.V - self.params.E_L) < 1e-15

    def test_reset_clears_spike_count(self):
        """reset() debe poner spike_count a 0."""
        for _ in range(int(1.0 / self.dt)):
            self.neuron.step(5e-9, self.dt)
        assert self.neuron.spike_count > 0
        self.neuron.reset()
        assert self.neuron.spike_count == 0

    def test_reset_clears_refractory(self):
        """reset() debe sacar a la neurona del estado refractario."""
        self.neuron.V = self.params.V_th + 0.001
        self.neuron.step(0.0, self.dt)  # disparo forzado
        assert self.neuron.is_refractory
        self.neuron.reset()
        assert not self.neuron.is_refractory

    def test_reset_with_custom_V_init(self):
        """reset(V_init) debe usar el valor personalizado."""
        V0 = -58e-3
        self.neuron.reset(V_init=V0)
        assert abs(self.neuron.V - V0) < 1e-15


# ── Tests de LIFNeuron — Consistencia física ──────────────────────────────────

class TestLIFNeuronPhysics:
    def setup_method(self):
        self.params = default_lif_params()
        self.neuron = LIFNeuron(self.params)
        self.dt = 0.1e-3

    def test_dvdt_at_rest_zero(self):
        """En reposo sin corriente, dV/dt debe ser 0."""
        dv = self.neuron.dvdt(self.params.E_L, 0.0)
        assert abs(dv) < 1e-15

    def test_dvdt_positive_with_current(self):
        """Con corriente positiva, dV/dt debe ser positiva."""
        dv = self.neuron.dvdt(self.params.E_L, 2e-9)
        assert dv > 0

    def test_firing_rate_increases_with_current(self):
        """Mayor corriente → mayor frecuencia de disparo (relación f-I)."""
        def count_spikes(I_amp, T=1.0):
            n = LIFNeuron(self.params)
            for _ in range(int(T / self.dt)):
                n.step(I_amp, self.dt)
            return n.spike_count

        spikes_low  = count_spikes(2e-9)
        spikes_high = count_spikes(8e-9)
        assert spikes_high > spikes_low, (
            "Mayor corriente debe producir más spikes (curva f-I)")

    def test_tau_affects_integration_speed(self):
        """Mayor τ_m → la neurona integra más lento → tarda más en disparar."""
        from memristor_simulator.models.lif_neuron import LIFParameters

        p_slow = LIFParameters(tau_m=40e-3)
        p_fast = LIFParameters(tau_m=10e-3)

        def time_to_first_spike(params, I_amp=3e-9, T=1.0):
            n = LIFNeuron(params)
            for step_i in range(int(T / self.dt)):
                fired, _ = n.step(I_amp, self.dt)
                if fired:
                    return step_i * self.dt
            return T  # no disparó

        t_slow = time_to_first_spike(p_slow)
        t_fast = time_to_first_spike(p_fast)
        assert t_fast < t_slow, (
            "Neurona rápida (τ pequeño) debe disparar antes que neurona lenta")


# ── Punto de entrada para ejecución directa ───────────────────────────────────

def run_all_tests():
    """Ejecuta todas las pruebas sin pytest (ejecución directa)."""
    test_classes = [
        TestLIFParameters,
        TestLIFNeuronInitialState,
        TestLIFNeuronIntegration,
        TestLIFNeuronSpiking,
        TestLIFNeuronRefractory,
        TestLIFNeuronReset,
        TestLIFNeuronPhysics,
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
    print("  TEST SUITE — Neurona LIF (Leaky Integrate-and-Fire)")
    print("="*55)
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
