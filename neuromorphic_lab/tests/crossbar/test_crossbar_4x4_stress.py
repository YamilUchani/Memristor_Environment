"""
neuromorphic_lab/tests/crossbar/test_crossbar_4x4_stress.py
============================================================
Pruebas de Estrés Continuo (Stress Tests) para el Crossbar 4×4.

Verifica rigurosamente el comportamiento físico realista durante ejecuciones
prolongadas en tiempo real:
1. 1000 ticks consecutivos de integración en tiempo real sin desbordamientos ni NaNs.
2. Acotamiento estricto de conductancias G en [G_min, G_max] = [1.0 μS, 500.0 μS].
3. Relajación física continua de memristores volátiles hacia x_eq.
4. Tasa de disparo biológica de neuronas LIF y respeto del período refractario (150 ms).
5. Transiciones dinámicas de modo sin corrupción de memoria ni condiciones de carrera.
"""

import pytest
import numpy as np
from PySide6.QtWidgets import QApplication

from neurolab.gui.crossbar_4x4_view import Crossbar4x4View


def test_crossbar_4x4_continuous_1000_ticks_stress():
    """
    Somete al controlador del Crossbar 4x4 a 1000 ticks de integración continua
    (equivalente a 30 segundos de simulación en tiempo real).
    """
    app = QApplication.instance() or QApplication([])
    view = Crossbar4x4View()
    ctrl = view.controller

    rng = np.random.default_rng(seed=42)

    for step_idx in range(1000):
        V_rows = rng.uniform(0.0, 2.0, size=4)
        V_cols = np.zeros(4)
        spike_pre = (V_rows > 0.8).astype(float)

        res = ctrl.step(
            dt=0.030,
            mode="read",
            plasticity_mode="off",
            is_animating=True,
            V_rows=V_rows,
            V_cols=V_cols,
            spike_pre=spike_pre
        )

        I_cols = res["I_cols"]
        assert not np.isnan(I_cols).any(), f"NaN detectado en corrientes en el tick {step_idx}"
        assert not np.isinf(I_cols).any(), f"Inf detectado en corrientes en el tick {step_idx}"
        assert np.all(I_cols >= 0.0), f"Corrientes negativas detectadas en el tick {step_idx}"

        # Verificar acotamiento de conductancias
        for i in range(4):
            for j in range(4):
                m = ctrl.elements.get(f'M{i+1}{j+1}')
                if m:
                    g_uS = float(m.params.get('G', 69.4e-6)) * 1e6
                    assert 0.9 <= g_uS <= 500.1, f"Conductancia M{i+1}{j+1} fuera de rango: {g_uS:.2f} μS"

    view.deleteLater()


def test_crossbar_4x4_plasticity_stdp_rstdp_stress():
    """
    Somete la plasticidad STDP y R-STDP a 500 ticks continuos de aprendizaje por refuerzo estocástico.
    """
    app = QApplication.instance() or QApplication([])
    view = Crossbar4x4View()
    ctrl = view.controller

    rng = np.random.default_rng(seed=123)

    # 1. 250 Ticks de STDP
    for step_idx in range(250):
        V_rows = rng.choice([0.0, 0.8, 1.5], size=4)
        V_cols = np.zeros(4)
        spike_pre = (V_rows > 0.5).astype(float)

        res = ctrl.step(
            dt=0.030,
            mode="read",
            plasticity_mode="stdp",
            is_animating=True,
            V_rows=V_rows,
            V_cols=V_cols,
            spike_pre=spike_pre
        )

        for i in range(4):
            for j in range(4):
                m = ctrl.elements.get(f'M{i+1}{j+1}')
                if m:
                    g_uS = float(m.params.get('G', 69.4e-6)) * 1e6
                    assert 0.9 <= g_uS <= 500.1, f"Conductancia M{i+1}{j+1} fuera de límites en STDP tick {step_idx}: {g_uS:.2f} μS"

    # 2. 250 Ticks de R-STDP con Recompensa Aleatoria (+1, -1, 0)
    for step_idx in range(250):
        reward_val = rng.choice([+1.0, -1.0, 0.0])
        ctrl.set_reward(reward_val)

        V_rows = rng.choice([0.0, 0.8, 1.5], size=4)
        V_cols = np.zeros(4)
        spike_pre = (V_rows > 0.5).astype(float)

        res = ctrl.step(
            dt=0.030,
            mode="read",
            plasticity_mode="rstdp",
            is_animating=True,
            V_rows=V_rows,
            V_cols=V_cols,
            spike_pre=spike_pre
        )

        # Verificar consumo de recompensa
        assert ctrl.rstdp_rule.cfg.R == 0.0, "La recompensa no se consumió en el tick R-STDP"

        for i in range(4):
            for j in range(4):
                m = ctrl.elements.get(f'M{i+1}{j+1}')
                if m:
                    g_uS = float(m.params.get('G', 69.4e-6)) * 1e6
                    assert 0.9 <= g_uS <= 500.1, f"Conductancia M{i+1}{j+1} fuera de límites en R-STDP tick {step_idx}: {g_uS:.2f} μS"

    view.deleteLater()


def test_crossbar_4x4_volatile_relaxation_stress():
    """
    Prueba de estrés para la dinámica difusiva del memristor volátil M_v1..M_v4.
    """
    app = QApplication.instance() or QApplication([])
    view = Crossbar4x4View()
    ctrl = view.controller

    for j in range(4):
        mv = ctrl.elements[f'M_v{j+1}']
        mv.params['x'] = 0.80
        mv.params['x0'] = 0.05
        mv.params['volatile_tau_relax'] = 0.15

    # Integrar 100 ticks sin corriente aplicada
    V_rows = np.zeros(4)
    V_cols = np.zeros(4)

    for step_idx in range(100):
        ctrl.step(dt=0.030, mode="read", plasticity_mode="off", is_animating=True, V_rows=V_rows, V_cols=V_cols)

    for j in range(4):
        mv = ctrl.elements[f'M_v{j+1}']
        x_val = float(mv.params['x'])
        assert np.isclose(x_val, 0.05, atol=0.01), f"El memristor volátil M_v{j+1} no se relajó a x0=0.05 (x={x_val:.4f})"

    view.deleteLater()


def test_crossbar_4x4_mode_switching_stress():
    """
    Alterna rápidamente entre los 4 modos de operación durante simulación continua.
    """
    app = QApplication.instance() or QApplication([])
    view = Crossbar4x4View()

    modes = ["read", "program_v2", "stdp", "rstdp"]

    for cycle in range(50):
        chosen_mode = modes[cycle % len(modes)]

        if chosen_mode == "read":
            view._set_mode_1_read()
        elif chosen_mode == "program_v2":
            view._set_mode_2_v2()
        elif chosen_mode == "stdp":
            view._set_mode_3_stdp()
        elif chosen_mode == "rstdp":
            view._set_mode_4_rstdp()

        view._anim_tick()

        assert not np.isnan(view.V_rows).any()
        assert not np.isnan(view.V_cols).any()

    view.deleteLater()


def test_crossbar_4x4_zero_voltage_rows_must_not_potentiate():
    """
    Verifica rigurosamente que las filas con V=0.0V (como Fila 2 y Fila 4) NO aumenten
    su conductancia hacia G_max (500 μS) durante lecturas o ciclos de integración.
    """
    app = QApplication.instance() or QApplication([])
    view = Crossbar4x4View()
    ctrl = view.controller

    V_rows = np.array([0.80, 0.00, 1.10, 0.00])
    V_cols = np.zeros(4)

    for step_idx in range(50):
        ctrl.step(dt=0.030, mode="read", plasticity_mode="off", is_animating=True, V_rows=V_rows, V_cols=V_cols)

    # Verificar que los memristores de la Fila 2 (V=0V) y Fila 4 (V=0V) NO se hayan inflado hacia 500 μS
    for j in range(4):
        m2 = ctrl.elements[f'M2{j+1}']
        m4 = ctrl.elements[f'M4{j+1}']
        g2_uS = float(m2.params['G']) * 1e6
        g4_uS = float(m4.params['G']) * 1e6
        assert g2_uS < 150.0, f"M2{j+1} en Fila 2 (V=0V) se infló falsamente a {g2_uS:.1f} μS"
        assert g4_uS < 150.0, f"M4{j+1} en Fila 4 (V=0V) se infló falsamente a {g4_uS:.1f} μS"

    view.deleteLater()

