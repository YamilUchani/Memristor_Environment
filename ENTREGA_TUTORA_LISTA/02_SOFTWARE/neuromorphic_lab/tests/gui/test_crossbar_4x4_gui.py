"""
test_crossbar_4x4_gui.py
========================
Tests de integración para Crossbar4x4View en tiempo real.
Verifica que los sensores con voltaje 0.0V NO emitan spikes ni causen
inflación indeseada de conductancia a ~500uS en sus filas correspondientes.
"""

import sys
import pytest
import numpy as np
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


from neurolab.gui.crossbar_4x4_view import Crossbar4x4View


def test_crossbar_4x4_gui_initialization(qapp):
    """Verifica que la vista Crossbar4x4View inicialice correctamente todos los elementos."""
    view = Crossbar4x4View()
    assert 'S1' in view.elements
    assert 'S4' in view.elements
    assert 'M11' in view.elements
    assert 'M44' in view.elements
    assert 'M_v1' in view.elements
    assert 'LIF_1' in view.elements
    assert 'Act_1' in view.elements
    view.close()
    view.deleteLater()


def test_crossbar_4x4_zero_voltage_rows_do_not_potentiate_in_stdp(qapp):
    """
    AUDITORÍA GUI: En STDP activo, si S1 (+0.80V) y S3 (+1.10V) están activos,
    pero S2 (0.00V) y S4 (0.00V) están inactivos (0V),
    las filas 2 y 4 NO deben inflarse hacia G_max (500 μS).
    Deben mantenerse cercanas a la conductancia base (69.4 μS).
    """
    view = Crossbar4x4View()

    # Configurar sensores como en el escenario reportado
    view.elements['S1'].params['V_out'] = 0.80
    view.elements['S2'].params['V_out'] = 0.00  # Inactivo
    view.elements['S3'].params['V_out'] = 1.10
    view.elements['S4'].params['V_out'] = 0.00  # Inactivo

    # Activar STDP y simulación en tiempo real
    view._set_mode_3_stdp()
    view.canvas.is_animating = True

    # Ejecutar 100 ticks de animación (equivalente a 3 segundos de ejecucion)
    for _ in range(100):
        view._anim_tick()

    g_final_r2 = [float(view.elements[f'M2{j+1}'].params.get('G', 69.4e-6)) * 1e6 for j in range(4)]
    g_final_r4 = [float(view.elements[f'M4{j+1}'].params.get('G', 69.4e-6)) * 1e6 for j in range(4)]

    # Las conductancias de la Fila 2 (S2=0V) deben estar cerca de la base (69.4 uS), NUNCA infladas a 500 uS!
    for j in range(4):
        assert g_final_r2[j] < 100.0, (
            f"Error: M2{j+1} en fila inactiva V=0V se infló a {g_final_r2[j]:.1f} μS!"
        )

    # Las conductancias de la Fila 4 (S4=0V) deben estar cerca de la base (69.4 uS), NUNCA infladas a 500 uS!
    for j in range(4):
        assert g_final_r4[j] < 100.0, (
            f"Error: M4{j+1} en fila inactiva V=0V se infló a {g_final_r4[j]:.1f} μS!"
        )

    view.close()
    view.deleteLater()


def test_crossbar_4x4_read_mode_does_not_inflate_conductance(qapp):
    """
    AUDITORÍA GUI: En Modo 1 (Lectura No Destructiva), con sensores por defecto (S1=0.8V, S2=0.4V, S3=0.3V, S4=0.2V),
    las conductancias de la matriz NO deben inflarse a ~500 μS al simular ticks de animación.
    Deben mantenerse constantes cerca de sus valores iniciales (~69.4 μS).
    """
    view = Crossbar4x4View()
    view._set_mode_1_read()
    view.canvas.is_animating = True

    g_initial = [float(view.elements[f'M1{j+1}'].params.get('G', 69.4e-6)) * 1e6 for j in range(4)]

    # Ejecutar 100 ticks de animación
    for _ in range(100):
        view._anim_tick()

    g_final_r1 = [float(view.elements[f'M1{j+1}'].params.get('G', 69.4e-6)) * 1e6 for j in range(4)]

    for j in range(4):
        assert abs(g_final_r1[j] - g_initial[j]) < 1.0, (
            f"Error: M1{j+1} en Lectura No Destructiva cambió de {g_initial[j]:.1f} a {g_final_r1[j]:.1f} μS!"
        )

    view.close()
    view.deleteLater()


def test_read_mode_uses_core_physics(qapp):
    """
    FASE 1 VERIFICACIÓN: Verifica que el controlador en modo lectura pase por la física del core.
    Al cambiar enable_sneak_paths en el core, la corriente leída DEBE cambiar.
    """
    view = Crossbar4x4View()
    ctrl = view.controller

    V_rows = np.array([1.0, 0.5, 0.5, 0.5])
    V_cols = np.zeros(4)

    # 1. Sin sneak paths
    ctrl.crossbar.cfg.enable_sneak_paths = False
    ctrl.crossbar.cfg.R_sneak_factor = 0.5
    res_no_sneak = ctrl.step(dt=0.03, mode="read", plasticity_mode="off", is_animating=True, V_rows=V_rows, V_cols=V_cols)
    I_without_sneak = res_no_sneak["I_cols"]

    # 2. Con sneak paths
    ctrl.crossbar.cfg.enable_sneak_paths = True
    res_sneak = ctrl.step(dt=0.03, mode="read", plasticity_mode="off", is_animating=True, V_rows=V_rows, V_cols=V_cols)
    I_with_sneak = res_sneak["I_cols"]

    assert not np.allclose(I_with_sneak, I_without_sneak), (
        "El modo read ignora enable_sneak_paths (no está enrutando por la física del core)"
    )

    view.close()
    view.deleteLater()


