"""
test_validation_loader.py
===========================
Prueba unitaria para la carga y renderizado de los datos CSV de validación
(CvsT.csv, VvsT.csv, WDvsT.csv) en Neuromorphic Lab.
"""

import numpy as np
import pytest
from neurolab.io.validation_loader import ValidationDataLoader
from neurolab.gui.widgets.plot_canvas import MplCanvas


def test_validation_data_loader():
    loader = ValidationDataLoader()
    data = loader.load_all()

    assert data is not None, "El cargador de datos de validación devolvió None."
    required_keys = [
        "t_i", "i_val_mA", "i_val", "t_v", "v_val", "t_w", "wd_val", "v_interp",
        "r_val_kohm", "g_val_us"
    ]
    for key in required_keys:
        assert key in data, f"Falta la clave '{key}' en los datos de validación."
        assert len(data[key]) > 0, f"El vector '{key}' está vacío."
        assert isinstance(data[key], np.ndarray), f"El elemento '{key}' debe ser un ndarray."

    # Verificar que los datos no contengan NaNs ni Infs
    assert not np.isnan(data["i_val"]).any(), "Corriente I contiene NaN."
    assert not np.isnan(data["v_val"]).any(), "Voltaje V contiene NaN."
    assert not np.isnan(data["wd_val"]).any(), "Estado w/d contiene NaN."


def test_plot_canvas_with_validation_data(qtbot=None):
    """Verifica que MplCanvas.plot_results acepte val_data sin errores."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    canvas = MplCanvas()
    t = np.linspace(0, 0.6, 100)
    v_in = np.sin(2 * np.pi * 5 * t)
    v_drop = v_in
    i_out = v_drop / 1000.0
    x_state = np.linspace(0.1, 0.8, 100)
    r_hist = np.full(100, 1000.0)
    g_hist = 1.0 / r_hist

    loader = ValidationDataLoader()
    val_data = loader.load_all()

    # Ejecutar sin val_data
    canvas.plot_results(t, v_in, v_drop, i_out, x_state, r_hist, g_hist)

    # Ejecutar con val_data
    canvas.plot_results(t, v_in, v_drop, i_out, x_state, r_hist, g_hist, val_data=val_data)


def test_config_panel_profile_serialization():
    """Verifica que to_dict y from_dict de ConfigPanel funcionen correctamente sin errores de atributos."""
    from PySide6.QtWidgets import QApplication
    from neurolab.gui.widgets.config_panel import ConfigPanel

    app = QApplication.instance() or QApplication([])
    panel = ConfigPanel()

    data = panel.to_dict()
    assert "enable_csv_validation" in data
    assert isinstance(data["enable_csv_validation"], bool)

    # Probar que from_dict deserialice sin errores
    panel.from_dict(data)
    assert panel.btn_csv_toggle.isChecked() == data["enable_csv_validation"]

