"""
test_validation_panel.py
========================
Prueba automatizada para verificar que ValidationTestsPanel ejecuta las 16 pruebas
en vivo sin errores de kwargs ni excepciones.
"""

import pytest
import numpy as np
from PySide6.QtWidgets import QApplication

from neurolab.gui.crossbar_view import CrossbarView
from neurolab.gui.widgets.validation_tests_panel import ValidationTestsPanel, TESTS_CATALOG


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_validation_tests_panel_all_tests(qapp):
    """Verifica que cada una de las 16 pruebas en el catálogo se ejecute limpiamente con la vista 2D enlazada."""
    view = CrossbarView()
    panel = view.tests_panel

    assert panel is not None
    assert len(TESTS_CATALOG) >= 21



    for test in TESTS_CATALOG:
        panel.current_test = test
        panel._build_param_controls(test)
        
        # Ejecución directa
        panel.run_current_test()
        
        # Debe pasar sin errores y actualizar status a PASS
        assert "PASS" in panel.lbl_status.text(), f"La prueba {test['id']}. {test['name']} falló con estado: {panel.lbl_status.text()}"
