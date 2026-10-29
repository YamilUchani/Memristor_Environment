"""
Prueba 4.2.2: Programación Selectiva de Memristores en Arreglo 2×2.
"""

import numpy as np
import pytest
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig
from neurolab.synapses import MemristiveSynapse, LTPRule


def test_2x2_programacion_selectiva():
    config = CrossbarConfig(n_rows=2, n_cols=2)
    cb = CrossbarIdeal(config)

    G_initial = cb.G_matrix.copy()

    # Programar solo M11 con LTP
    syn = MemristiveSynapse(cb.memristors[0, 0])
    ltp = LTPRule(n_pulses=40, V_pulse=+1.0)
    ltp.apply(syn, dt=1e-3)

    G_final = cb.G_matrix.copy()
    delta = G_final - G_initial

    # M11 debe haber incrementado
    assert delta[0, 0] > 0
    # M12, M21, M22 deben permanecer inalterados
    assert delta[0, 1] == 0
    assert delta[1, 0] == 0
    assert delta[1, 1] == 0
