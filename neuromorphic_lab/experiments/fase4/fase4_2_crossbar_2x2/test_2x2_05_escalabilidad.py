"""
Prueba 4.2.5: Escalabilidad 1×1 vs 2×2.
"""

import numpy as np
import pytest
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig


def test_2x2_escalabilidad():
    config_1 = CrossbarConfig(n_rows=1, n_cols=1)
    config_2 = CrossbarConfig(n_rows=2, n_cols=2)

    cb1 = CrossbarIdeal(config_1)
    cb2 = CrossbarIdeal(config_2)

    cb1.apply_voltages([0.5])
    cb2.apply_voltages([0.5, 0.5])

    I1 = cb1.read_currents()
    I2 = cb2.read_currents()

    assert len(I1) == 1
    assert len(I2) == 2
