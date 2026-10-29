"""
neurolab.gui.tests.tests_2x2
============================
Pruebas del crossbar 2×2 con visualización en tiempo real.
"""

from .test_01_operacion import draw_2x2_01
from .test_02_barrido_v import draw_2x2_02
from .test_03_barrido_g import draw_2x2_03
from .test_04_no_destructiva import draw_2x2_04
from .test_05_ltp_selectivo import draw_2x2_05
from .test_06_ltd_selectivo import draw_2x2_06
from .test_07_ciclo import draw_2x2_07
from .test_08_sneak import draw_2x2_08
from .test_09_line_resistance import draw_2x2_09
from .test_10_lif_dinamico import draw_2x2_10
from .test_11_stdp import draw_2x2_11
from .test_12_cross_talk import draw_2x2_12
from .test_13_escalabilidad import draw_2x2_13
from .test_14_uniformidad import draw_2x2_14
from .test_15_comparacion_1x1 import draw_2x2_15

__all__ = [
    'draw_2x2_01', 'draw_2x2_02', 'draw_2x2_03', 'draw_2x2_04',
    'draw_2x2_05', 'draw_2x2_06', 'draw_2x2_07', 'draw_2x2_08',
    'draw_2x2_09', 'draw_2x2_10', 'draw_2x2_11', 'draw_2x2_12',
    'draw_2x2_13', 'draw_2x2_14', 'draw_2x2_15',
]
