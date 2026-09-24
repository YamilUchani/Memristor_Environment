"""
generate_2x2_validation_figures.py
==================================
Genera las 15 imágenes de las pruebas de validación del Crossbar 2×2
y las guarda en la carpeta neuromorphic_lab/validaciones_2x2/
"""

import os
import sys
from matplotlib.figure import Figure

# Agregar la ruta del paquete neurolab
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neurolab.gui.tests.tests_2x2 import (
    draw_2x2_01, draw_2x2_02, draw_2x2_03, draw_2x2_04,
    draw_2x2_05, draw_2x2_06, draw_2x2_07, draw_2x2_08,
    draw_2x2_09, draw_2x2_10, draw_2x2_11, draw_2x2_12,
    draw_2x2_13, draw_2x2_14, draw_2x2_15
)


class MockCanvas:
    def draw(self):
        pass


class MockGUI:
    def __init__(self):
        self.figure = Figure(figsize=(10, 5), dpi=150, facecolor='white')
        self.canvas = MockCanvas()

    def get_axis(self):
        self.figure.clear()
        return self.figure.add_subplot(111)

    def _style_axis(self, ax):
        ax.set_facecolor('#ffffff')
        ax.tick_params(colors='#1f2937', labelsize=8)
        ax.xaxis.label.set_color('#1f2937')
        ax.yaxis.label.set_color('#1f2937')
        ax.title.set_color('#1e3a8a')
        for spine in ax.spines.values():
            spine.set_color('#cbd5e1')
        ax.grid(True, linestyle='--', alpha=0.5, color='#e2e8f0')

    def refresh_plot(self):
        pass


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validaciones_2x2')
    os.makedirs(output_dir, exist_ok=True)

    gui = MockGUI()

    tests = [
        ("fig_01_operacion_matricial.png", draw_2x2_01, {}),
        ("fig_02_barrido_voltaje.png", draw_2x2_02, {}),
        ("fig_03_barrido_conductancia.png", draw_2x2_03, {}),
        ("fig_04_lectura_no_destructiva.png", draw_2x2_04, {}),
        ("fig_05_ltp_selectivo.png", draw_2x2_05, {}),
        ("fig_06_ltd_selectivo.png", draw_2x2_06, {}),
        ("fig_07_ciclo_reversible.png", draw_2x2_07, {}),
        ("fig_08_sneak_paths.png", draw_2x2_08, {}),
        ("fig_09_line_resistance.png", draw_2x2_09, {}),
        ("fig_10_2_lif_dinamicas.png", draw_2x2_10, {}),
        ("fig_11_stdp_ventana.png", draw_2x2_11, {}),
        ("fig_12_cross_talk.png", draw_2x2_12, {}),
        ("fig_13_escalabilidad.png", draw_2x2_13, {}),
        ("fig_14_uniformidad.png", draw_2x2_14, {}),
        ("fig_15_comparacion_1x1_vs_2x2.png", draw_2x2_15, {}),
    ]

    print(f"Generando 15 imágenes de validación en: {output_dir}\n")

    for fname, func, kwargs in tests:
        res = func(gui, **kwargs)
        filepath = os.path.join(output_dir, fname)
        gui.figure.savefig(filepath, bbox_inches='tight', facecolor='white')
        print(f"  [OK] Guardado: {fname}")

    print("\n¡Todas las 15 imágenes fueron generadas exitosamente!")


if __name__ == "__main__":
    main()
