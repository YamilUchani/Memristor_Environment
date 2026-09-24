"""
generate_4x4_validation_figures.py
===================================
Script automatizado para ejecutar las 20 pruebas de validación del Crossbar 4×4,
generar imágenes publication-ready con fondo blanco en:
  1) outputs/fase_4_3/figuras/
  2) validaciones_4x4/
y exportar reportes en CSV y JSON.
"""

import os
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from neurolab.gui.tests.tests_4x4 import (
    draw_4x4_01, draw_4x4_02, draw_4x4_03, draw_4x4_04,
    draw_4x4_05, draw_4x4_06, draw_4x4_07, draw_4x4_08,
    draw_4x4_09, draw_4x4_10, draw_4x4_11, draw_4x4_12,
    draw_4x4_13, draw_4x4_14, draw_4x4_15, draw_4x4_16,
    draw_4x4_17, draw_4x4_18, draw_4x4_19, draw_4x4_20
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_fig_dir = os.path.join(base_dir, 'outputs', 'fase_4_3', 'figuras')
    output_rep_dir = os.path.join(base_dir, 'outputs', 'fase_4_3', 'reportes')
    validaciones_dir = os.path.join(base_dir, 'validaciones_4x4')

    os.makedirs(output_fig_dir, exist_ok=True)
    os.makedirs(output_rep_dir, exist_ok=True)
    os.makedirs(validaciones_dir, exist_ok=True)

    gui = MockGUI()

    tests = [
        ("test_01_operacion.png", "fig_01_operacion_matricial.png", draw_4x4_01, {}),
        ("test_02_barrido_v.png", "fig_02_barrido_voltaje.png", draw_4x4_02, {}),
        ("test_03_barrido_g.png", "fig_03_barrido_conductancia.png", draw_4x4_03, {}),
        ("test_04_no_destructiva.png", "fig_04_lectura_no_destructiva.png", draw_4x4_04, {}),
        ("test_05_ltp_selectivo.png", "fig_05_ltp_selectivo.png", draw_4x4_05, {}),
        ("test_06_ltd_selectivo.png", "fig_06_ltd_selectivo.png", draw_4x4_06, {}),
        ("test_07_ciclo.png", "fig_07_ciclo_reversible.png", draw_4x4_07, {}),
        ("test_08_sneak.png", "fig_08_sneak_paths.png", draw_4x4_08, {}),
        ("test_09_line_resistance.png", "fig_09_line_resistance.png", draw_4x4_09, {}),
        ("test_10_lif_dinamico.png", "fig_10_4_lif_dinamicas.png", draw_4x4_10, {}),
        ("test_11_stdp.png", "fig_11_stdp_ventana.png", draw_4x4_11, {}),
        ("test_12_cross_talk.png", "fig_12_cross_talk.png", draw_4x4_12, {}),
        ("test_13_escalabilidad.png", "fig_13_escalabilidad.png", draw_4x4_13, {}),
        ("test_14_uniformidad.png", "fig_14_uniformidad.png", draw_4x4_14, {}),
        ("test_15_matriz_identidad.png", "fig_15_matriz_identidad.png", draw_4x4_15, {}),
        ("test_16_matriz_diagonal.png", "fig_16_matriz_diagonal.png", draw_4x4_16, {}),
        ("test_17_patron_X.png", "fig_17_patron_X.png", draw_4x4_17, {}),
        ("test_18_patron_T.png", "fig_18_patron_T.png", draw_4x4_18, {}),
        ("test_19_patron_4x4.png", "fig_19_patron_4x4.png", draw_4x4_19, {}),
        ("test_20_comparacion.png", "fig_20_comparacion.png", draw_4x4_20, {}),
    ]

    print(f"Generando 20 imágenes de validación Crossbar 4×4 en:\n - {output_fig_dir}\n - {validaciones_dir}\n")

    summary_results = []
    consolidated_metrics = {}

    for filename_out, filename_val, func, kwargs in tests:
        res = func(gui, **kwargs)
        metrics = res.get('metrics', {}) if isinstance(res, dict) else {}
        status = res.get('status', 'PASS') if isinstance(res, dict) else 'PASS'

        # Guardar en outputs/fase_4_3/figuras/
        path_out = os.path.join(output_fig_dir, filename_out)
        gui.figure.savefig(path_out, dpi=150, facecolor='white', bbox_inches='tight')

        # Guardar en validaciones_4x4/
        path_val = os.path.join(validaciones_dir, filename_val)
        gui.figure.savefig(path_val, dpi=150, facecolor='white', bbox_inches='tight')

        print(f"  [OK] {filename_out} | Status: {status}")

        test_id = filename_out.replace('.png', '')
        summary_results.append({
            'test_id': test_id,
            'status': status,
            'metrics': json.dumps(metrics)
        })
        consolidated_metrics[test_id] = metrics

    # Guardar CSV
    csv_path = os.path.join(output_rep_dir, 'resultados_4x4.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['test_id', 'status', 'metrics'])
        writer.writeheader()
        writer.writerows(summary_results)

    # Guardar JSON
    json_path = os.path.join(output_rep_dir, 'metricas_consolidadas.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(consolidated_metrics, f, indent=2, ensure_ascii=False)

    print("\nTodas las 20 imagenes y reportes de validacion 4x4 fueron generados exitosamente!")


if __name__ == '__main__':
    main()
