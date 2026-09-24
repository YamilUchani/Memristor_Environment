"""
utils/data_export.py
=====================
Exportación de resultados de simulación a CSV, JSON y paquete Unity.

Formatos soportados:
  - CSV:  compatible con Excel/MATLAB/Origin
  - JSON: compatible con APIs REST y Unity C#
  - Unity Package: serie temporal + metadatos para el gemelo digital

Uso::

    from memristor_simulator.utils.data_export import export_to_csv
    export_to_csv(results, "output/fig2b.csv")
"""

import os
import json
import csv
from datetime import datetime
from typing import Dict, Any
import numpy as np


def export_to_csv(results: Dict[str, np.ndarray],
                  filepath: str,
                  metadata: Dict[str, Any] = None) -> str:
    """
    Exporta resultados de simulación a un archivo CSV.

    Formato de columnas:
        time(s), voltage(V), current(A), current(mA), resistance(Ohm),
        state_variable(x), memristance(Ohm)

    Parámetros
    ----------
    results : dict
        Diccionario con claves 'time', 'voltage', 'current',
        'resistance', 'state_variable'.
    filepath : str
        Ruta de salida del archivo .csv.
    metadata : dict | None
        Metadatos adicionales escritos como comentarios en el encabezado.

    Retorna
    -------
    str : Ruta absoluta del archivo generado.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    time = np.asarray(results["time"])
    voltage = np.asarray(results["voltage"])
    current = np.asarray(results["current"])
    resistance = np.asarray(results["resistance"])
    state = np.asarray(results["state_variable"])

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        # Encabezado con metadatos
        f.write(f"# Memristor Strukov (2008) — Simulación numérica\n")
        f.write(f"# Generado: {datetime.now().isoformat()}\n")
        if metadata:
            for k, v in metadata.items():
                f.write(f"# {k}: {v}\n")

        writer = csv.writer(f)
        writer.writerow([
            "time_s", "voltage_V", "current_A", "current_mA",
            "resistance_Ohm", "state_variable_x", "memristance_Ohm"
        ])
        for t, v, i, r, x in zip(time, voltage, current, resistance, state):
            writer.writerow([
                f"{t:.6e}", f"{v:.6e}", f"{i:.6e}", f"{i*1e3:.6e}",
                f"{r:.4f}", f"{x:.6f}", f"{r:.4f}"
            ])

    abs_path = os.path.abspath(filepath)
    print(f"  [CSV] Exportado: {abs_path}  ({len(time)} filas)")
    return abs_path


def export_to_json(results: Dict[str, np.ndarray],
                   filepath: str,
                   params: Any = None,
                   decimals: int = 6) -> str:
    """
    Exporta resultados y parámetros a JSON (compatible con Unity C#).

    Estructura del JSON::

        {
          "metadata": { "generated": "...", "model": "strukov_2008", ... },
          "parameters": { "R_on": 100, "R_off": 16000, ... },
          "data": {
            "time": [...],
            "voltage": [...],
            "current_mA": [...],
            "resistance": [...],
            "state_variable": [...]
          }
        }

    Parámetros
    ----------
    results : dict
        Resultados de la simulación.
    filepath : str
        Ruta de salida del archivo .json.
    params : StrukovParameters | None
        Parámetros del dispositivo (añadidos al JSON).
    decimals : int
        Decimales de precisión para los datos.

    Retorna
    -------
    str : Ruta absoluta del archivo generado.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    def _round(arr):
        return [round(float(x), decimals) for x in np.asarray(arr)]

    payload = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "model": "strukov_2008",
            "reference": "Strukov et al., Nature 453, 80-83 (2008)",
            "n_points": len(results["time"]),
        },
        "data": {
            "time_s": _round(results["time"]),
            "voltage_V": _round(results["voltage"]),
            "current_mA": _round(np.asarray(results["current"]) * 1e3),
            "resistance_Ohm": _round(results["resistance"]),
            "state_variable_x": _round(results["state_variable"]),
        }
    }

    if params is not None:
        payload["parameters"] = {
            "R_on_Ohm": float(params.R_on),
            "R_off_Ohm": float(params.R_off),
            "ratio": float(params.R_off / params.R_on),
            "D_nm": float(params.D * 1e9),
            "mu_v_m2Vs": float(params.mu_v),
            "w_init": float(params.w_init),
        }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    abs_path = os.path.abspath(filepath)
    print(f"  [JSON] Exportado: {abs_path}")
    return abs_path


def export_unity_package(results: Dict[str, np.ndarray],
                          output_dir: str,
                          scenario_name: str = "memristor_sim",
                          params: Any = None) -> dict:
    """
    Genera el paquete completo para el Gemelo Digital en Unity.

    Archivos generados en output_dir/scenario_name/:
      - time_series.csv     → Serie temporal completa
      - iv_curve.csv        → Curva I-V (para gráficas en Unity)
      - metadata.json       → Parámetros y estadísticas del dispositivo
      - summary.txt         → Resumen legible en texto plano

    Parámetros
    ----------
    results : dict
        Resultados de la simulación.
    output_dir : str
        Directorio raíz de exportación.
    scenario_name : str
        Nombre del escenario (subcarpeta).
    params : StrukovParameters | None
        Parámetros del dispositivo.

    Retorna
    -------
    dict : Rutas de todos los archivos generados.
    """
    pkg_dir = os.path.join(output_dir, scenario_name)
    os.makedirs(pkg_dir, exist_ok=True)

    generated_files = {}

    # 1. Serie temporal completa
    ts_path = os.path.join(pkg_dir, "time_series.csv")
    export_to_csv(results, ts_path, {"scenario": scenario_name})
    generated_files["time_series_csv"] = ts_path

    # 2. Curva I-V (voltaje vs corriente)
    iv_path = os.path.join(pkg_dir, "iv_curve.csv")
    v = np.asarray(results["voltage"])
    i_ma = np.asarray(results["current"]) * 1e3
    with open(iv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["voltage_V", "current_mA"])
        for vi, ii in zip(v, i_ma):
            writer.writerow([f"{vi:.6f}", f"{ii:.6f}"])
    generated_files["iv_curve_csv"] = iv_path

    # 3. Metadatos JSON
    meta_path = os.path.join(pkg_dir, "metadata.json")
    export_to_json(
        {k: results[k] for k in ["time", "voltage", "current",
                                   "resistance", "state_variable"]},
        meta_path, params=params, decimals=4
    )
    generated_files["metadata_json"] = meta_path

    # 4. Resumen en texto plano
    summary_path = os.path.join(pkg_dir, "summary.txt")
    v_arr = np.asarray(results["voltage"])
    i_arr = np.asarray(results["current"]) * 1e3
    x_arr = np.asarray(results["state_variable"])
    r_arr = np.asarray(results["resistance"])

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"MEMRISTOR STRUKOV (2008) — GEMELO DIGITAL\n")
        f.write(f"Escenario: {scenario_name}\n")
        f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Puntos de simulación : {len(results['time'])}\n")
        f.write(f"Tiempo total         : {results['time'][-1]:.3f} s\n\n")
        f.write(f"Voltaje   : [{v_arr.min():.3f}, {v_arr.max():.3f}] V\n")
        f.write(f"Corriente : [{i_arr.min():.3f}, {i_arr.max():.3f}] mA\n")
        f.write(f"Estado x  : [{x_arr.min():.4f}, {x_arr.max():.4f}]\n")
        f.write(f"Resistencia: [{r_arr.min():.0f}, {r_arr.max():.0f}] Ohm\n")
        if params is not None:
            f.write(f"\nParámetros del dispositivo:\n")
            f.write(f"  R_on  = {params.R_on:.1f} Ohm\n")
            f.write(f"  R_off = {params.R_off:.1f} Ohm\n")
            f.write(f"  Ratio = {params.R_off/params.R_on:.1f}:1\n")
            f.write(f"  D     = {params.D*1e9:.1f} nm\n")
    generated_files["summary_txt"] = summary_path

    print(f"  [UNITY] Paquete exportado en: {os.path.abspath(pkg_dir)}/")
    for key, path in generated_files.items():
        print(f"    |- {os.path.basename(path)}")

    return generated_files
