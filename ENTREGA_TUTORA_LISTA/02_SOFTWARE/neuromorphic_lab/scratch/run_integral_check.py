import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("\n" + "=" * 70)
print("  VERIFICACIÓN INTEGRAL A+B+C")
print("=" * 70)

# [1/6] Validaciones
val_dir = ROOT / "validaciones"
py_scripts = len(list(val_dir.glob("*.py")))
figuras = len(list((val_dir / "figuras").glob("*"))) if (val_dir / "figuras").exists() else 0
datos = len(list((val_dir / "datos").glob("*"))) if (val_dir / "datos").exists() else 0
reportes = len(list((val_dir / "reportes").glob("*"))) if (val_dir / "reportes").exists() else 0

print("\n[1/6] Estructura de validaciones/")
print(f"  Scripts (.py):   {py_scripts}")
print(f"  Figuras (.png):  {figuras}")
print(f"  Datos (.csv):    {datos}")
print(f"  Reportes (.txt): {reportes}")

# [2/6] Tests
tests_dir = ROOT / "tests"
dirs = [d.name for d in tests_dir.iterdir() if d.is_dir()]
total_test_files = len(list(tests_dir.rglob("test_*.py")))

print("\n[2/6] Estructura de tests/")
for d in sorted(dirs):
    print(f"  {d}/")
print(f"  Total tests .py: {total_test_files}")

# [3/6] Canvas Matplotlib
widgets_dir = ROOT / "neurolab" / "gui" / "widgets"
canvas_base = (widgets_dir / "canvas_base.py").exists()
print("\n[3/6] Canvas Matplotlib (Fase 4)")
print(f"  canvas_base.py : {canvas_base}")
for f in ["plot_canvas.py", "neuron_plot_canvas.py", "hybrid_plot_canvas.py", "synapse_plot_canvas.py"]:
    print(f"  {f} : {(widgets_dir / f).exists()}")

# [4/6] Imports críticos
print("\n[4/6] Imports críticos")
errors = []
try:
    from neurolab.core.memristor import Memristor
    from neurolab.crossbar import Crossbar, CrossbarConfig
except Exception as e:
    errors.append(f"core/crossbar: {e}")

try:
    from neurolab.gui.base_view import BaseCrossbarCanvas
    from neurolab.gui.crossbar_view import CrossbarView
    from neurolab.gui.crossbar_2x2_view import Crossbar2x2View
    from neurolab.gui.crossbar_4x4_view import Crossbar4x4View
except Exception as e:
    errors.append(f"gui crossbar: {e}")

try:
    from neurolab.gui.widgets.canvas_base import BaseMplCanvas
    from neurolab.gui.widgets.plot_canvas import MplCanvas
    from neurolab.gui.widgets.neuron_plot_canvas import NeuronMplCanvas
    from neurolab.gui.widgets.hybrid_plot_canvas import HybridMplCanvas
    from neurolab.gui.widgets.synapse_plot_canvas import SynapseMplCanvas
except Exception as e:
    errors.append(f"widgets canvas: {e}")

try:
    from validaciones import run_all_validaciones
except Exception as e:
    errors.append(f"validaciones: {e}")

if errors:
    print("[FAIL] Errores:")
    for err in errors:
        print(f"  - {err}")
else:
    print("[OK] Todos los imports")

# [5/6] Pytest completo
print("\n[5/6] Pytest completo")
res = subprocess.run(["python", "-m", "pytest", "tests/", "-q", "--tb=short"], capture_output=True, text=True)
print(res.stdout.strip())

# [6/6] Estructura raíz
print("\n[6/6] Estructura raíz")
for item in sorted(ROOT.iterdir()):
    if not item.name.startswith("."):
        tag = "[DIR]" if item.is_dir() else "[FILE]"
        print(f"  {tag:<7} {item.name}")

print("\n" + "=" * 70)
print("  VERIFICACIÓN COMPLETADA")
print("=" * 70)
