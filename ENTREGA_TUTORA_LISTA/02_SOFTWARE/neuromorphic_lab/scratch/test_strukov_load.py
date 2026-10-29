import sys
import os
from pathlib import Path

lab_dir = Path(r"g:\Github\Software de simulacion\Memristor_Environment\neuromorphic_lab")
sys.path.insert(0, str(lab_dir))

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from neurolab.gui.main_window import MainWindow

def test_strukov_load():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    app = QApplication.instance() or QApplication(sys.argv)

    window = MainWindow()
    
    # Load strukov_ideal preset on config_panel_1
    window.config_panel_1._load_strukov_paper_preset()
    
    panel = window.config_panel_1
    r_on = panel.get_r_on_ohms()
    r_off = panel.get_r_off_ohms()
    
    print("Strukov Ideal Loaded:")
    print(f"  spin_r_on: {panel.spin_r_on.value()}, unit: {panel.combo_ron_unit.currentText()} -> {r_on} Ohms")
    print(f"  spin_r_off: {panel.spin_r_off.value()}, unit: {panel.combo_roff_unit.currentText()} -> {r_off} Ohms")
    
    mem = panel.build_memristor()
    print("  Memristor built successfully! R_ON =", mem.electrical.r_on, "R_OFF =", mem.electrical.r_off)

if __name__ == "__main__":
    test_strukov_load()
