import sys
import os
from pathlib import Path

lab_dir = Path(r"g:\Github\Software de simulacion\Memristor_Environment\neuromorphic_lab")
sys.path.insert(0, str(lab_dir))

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
import numpy as np
from neurolab.gui.main_window import MainWindow


def test_gui_hybrid():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
        
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    
    p2 = window.config_panel_2
    print("P2 chk_volatile isChecked:", p2.chk_volatile.isChecked())
    print("P2 combo_realism_mode text:", p2.combo_realism_mode.currentText())
    print("P2 txt_device_name text:", p2.txt_device_name.text())
    
    # Build memristor from P2
    mem2 = p2.build_memristor()
    modifiers = [mod.__class__.__name__ for mod in mem2.modifiers]
    print("Memristor 2 modifiers:", modifiers)
    
    # Run hybrid simulation
    window.combo_mem_rs.setCurrentIndex(1) # Select Memristor 2 (HfO2 Serie Neurona)
    window.rb_mem_series.setChecked(True)
    window._run_hybrid_simulation()
    
    # Check simulated R_S trajectory
    t, v_signal, dt = window.hybrid_signal_panel.generate_voltage_signal()
    # R_S values are stored during simulation in window._run_hybrid_simulation()
    # Let's verify by stepping mem2 directly with the same voltage signal and dt
    r_hist = []
    v_m = 0.0
    for s_idx in range(len(t)):
        v_in = v_signal[s_idx]
        v_drop = max(v_in - v_m, 0.0)
        mem2.step(v_drop, dt)
        r_hist.append(mem2.resistance)
        
    r_hist = np.array(r_hist)
    print(f"R_S initial: {r_hist[0]/1e3:.2f} kOhm")
    print(f"R_S min during pulse: {np.min(r_hist)/1e3:.2f} kOhm")
    print(f"R_S final at t_end: {r_hist[-1]/1e3:.2f} kOhm")
    
    if r_hist[-1] > np.min(r_hist) * 1.5:
        print("[SUCCESS] R_S recovers/returns back towards R_OFF after pulse!")
    else:
        print("[FAIL] R_S did not recover!")

if __name__ == "__main__":
    test_gui_hybrid()
