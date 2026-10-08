import sys, json, inspect
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent / "neuromorphic_lab"
sys.path.insert(0, str(LAB))

# A) ¿existe neurolab.io?
io_path = LAB / "neurolab" / "io"
print("A) neurolab/io/ existe:", io_path.exists())
try:
    from neurolab.io.profile_manager import ProfileManager
    print("   ProfileManager:", ProfileManager)
except Exception as e:
    print("   import falla:", e)

# B) firma real del dispositivo
from neurolab.devices.models import strukov as sk   # ajustar si la ruta difiere
print("\nB) contenido de neurolab/devices/models/:")
for f in (LAB / "neurolab" / "devices" / "models").glob("*.py"):
    print("   -", f.name)
    try:
        txt = f.read_text(encoding="utf-8", errors="ignore")
        for line in txt.splitlines():
            if line.strip().startswith("class ") or " def update" in line or " def __init__" in line:
                print("       ", line.strip())
    except:
        pass

# C) constructor del modelo Strukov
try:
    from neurolab.core.memristor import Memristor
    print("\nC) Memristor.__init__:")
    print("   ", inspect.signature(Memristor.__init__))
    print("   update:", inspect.signature(Memristor.update))
except Exception as e:
    print("\nC) Memristor import falla:", e)

# D) literal del JSON LIF
print("\nD) lif_config.json completo:")
try:
    print(json.dumps(json.loads((LAB / "configs" / "lif_config.json").read_text()), indent=2)[:2000])
except Exception as e:
    print("   falló:", e)
