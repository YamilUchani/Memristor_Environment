import json
from pathlib import Path
import numpy as np

print("=== BLOQUE 1: SUMMARY JSON ===")
p = Path("robot_sim/outputs/ablations_real_physics_summary.json")
if p.exists():
    data = json.loads(p.read_text(encoding="utf-8"))
    for r in data:
        print(f"{r['mode']} seed={r['seed']}  R20={r['r_last20']:>8.1f}  "
              f"R50={r['r_last50']:>8.1f}  Rmax={r['r_max']:>8.1f}  t={r['elapsed_s']:.0f}s")
else:
    print("No existe ablations_real_physics_summary.json")

print("\n=== BLOQUE 2: DETALLE DE ARCHIVOS JSON INDIVIDUALES ===")
files_to_check = [
    "run_24x12_A_base_seed0.json",
    "run_24x12_A_base_seed1.json",
    "run_24x12_A_base_seed2.json",
    "run_24x12_A_base_seed3.json",
    "run_24x12_A_base_seed4.json",
    "run_24x12_B_seed0.json",
    "run_24x12_C_seed0.json",
    "run_24x12_D_seed0.json"
]

for fname in files_to_check:
    fpath = Path("robot_sim/outputs") / fname
    if fpath.exists():
        log = json.loads(fpath.read_text(encoding="utf-8"))
        r = np.array(log["rewards"])
        print(f"\n{fname}:")
        print(f"  primeros 5: {r[:5].tolist()}")
        print(f"  últimos 20: {r[-20:].tolist()}")
        print(f"  media últimos 50: {r[-50:].mean():.1f}")
        print(f"  max: {r.max():.1f}")
    else:
        print(f"\n{fname}: NO ENCONTRADO")
