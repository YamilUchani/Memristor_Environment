import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from neurolab.devices.models.prezioso import PreziosoMathModel
from neurolab.core.config import ElectricalConfig, PreziosoConfig, DeviceIdentity
from neurolab.core.memristor import Memristor
from neurolab.io.validation_loader import ValidationDataLoader
from neurolab.core.validation_metrics import compute_metrics

# 1. Simular Prezioso con la configuración del paper
dev = Memristor(
    math_model=PreziosoMathModel(),
    electrical=ElectricalConfig(r_on=4750.0, r_off=1000000.0, initial_state=0.05),
    identity=DeviceIdentity(device_name="Prezioso 2014", model_name="prezioso"),
    model_config=PreziosoConfig(
        V_p=0.60, V_n=0.85,
        A_p=200.0, A_n=600.0,
        alpha_p=1.0, alpha_n=1.0,
        x_p=0.2, x_n=0.2
    )
)

dt_sec = 1e-6
duration = 0.4
t = np.linspace(0.0, duration, int(duration / dt_sec))
v0 = 2.0
f0 = 5.0
phase = (t * f0) % 1.0
v_sim = np.where(phase < 0.5, 4.0 * v0 * phase - v0, 3.0 * v0 - 4.0 * v0 * phase)

i_sim = np.zeros_like(t)
for k in range(len(t)):
    i_sim[k] = dev.step(v_sim[k], dt_sec)

i_sim_mA = i_sim * 1e3

# Tomar ciclo 2 (estabilizado)
mask2 = t >= 0.2
v_sim_c2 = v_sim[mask2]
i_sim_c2 = i_sim_mA[mask2]

# 2. Cargar CSV Prezioso
loader = ValidationDataLoader()
val_data = loader.load_all(dataset="prezioso")
v_ref = val_data["v_interp"]
i_ref_mA = val_data["i_val_mA"]

# Método C2: Mínima distancia V-I en ciclo 2
v_span = np.ptp(v_sim_c2) if np.ptp(v_sim_c2) > 0 else 1.0
i_span = np.ptp(i_sim_c2) if np.ptp(i_sim_c2) > 0 else 1.0

i_interp_c2 = np.array([
    i_sim_c2[np.argmin(((v_sim_c2 - vr) / v_span)**2 + ((i_sim_c2 - ir) / i_span)**2)]
    for vr, ir in zip(v_ref, i_ref_mA)
])

res_c2 = compute_metrics(i_ref_mA, i_interp_c2, "I(V) Dist (Ciclo 2)", "mA")

print(f"Ciclo 2 Mínima Distancia V-I: MAE = {res_c2.mae*1e3:.2f} uA | RMSE = {res_c2.rmse*1e3:.2f} uA | R2 = {res_c2.r2:.4f}")
