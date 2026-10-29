import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from neurolab.devices.models.prezioso import PreziosoMathModel
from neurolab.core.config import ElectricalConfig, PreziosoConfig, DeviceIdentity
from neurolab.core.memristor import Memristor
from neurolab.io.validation_loader import ValidationDataLoader
from neurolab.core.validation_metrics import compute_all_metrics

# 1. Instanciar memristor Prezioso igual que la Subpestaña 4
dev = Memristor(
    math_model=PreziosoMathModel(),
    electrical=ElectricalConfig(r_on=4000.0, r_off=1000000.0, initial_state=0.05),
    identity=DeviceIdentity(device_name="Prezioso 2014", model_name="prezioso"),
    model_config=PreziosoConfig()
)

# 2. Simular barrido triangular V = 2.0 V, f = 5 Hz, T = 0.4 s, dt = 1 us
v0 = 2.0
f0 = 5.0
duration = 0.4
dt_sec = 1e-6

t = np.linspace(0.0, duration, int(duration / dt_sec))
phase = (t * f0) % 1.0
v_signal = np.where(phase < 0.5, 4.0 * v0 * phase - v0, 3.0 * v0 - 4.0 * v0 * phase)

i_out = np.zeros_like(t)
x_hist = np.zeros_like(t)
r_hist = np.zeros_like(t)
g_hist = np.zeros_like(t)

for k in range(len(t)):
    i_out[k] = dev.step(v_signal[k], dt_sec)
    x_hist[k] = dev.x
    r_hist[k] = dev.resistance
    g_hist[k] = dev.conductance

# 3. Cargar CSV Prezioso
loader = ValidationDataLoader()
val_data = loader.load_all(dataset="prezioso")

# 4. Calcular métricas
metrics = compute_all_metrics(
    t_sim=t,
    i_sim_mA=i_out * 1e3,
    x_sim=x_hist,
    r_sim_kohm=r_hist / 1e3,
    g_sim_us=g_hist * 1e6,
    val_data=val_data,
    v_sim=v_signal
)

print(f"=== METRICAS SUBPESTANA 4 (PREZIOSO) ===")
for m in metrics:
    print(f"Magnitud: {m.name} | MAE: {m.mae:.4f} | RMSE: {m.rmse:.4f} | R2: {m.r2:.4f}")

print(f"I_SET max: {np.max(i_out)*1e6:.1f} uA")
print(f"I_RESET max: {np.min(i_out)*1e6:.1f} uA")
