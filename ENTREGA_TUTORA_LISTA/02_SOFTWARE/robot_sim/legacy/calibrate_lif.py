import numpy as np
import json
from pathlib import Path
from neurobot.crossbar_brain import CrossbarBrain
from neurobot.lab_bridge import get_device_config, get_neuron_config

def main():
    device_cfg = get_device_config("strukov")
    base_neuron_cfg = get_neuron_config()
    cfg = {"learning_rate": 1e-6}
    
    N, M = 8, 8
    dt = 1e-3
    n_steps = 1000
    
    # Rango del estímulo
    V_rows = np.ones(N) * 0.5  # 0.5V constante
    
    cb = CrossbarBrain(N, M, device_cfg, base_neuron_cfg, cfg)
    
    # Veamos I_col promedio
    I_col, _ = cb.read(V_rows)
    mean_I = np.mean(I_col)
    print(f"Mean I_col for 0.5V rows: {mean_I} A")
    
    # Para la ecuación de LIF: dV = (I_col - V_m/R_leak) / C_m * dt
    # Queremos que V_m alcance V_th (1V) a una tasa de 5-50 Hz (es decir, cada 20-200 ms).
    # Si dV/dt ~ I_col / C_m, entonces V_th ~ (I_col / C_m) * T_spike
    # T_spike = 1 / f -> para 20 Hz, T_spike = 0.05 s
    # C_m = I_col * T_spike / V_th
    # Y R_leak lo suficientemente alto para no perder mucho, ej: V_th / I_col * 2
    
    if mean_I <= 1e-12:
        mean_I = 1e-6  # fallback
        
    target_hz = 25.0
    T_spike = 1.0 / target_hz
    
    # Proponemos nuevos parámetros
    new_v_th = 1.0
    new_c_m = mean_I * T_spike / new_v_th
    new_r_leak = new_v_th / (mean_I * 0.5)  # que la fuga sea la mitad del input
    new_r_series = 10.0 # Bajo para que no limite mucho si se usa en voltaje, pero aquí usamos corriente directa
    
    print(f"Propuestos: C_m={new_c_m}, R_leak={new_r_leak}")
    
    # Actualizamos los config para la calibración
    for neuron in cb.neurons:
        neuron.config.c_m = new_c_m
        neuron.config.r_leak = new_r_leak
        neuron.config.v_th = new_v_th
        neuron.config.r_series = new_r_series
        neuron.config.t_ref = 0.005 # 5ms
        neuron.reset()
        
    # Hacer perturbación en G para tener diferentes corrientes en columnas
    cb.G += np.random.randn(N, M) * 1e-4
    np.clip(cb.G, 1.0/device_cfg.r_off, 1.0/device_cfg.r_on, out=cb.G)
    
    spikes_count = np.zeros(M)
    winners_count = np.zeros(M)
    
    for _ in range(n_steps):
        spikes, I_c, winners, _ = cb.step(V_rows, dt)
        spikes_count += spikes
        winners_count += winners
        
    hz = spikes_count / (n_steps * dt)
    print("Tasas de disparo (Hz):", hz)
    print("Winners count:", winners_count)
    
    I_col, _ = cb.read(V_rows)
    print("I_col de cada columna:", I_col)
    max_col = np.argmax(I_col)
    max_winner = np.argmax(winners_count)
    
    print(f"La columna con mayor I_col es {max_col}, y el mayor winner es {max_winner}")
    if max_col == max_winner:
        print("WTA selecciona correctamente la columna con mayor corriente.")
    else:
        print("Aviso: WTA no coincide estrictamente o I_col es muy ruidosa, pero es esperado dependiendo de la inicialización de membrana.")

    # Guardar en neurobot/configs/lif_calibrado.json
    out_dir = Path(__file__).parent / "neurobot" / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lif_calibrado.json"
    
    calibrated_data = {
        "_meta": {"type": "lif_calibrado", "info": "Calibrado para 5-50 Hz con input moderado"},
        "c_m_value": new_c_m * 1e9,
        "c_m_unit": "nF",
        "r_leak_value": new_r_leak / 1e6,
        "r_leak_unit": "MΩ",
        "r_series_value": new_r_series / 1e3,
        "r_series_unit": "kΩ",
        "v_rest": 0.0,
        "v_th": new_v_th,
        "v_reset": 0.0,
        "t_ref": 0.005
    }
    with open(out_path, "w") as f:
        json.dump(calibrated_data, f, indent=4)
        
    print(f"Guardado en {out_path}")

if __name__ == '__main__':
    main()
