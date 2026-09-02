import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Asegurar que el módulo neurolab esté en el path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from neurolab.neurons import LIFConfig, LIFNeuron

def analytical_solution(t_array, v_rest, r, i_in, tau):
    """Calcula la respuesta analítica exacta de un circuito RC."""
    return v_rest + r * i_in * (1 - np.exp(-t_array / tau))

def run_rc_experiment(dt, total_time, i_in):
    """
    Ejecuta la simulación numérica de la neurona LIF y compara con la solución analítica.
    """
    config = LIFConfig()
    neuron = LIFNeuron(config)
    
    steps = int(total_time / dt)
    t_array = np.linspace(0, total_time, steps, endpoint=False)
    
    v_simulated = np.zeros(steps)
    
    for i in range(steps):
        v_simulated[i] = neuron.v_membrane
        neuron.step(i_in, dt)
        
    v_analytical = analytical_solution(t_array, config.v_rest, config.r_leak, i_in, config.tau)
    
    # Cálculo de errores
    error_abs = np.abs(v_simulated - v_analytical)
    mae = np.mean(error_abs)
    mse = np.mean(error_abs**2)
    max_error = np.max(error_abs)
    
    # Error relativo (evitando divisiones por cero al inicio donde V_ana = 0)
    mask = np.abs(v_analytical) > 1e-9
    if np.any(mask):
        rel_error = np.mean(error_abs[mask] / np.abs(v_analytical[mask]))
    else:
        rel_error = 0.0
        
    return t_array, v_simulated, v_analytical, mae, mse, max_error, rel_error

def main():
    # Parámetros del experimento
    total_time = 0.3  # 300 ms (la constante de tiempo tau es de 50 ms, llegará al estado estacionario)
    
    # Queremos que V_m no supere V_th (0.95 V).
    # V_max = R * I_in -> 100 kOhm * I_in
    # Para V_max = 0.8 V -> I_in = 8 uA (8e-6 A)
    i_in = 8e-6 
    
    # Resoluciones temporales (dt) a evaluar
    dts = [1e-3, 1e-4, 1e-5]  # 1 ms, 0.1 ms, 0.01 ms
    
    results = []
    
    # Configuración de la gráfica
    plt.figure(figsize=(10, 6))
    
    # Solución analítica de referencia (usando el dt más fino)
    t_finest, _, v_ana_finest, _, _, _, _ = run_rc_experiment(dts[-1], total_time, i_in)
    plt.plot(t_finest * 1000, v_ana_finest, label='Solución Analítica Exacta', color='black', linewidth=2)
    
    for dt in dts:
        t, v_sim, v_ana, mae, mse, max_e, rel_e = run_rc_experiment(dt, total_time, i_in)
        results.append((dt, mae, mse, max_e, rel_e))
        
        # Graficamos la simulación para visualizar su acoplamiento
        if dt == dts[0]:
            plt.plot(t * 1000, v_sim, label=f'Simulación (Euler) dt={dt*1000:.1f} ms', linestyle='--')
        elif dt == dts[-1]:
            plt.plot(t * 1000, v_sim, label=f'Simulación (Euler) dt={dt*1000:.2f} ms', linestyle='-.', color='green')
            
    # Línea del umbral
    plt.axhline(0.95, color='red', linestyle=':', label='$V_{th}$ (Umbral = 0.95 V)')
    
    plt.title("Validación de Respuesta RC de la Neurona LIF (Corriente Subumbral)", fontsize=14)
    plt.xlabel("Tiempo (ms)", fontsize=12)
    plt.ylabel("Potencial de Membrana $V_m$ (V)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Guardar gráfica
    output_dir = "graficas"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "lif_rc_validation.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    
    # Mostrar resultados numéricos
    print("=== RESULTADOS EXPERIMENTO DE VALIDACIÓN RC ===")
    print(f"Constante de tiempo tau = 50 ms | Corriente inyectada = 8 uA | V_max asintotico = 0.80 V\n")
    print(f"{'dt (s)':<12} | {'MAE (V)':<14} | {'MSE (V^2)':<14} | {'Error Max (V)':<14} | {'Error Relativo':<14}")
    print("-" * 75)
    for res in results:
        print(f"{res[0]:<12.1e} | {res[1]:<14.6e} | {res[2]:<14.6e} | {res[3]:<14.6e} | {res[4]:<14.6e}")
        
    print(f"\n[OK] Gráfica generada y guardada en: {file_path}")

if __name__ == '__main__':
    main()
