import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from neurolab.neurons import LIFConfig, LIFNeuron

def etapa28_spike_individual():
    """ETAPA 2.8 - Generar exactamente un spike."""
    config = LIFConfig()
    neuron = LIFNeuron(config)
    dt = 1e-4
    
    # Corriente apenas por encima del umbral (I_th = 9.5 uA)
    # Por ejemplo, 9.8 uA
    i_in = 9.8e-6 
    
    total_time = 0.2 # 200 ms
    steps = int(total_time / dt)
    t_array = np.linspace(0, total_time, steps, endpoint=False)
    
    v_sim = np.zeros(steps)
    
    # Inyección de pulso corto (para generar 1 solo spike) en vez de corriente continua infinita
    # o detener la corriente justo después del spike.
    for i in range(steps):
        v_sim[i] = neuron.v_membrane
        
        # Inyectar corriente solo hasta que veamos 1 spike
        if len(neuron.spike_times) < 1:
            neuron.step(i_in, dt)
        else:
            neuron.step(0.0, dt) # Dejar que caiga/se mantenga en reset
            
    plt.figure(figsize=(10, 4))
    plt.plot(t_array * 1000, v_sim, label='$V_m(t)$')
    plt.axhline(config.v_th, color='red', linestyle='--', label='Umbral ($V_{th}$)')
    plt.axhline(config.v_reset, color='green', linestyle=':', label='Reset ($V_{reset}$)')
    
    # Marcar spike manual para visualización
    if neuron.spike_times:
        t_sp = neuron.spike_times[0]
        plt.scatter([t_sp * 1000], [config.v_th], color='red', zorder=5, label='Spike Generado')
        
    plt.title("ETAPA 2.8: Spike Individual")
    plt.xlabel("Tiempo (ms)")
    plt.ylabel("Potencial de Membrana (V)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graficas/lif_etapa28_spike.png")
    plt.close()
    
    print("OK Etapa 2.8 (Spike Individual) completada. Gráfico guardado.")

def etapa29_tren_spikes():
    """ETAPA 2.9 - Generar un tren de spikes con corriente alta constante."""
    config = LIFConfig()
    neuron = LIFNeuron(config)
    dt = 1e-4
    
    # Corriente alta para múltiples disparos
    i_in = 15e-6 
    
    total_time = 0.5 # 500 ms
    steps = int(total_time / dt)
    t_array = np.linspace(0, total_time, steps, endpoint=False)
    
    v_sim = np.zeros(steps)
    
    for i in range(steps):
        v_sim[i] = neuron.v_membrane
        neuron.step(i_in, dt)
            
    # Análisis cuantitativo
    n_spikes = len(neuron.spike_times)
    f_spike = n_spikes / total_time
    
    isi_array = np.diff(neuron.spike_times)
    mean_isi = np.mean(isi_array) * 1000 if len(isi_array) > 0 else 0
    
    print(f"OK Etapa 2.9 (Tren de Spikes) completada:")
    print(f"  - Spikes generados: {n_spikes}")
    print(f"  - Frecuencia media: {f_spike:.2f} Hz")
    print(f"  - Intervalo Inter-Spike (ISI) promedio: {mean_isi:.2f} ms")
    
    plt.figure(figsize=(10, 4))
    plt.plot(t_array * 1000, v_sim, label='$V_m(t)$')
    for tsp in neuron.spike_times:
        plt.axvline(tsp * 1000, color='red', alpha=0.3)
    plt.title(f"ETAPA 2.9: Tren de Spikes ($f = {f_spike:.1f}$ Hz)")
    plt.xlabel("Tiempo (ms)")
    plt.ylabel("Potencial de Membrana (V)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graficas/lif_etapa29_train.png")
    plt.close()

def etapa210_curva_fi():
    """ETAPA 2.10 - Curva Frecuencia vs Corriente (F-I Curve)."""
    config = LIFConfig()
    dt = 1e-4
    total_time = 1.0 # 1 segundo para medir frec fácilmente
    
    currents = np.linspace(5e-6, 30e-6, 50)
    frequencies = []
    
    for i_in in currents:
        neuron = LIFNeuron(config)
        steps = int(total_time / dt)
        
        for _ in range(steps):
            neuron.step(i_in, dt)
            
        frequencies.append(len(neuron.spike_times) / total_time)
        
    plt.figure(figsize=(8, 5))
    plt.plot(currents * 1e6, frequencies, 'o-', color='purple')
    plt.axvline(9.5, color='red', linestyle='--', label='I_th teórica (9.5 $\mu$A)')
    
    plt.title("ETAPA 2.10: Curva Frecuencia-Corriente (F-I)")
    plt.xlabel("Corriente de Entrada ($\mu$A)")
    plt.ylabel("Frecuencia de Disparo (Hz)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graficas/lif_etapa210_FI_curve.png")
    plt.close()
    
    print("OK Etapa 2.10 (Curva F-I) completada. Gráfico guardado.")

def main():
    os.makedirs("graficas", exist_ok=True)
    print("=== INICIANDO EXPERIMENTOS LIF (ETAPAS 2.8 - 2.10) ===")
    etapa28_spike_individual()
    etapa29_tren_spikes()
    etapa210_curva_fi()
    print("=== EXPERIMENTOS FINALIZADOS ===")

if __name__ == '__main__':
    main()
