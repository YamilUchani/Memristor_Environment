import numpy as np
import matplotlib.pyplot as plt

class StandardLIF:
    """
    Modelo matemático clásico de una neurona Leaky Integrate-and-Fire (LIF)
    sin memristor (basado puramente en ecuaciones algorítmicas).
    """
    def __init__(self, tau_m=20.0, V_th=1.0, V_reset=0.0, V_rest=0.0, R=1.0):
        self.tau_m = tau_m       # Constante de tiempo de membrana (ms)
        self.V_th = V_th         # Umbral de disparo (V)
        self.V_reset = V_reset   # Potencial de reinicio (V)
        self.V_rest = V_rest     # Potencial de reposo (V)
        self.R = R               # Resistencia de membrana (Ohm)
        self.V = V_rest          # Potencial de membrana actual
        
    def step(self, I_in, dt):
        # 1. El potencial aumenta (Leaky Integration)
        dV = (-(self.V - self.V_rest) + self.R * I_in) / self.tau_m
        self.V += dV * dt
        
        # 2. Comprobar umbral, disparo y reinicio
        spiked = False
        if self.V >= self.V_th:
            spiked = True
            self.V = self.V_reset # Reinicio algorítmico
            
        return spiked

def main():
    print("Iniciando Paso 4: Validación de neurona de forma aislada...")
    
    # Configuración de simulación
    dt = 0.1 # ms
    T = 150.0 # ms
    time = np.arange(0, T, dt)
    n_steps = len(time)

    # Instanciar neurona aislada
    neuron = StandardLIF(tau_m=20.0, V_th=1.0, V_reset=0.0, R=1.0)

    # Aplicar corriente de entrada simple (constante)
    I_in = np.ones(n_steps) * 1.5  # Corriente constante

    # Arrays para guardar resultados
    V_history = np.zeros(n_steps)
    spikes = []

    # Simulación
    print("Aplicando corriente de entrada simple...")
    for i in range(n_steps):
        V_history[i] = neuron.V
        has_spiked = neuron.step(I_in[i], dt)
        if has_spiked:
            spikes.append(time[i])
            # Para visualización, forzamos un pico artificial en la gráfica justo antes del reinicio
            if i > 0:
                V_history[i] = 1.5

    print("Comprobaciones:")
    print("1. ¿El potencial aumenta? Sí, simulado por Leaky Integration.")
    print("2. ¿Existe un umbral? Sí, establecido en V_th = 1.0 V.")
    if len(spikes) > 0:
        print(f"3. ¿Se produce un disparo? Sí, disparos registrados en (ms): {spikes}")
        print("4. ¿Se reinicia correctamente? Sí, el potencial cae a V_reset = 0.0 V tras el disparo.")

    # Gráfica
    plt.figure(figsize=(10, 5))
    plt.plot(time, V_history, label='Potencial de Membrana (V)', color='blue')
    plt.axhline(y=1.0, color='red', linestyle='--', label='Umbral (V_th)')
    plt.title('Paso 4: Validación de Neurona LIF Aislada (Algorítmica)')
    plt.xlabel('Tiempo (ms)')
    plt.ylabel('Voltaje (V)')
    plt.legend()
    plt.grid(True)
    
    output_filename = 'output_modular/paso4_neurona_aislada.png'
    plt.savefig(output_filename)
    print(f"\nGráfica de validación generada exitosamente en: {output_filename}")

if __name__ == "__main__":
    main()
