import numpy as np
import matplotlib.pyplot as plt
import os

# Configuración de estilo de gráfico
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12})

def simular_panel_completo_virgin_sample(filas=10, columnas=8, v_max=0.8, v_read=0.1):
    """
    Simula el panel completo de la figura Pre-Forming (Virgin Sample):
    (a) Curvas I-V representativas (Efecto Túnel S-shape)
    (b) Mapa de Conductancias espaciales
    (c) Histograma de variabilidad de conductancia
    """
    num_devices = filas * columnas
    
    # Parámetros base calibrados matemáticamente para dar G ~ 0.4 uS a 0.1V 
    # y I ~ 4 uA a 0.8V
    alpha_mean = 0.08   # uA
    beta_mean = 5.5     # V^-1
    
    # Generar la variabilidad para los 80 dispositivos
    np.random.seed(42)
    alphas = np.random.normal(alpha_mean, alpha_mean * 0.15, num_devices)
    betas = np.random.normal(beta_mean, beta_mean * 0.05, num_devices)
    
    # Calcular conductancia a V_read = 0.1V para cada dispositivo
    # I = alpha * sinh(beta * V)
    I_read = alphas * np.sinh(betas * v_read) # Corriente en uA
    G_read = (I_read / v_read) # Conductancia en uS (micro-Siemens)
    
    # Asegurarnos de que no haya conductancias negativas por el ruido
    G_read = np.clip(G_read, 0.1, 1.0)
    
    # Preparar el lienzo de 3 paneles
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # ==========================================
    # Panel (a) - Curvas I-V (10 representativas)
    # ==========================================
    ax_iv = axes[0]
    v_forward = np.linspace(-v_max, v_max, 100)
    v_reverse = np.linspace(v_max, -v_max, 100)
    V_sweep = np.concatenate([v_forward, v_reverse])
    
    colores = ['cyan', 'magenta', 'yellow', 'deepskyblue', 'gold', 'hotpink', 'aqua', 'orange', 'violet', 'lightskyblue']
    
    # Elegir 10 dispositivos al azar para mostrar
    idx_representativos = np.random.choice(num_devices, 10, replace=False)
    
    for i, idx in enumerate(idx_representativos):
        I_sweep = alphas[idx] * np.sinh(betas[idx] * V_sweep)
        ax_iv.plot(V_sweep, I_sweep, color=colores[i % len(colores)], alpha=0.8, linewidth=2)

    ax_iv.set_xlabel("Voltage (V)", fontweight='bold')
    ax_iv.set_ylabel("Current (μA)", fontweight='bold')
    ax_iv.set_xlim(-1.0, 1.0)
    ax_iv.set_ylim(-5, 5)
    ax_iv.set_title("(a) Representative I-V curves", fontweight='bold')
    
    # ==========================================
    # Panel (b) - Mapa de Conductancias (10x8)
    # ==========================================
    ax_map = axes[1]
    G_matrix = G_read.reshape((filas, columnas))
    
    # El paper describe colores verde (alta conductancia) a rojo (baja)
    im = ax_map.imshow(G_matrix, cmap='RdYlGn', aspect='auto')
    fig.colorbar(im, ax=ax_map, label='Conductance at 0.1V (μS)')
    
    ax_map.set_title("(b) Conductance Map (10x8)", fontweight='bold')
    ax_map.set_xlabel("Columns", fontweight='bold')
    ax_map.set_ylabel("Rows", fontweight='bold')
    
    # ==========================================
    # Panel (c) - Histograma de Conductancias
    # ==========================================
    ax_hist = axes[2]
    
    # Histograma con bins para mostrar la distribución unimodal
    ax_hist.hist(G_read, bins=12, color='royalblue', edgecolor='black', alpha=0.7)
    
    ax_hist.set_xlabel("Conductance G(0.1V) [μS]", fontweight='bold')
    ax_hist.set_ylabel("Number of Devices", fontweight='bold')
    ax_hist.set_title("(c) Conductance Distribution", fontweight='bold')
    ax_hist.set_xlim(0.2, 0.8)
    
    # Estilizar todos los paneles
    for ax in axes:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
    
    # Guardar
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'paso29_pre_forming_paneles.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Estadísticas del Crossbar Pre-Forming:")
    print(f"  Conductancia Media a 0.1V : {np.mean(G_read):.3f} uS")
    print(f"  Conductancia Min a 0.1V   : {np.min(G_read):.3f} uS")
    print(f"  Conductancia Max a 0.1V   : {np.max(G_read):.3f} uS")
    print(f"[PNG] Guardado panel completo: {out_path}")
    plt.close()

if __name__ == "__main__":
    print("Generando caracterización completa Pre-Forming (Curvas, Mapa, Histograma)...")
    simular_panel_completo_virgin_sample()
    print("Proceso completado.")
