import numpy as np
import matplotlib.pyplot as plt
import os, sys

# Añadir el entorno al path para poder importar los módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from memristor_simulator.models.physical_crossbar import PhysicalCrossbarArray
from memristor_simulator.models.stochastic_model import StochasticMemristor, C2CConfig
from memristor_simulator.config.parameters import StrukovParameters

# ─── Clase Helper para Simulación Estocástica de Crossbar ───
class StochasticCrossbarArray(PhysicalCrossbarArray):
    def __init__(self, size: int, base_params: StrukovParameters, on_off_ratio: float = 10000.0, seed=None):
        self.size = size
        self.grid = []
        self.on_off_ratio = on_off_ratio
        
        c2c_cfg = C2CConfig(enabled=True, r_on_cv=0.15, r_off_cv=0.10, use_lognormal=True, seed=seed)
        for i in range(size):
            row = []
            for j in range(size):
                p_copy = StrukovParameters(
                    R_on=base_params.R_on,
                    R_off=base_params.R_on * on_off_ratio,
                    D=base_params.D,
                    mu_v=base_params.mu_v,
                    w_init=0.0
                )
                mem = StochasticMemristor(p_copy, c2c_config=c2c_cfg)
                mem._apply_c2c_variability() # Randomizar R_on inicial simulando variabilidad D2D
                row.append(mem)
            self.grid.append(row)

# ─── Parámetros Físicos (Coherentes con Memristores Strukov y Estocástico) ───
array_sizes = ['4x4', '8x8', '16x16', '32x32', '64x64']
x = np.arange(len(array_sizes))
Ns = [4, 8, 16, 32, 64]

p_base = StrukovParameters(R_on=10e6, R_off=100e9, D=10e-9, mu_v=0.0) # Kon = 10^-7
# Ajustamos V_DD para que la escala de corriente máxima (en 64x64) sea ~3.4 uA como en el paper
v_dd = 1.95 

# Resistencias aproximadas por capa de interconexión (para evidenciar diferencias)
R_wires = {
    'M3': 390e3,
    'M5': 300e3,
    'M6': 210e3
}

# Diccionarios para almacenar resultados
isn_mean = {'M3':[], 'M5':[], 'M6':[]}
isn_std  = {'M3':[], 'M5':[], 'M6':[]}
nm_mean  = {'M3':[], 'M5':[], 'M6':[]}
nm_std   = {'M3':[], 'M5':[], 'M6':[]}

N_SIMS = 20 # Número de matrices estocásticas a simular por cada tamaño

print("Simulando matrices físicas para M3, M5 y M6 con modelo Estocástico y Strukov...")

for n in Ns:
    # Simulación de N_SIMS redes estocásticas para extraer media y varianza
    rs_samples = []
    for s in range(N_SIMS):
        cross_stoch = StochasticCrossbarArray(size=n, base_params=p_base, on_off_ratio=1e4, seed=s*100+n)
        cross_stoch.set_pattern_all_ones()
        rs_samples.append(cross_stoch.get_sneak_resistance(0, 0))
    
    # Calcular métricas para cada metal
    for metal in ['M3', 'M5', 'M6']:
        rw = R_wires[metal]
        isn_list = [(v_dd / (rs + rw)) * 1e6 for rs in rs_samples]
        nm_list  = [rs / (rs + rw) for rs in rs_samples]
        
        isn_mean[metal].append(np.mean(isn_list))
        isn_std[metal].append(np.std(isn_list))
        
        nm_mean[metal].append(np.mean(nm_list))
        nm_std[metal].append(np.std(nm_list))

print("Simulaciones completadas. Generando gráfica físicamente coherente...")

# ─── Configuración de Gráfica ───
width = 0.22

plt.rcParams['font.family'] = 'serif'
if 'Times New Roman' in plt.rcParams['font.serif']:
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('white')

color_m3 = '#2078B4' # Azul
color_m5 = '#FF7F0E' # Naranja
color_m6 = '#2CA02C' # Verde

# ─────────────────────────────────────────────────────────────
# Panel a: Sneak Path Current (Simulada Físicamente)
# ─────────────────────────────────────────────────────────────
axs[0].bar(x - width, isn_mean['M3'], width, yerr=isn_std['M3'], capsize=3, label='M3 (Simulated)', color=color_m3, error_kw=dict(lw=1, capthick=1))
axs[0].bar(x,         isn_mean['M5'], width, yerr=isn_std['M5'], capsize=3, label='M5 (Simulated)', color=color_m5, error_kw=dict(lw=1, capthick=1))
axs[0].bar(x + width, isn_mean['M6'], width, yerr=isn_std['M6'], capsize=3, label='M6 (Simulated)', color=color_m6, error_kw=dict(lw=1, capthick=1))

axs[0].set_ylim(0, 4)
axs[0].set_yticks(np.arange(0, 4.5, 0.5))
axs[0].set_xticks(x)
axs[0].set_xticklabels(array_sizes, fontweight='bold', fontsize=12)

axs[0].set_ylabel(r'$\mathbf{I_{sneak} \ (\mu A)}$', fontsize=14)
axs[0].set_xlabel('Array Size', fontsize=14, fontweight='bold')

axs[0].tick_params(axis='both', which='major', labelsize=11, direction='in', 
                   length=5, width=1.2, bottom=True, top=True, left=True, right=True)
for spine in axs[0].spines.values():
    spine.set_linewidth(1.2)

leg1 = axs[0].legend(loc='upper left', frameon=False, fontsize=10, 
                     handlelength=1.2, handletextpad=0.4, borderpad=0.2)
for text in leg1.get_texts():
    text.set_fontweight('bold')

axs[0].text(-0.12, 1.06, 'a.', transform=axs[0].transAxes, fontsize=20, 
            fontweight='bold', va='bottom', fontfamily='serif')

# ─────────────────────────────────────────────────────────────
# Panel b: Noise Margin (Simulada Físicamente)
# ─────────────────────────────────────────────────────────────
axs[1].bar(x - width, nm_mean['M3'], width, yerr=nm_std['M3'], capsize=3, label='M3', color=color_m3, error_kw=dict(lw=1, capthick=1))
axs[1].bar(x,         nm_mean['M5'], width, yerr=nm_std['M5'], capsize=3, label='M5', color=color_m5, error_kw=dict(lw=1, capthick=1))
axs[1].bar(x + width, nm_mean['M6'], width, yerr=nm_std['M6'], capsize=3, label='M6', color=color_m6, error_kw=dict(lw=1, capthick=1))

axs[1].set_ylim(0, 1.1)
axs[1].set_yticks(np.arange(0, 1.2, 0.2))
axs[1].set_xticks(x)
axs[1].set_xticklabels(array_sizes, fontweight='bold', fontsize=12)

axs[1].set_ylabel('Noise Margin', fontsize=14, fontweight='bold')
axs[1].set_xlabel('Array Size', fontsize=14, fontweight='bold')

axs[1].tick_params(axis='both', which='major', labelsize=11, direction='in', 
                   length=5, width=1.2, bottom=True, top=True, left=True, right=True)
for spine in axs[1].spines.values():
    spine.set_linewidth(1.2)

leg2 = axs[1].legend(loc='upper right', frameon=False, fontsize=10, 
                     handlelength=1.2, handletextpad=0.4, borderpad=0.2)
for text in leg2.get_texts():
    text.set_fontweight('bold')

axs[1].text(-0.12, 1.06, 'b.', transform=axs[1].transAxes, fontsize=20, 
            fontweight='bold', va='bottom', fontfamily='serif')

plt.subplots_adjust(wspace=0.3)

# Guardar figura
out_dir = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, "paso27_replicacion_figura5_sneak.png")
plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[PNG] Imagen guardada en: {png_path}")
