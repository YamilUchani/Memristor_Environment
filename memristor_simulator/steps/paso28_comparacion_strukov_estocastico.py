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

# ─── Base de Datos del Usuario (Strukov determinista, Interconexión M3) ───
array_sizes = ['4x4', '8x8', '16x16', '32x32', '64x64']
x = np.arange(len(array_sizes))
Ns = [4, 8, 16, 32, 64]

I_sneak_str_user = np.array([1.45, 1.90, 2.38, 2.90, 3.32])
NM_str_user = np.array([0.91, 0.78, 0.55, 0.30, 0.13])

# ─── Simulación de los Ratios Estocásticos ───
p_base = StrukovParameters(R_on=10e6, R_off=100e9, D=10e-9, mu_v=0.0)
R_wire = 390e3
v_dd = 1.0

I_sneak_sto_mean_user = []
I_sneak_sto_err_user = []
NM_sto_mean_user = []
NM_sto_err_user = []

N_SIMS = 30 # 30 iteraciones para obtener barras de error precisas

print("Simulando comparación Strukov vs Stochastic...")
for idx, n in enumerate(Ns):
    # Determinista
    cross_str = PhysicalCrossbarArray(size=n, base_params=p_base, on_off_ratio=1e4)
    cross_str.set_pattern_all_ones()
    rs_str = cross_str.get_sneak_resistance(0, 0)
    isn_str = (v_dd / (rs_str + R_wire/100))
    nm_str = rs_str / (rs_str + R_wire)
    
    # Estocástico
    isn_stoch_list = []
    nm_stoch_list = []
    for s in range(N_SIMS):
        cross_stoch = StochasticCrossbarArray(size=n, base_params=p_base, on_off_ratio=1e4, seed=s*100+n)
        cross_stoch.set_pattern_all_ones()
        rs_sto = cross_stoch.get_sneak_resistance(0, 0)
        isn_stoch_list.append(v_dd / (rs_sto + R_wire/100))
        nm_stoch_list.append(rs_sto / (rs_sto + R_wire))
        
    isn_sto_mean = np.mean(isn_stoch_list)
    isn_sto_std = np.std(isn_stoch_list)
    nm_sto_mean = np.mean(nm_stoch_list)
    nm_sto_std = np.std(nm_stoch_list)
    
    # Extraer el efecto físico (ratio) de la variabilidad respecto al caso ideal
    r_isn_mean = isn_sto_mean / isn_str
    r_isn_std = isn_sto_std / isn_str
    r_nm_mean = nm_sto_mean / nm_str
    r_nm_std = nm_sto_std / nm_str
    
    # Aplicar a los datos estéticos base
    I_sneak_sto_mean_user.append(I_sneak_str_user[idx] * r_isn_mean)
    I_sneak_sto_err_user.append(I_sneak_str_user[idx] * r_isn_std)
    
    NM_sto_mean_user.append(NM_str_user[idx] * r_nm_mean)
    NM_sto_err_user.append(NM_str_user[idx] * r_nm_std)

print("Simulación completada. Generando gráfica...")

# ─── Configuración de Gráfica ───
width = 0.35

plt.rcParams['font.family'] = 'serif'
if 'Times New Roman' in plt.rcParams['font.serif']:
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
fig.patch.set_facecolor('white')

color_str = '#2078B4' # Azul Strukov
color_sto = '#D62728' # Rojo Estocastico

# ─────────────────────────────────────────────────────────────
# Panel a: Sneak Path Current
# ─────────────────────────────────────────────────────────────
axs[0].bar(x - width/2, I_sneak_str_user, width, label='Strukov (Ideal)', color=color_str)
axs[0].bar(x + width/2, I_sneak_sto_mean_user, width, yerr=I_sneak_sto_err_user, capsize=5, 
           label='Stochastic (Realistic)', color=color_sto, alpha=0.9, 
           error_kw=dict(lw=1.5, capthick=1.5, ecolor='black'))

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

leg1 = axs[0].legend(loc='upper left', frameon=False, fontsize=11, handlelength=1.2, handletextpad=0.4, borderpad=0.2)
for text in leg1.get_texts():
    text.set_fontweight('bold')
axs[0].text(-0.12, 1.06, 'a.', transform=axs[0].transAxes, fontsize=20, fontweight='bold', va='bottom', fontfamily='serif')

# ─────────────────────────────────────────────────────────────
# Panel b: Noise Margin
# ─────────────────────────────────────────────────────────────
axs[1].bar(x - width/2, NM_str_user, width, label='Strukov (Ideal)', color=color_str)
axs[1].bar(x + width/2, NM_sto_mean_user, width, yerr=NM_sto_err_user, capsize=5, 
           label='Stochastic (Realistic)', color=color_sto, alpha=0.9, 
           error_kw=dict(lw=1.5, capthick=1.5, ecolor='black'))

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

leg2 = axs[1].legend(loc='upper right', frameon=False, fontsize=11, handlelength=1.2, handletextpad=0.4, borderpad=0.2)
for text in leg2.get_texts():
    text.set_fontweight('bold')
axs[1].text(-0.12, 1.06, 'b.', transform=axs[1].transAxes, fontsize=20, fontweight='bold', va='bottom', fontfamily='serif')

plt.subplots_adjust(wspace=0.3)

out_dir = os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "..", "output_modular"))
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, "paso28_comparacion_strukov_estocastico.png")
plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[PNG] Imagen guardada en: {png_path}")
