import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from neurolab.validation.jo2010_stdp import validate_jo2010_stdp
from neurolab.validation.jo2010_ltp_ltd import validate_jo2010_ltp_ltd
from neurolab.synapses import STDPRule, LTPRule, LTDRule, MemristiveSynapse

from neurolab.core.memristor import Memristor
from neurolab.devices.config import DeviceConfig, StrukovConfig



def plot_stdp():
    # Parameters from GUI
    A_plus = 0.20
    A_minus = -0.20
    tau_plus = 30.0 * 1e-3
    tau_minus = 30.0 * 1e-3

    stdp = STDPRule(A_plus=A_plus, A_minus=A_minus, tau_plus=tau_plus, tau_minus=tau_minus)
    dt_values = np.linspace(-80.0, 80.0, 161)
    dw_sim = np.array([stdp.delta_w(dt * 1e-3) * 100.0 for dt in dt_values])

    results = validate_jo2010_stdp(dt_values, dw_sim)

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', alpha=0.6)

    ax.plot(results['dt_sim_ms'], results['dw_sim_pct'], color='#2ca02c', linewidth=2, label='Modelo analítico STDP')
    
    # Split exp data into LTP and LTD
    dt_exp = results['dt_exp_ms']
    dw_exp = results['dw_exp_pct']
    mask_ltp = dt_exp > 0
    mask_ltd = dt_exp < 0
    
    ax.scatter(dt_exp[mask_ltp], dw_exp[mask_ltp], s=80, color='#1f77b4', edgecolors='white', label='Jo 2010 (exp.) — LTP', zorder=3)
    ax.scatter(dt_exp[mask_ltd], dw_exp[mask_ltd], s=80, color='#d62728', edgecolors='white', label='Jo 2010 (exp.) — LTD', zorder=3)

    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel(r'$\Delta t = t_{\mathrm{post}} - t_{\mathrm{pre}}$ (ms)', fontsize=12)
    ax.set_ylabel(r'$\Delta W$ (%)', fontsize=12)
    ax.set_title('Jo 2010 — Ventana STDP (Validación Cuantitativa)', fontsize=14, fontweight='bold', color='#1f77b4')

    ax.legend(fontsize=10, facecolor='white', framealpha=0.9)

    text_str = f"$R^2 = {results['r2']:.4f}$\nMAE = {results['mae']*100:.2f}%\nRMSE = {results['rmse']*100:.2f}%"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.95, 0.05, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    out_path = Path(__file__).parent / 'figuras' / 'fig_15b_stdp_jo2010.png'
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

def plot_ltp_ltd():
    from neurolab.core.config import ElectricalConfig, StrukovConfig
    from neurolab.devices.models.strukov import StrukovMathModel
    from neurolab.devices.config import DeviceConfig
    
    results = validate_jo2010_ltp_ltd()
    
    # 1. Instantiate the Memristor EXACTLY like the GUI
    identity = DeviceConfig(name="Strukov TiO2", family="TiO2_oxide", model="strukov")
    electrical = ElectricalConfig(r_on=100.0, r_off=16000.0, initial_state=0.10)
    math_model = StrukovMathModel()
    model_config = StrukovConfig(D=10.0 * 1e-9, mu_v=1e-14) # D in meters
    
    modifiers = []
    
    mem = Memristor(
        math_model=math_model,
        electrical=electrical,
        identity=identity,
        model_config=model_config,
        modifiers=modifiers,
        clip_x=True
    )
    
    syn = MemristiveSynapse(mem)
    ltp = LTPRule(n_pulses=100, V_pulse=+1.0, saturation=True, tau_sat=35.0)
    G_ltp = ltp.apply(syn, dt=1e-3)
    ltd = LTDRule(n_pulses=100, V_pulse=-1.0, saturation=True, tau_sat=35.0)
    G_ltd = ltd.apply(syn, dt=1e-3)
    
    G_sim = np.concatenate([G_ltp, G_ltd[1:]])
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', alpha=0.6)

    exp_pulses = results['pulse_num']
    exp_I = results['current_exp']
    idx_peak = results['idx_peak']
    
    ax.plot(exp_pulses[:idx_peak+1], exp_I[:idx_peak+1], 's-', color='#1f77b4', markersize=6, linewidth=1.5, label='LTP (Jo 2010, exp.)', zorder=3)
    ax.plot(exp_pulses[idx_peak:], exp_I[idx_peak:], 'o-', color='#d62728', markersize=6, linewidth=1.5, label='LTD (Jo 2010, exp.)', zorder=3)
    
    # Normalize G_sim to match current scale as in GUI
    G_norm = (G_sim - G_sim.min()) / (G_sim.max() - G_sim.min() + 1e-12) * (exp_I.max() - exp_I.min()) + exp_I.min()
    pulse_sim = np.linspace(exp_pulses[0], exp_pulses[-1], len(G_sim))
    ax.plot(pulse_sim, G_norm, '--', color='#2ca02c', linewidth=2.5, label='Simulación', zorder=2)
    
    ax.set_xlabel('Pulso', fontsize=12)
    ax.set_ylabel('Corriente (100 nA/pulso)', fontsize=12)
    ax.set_title('Jo 2010 — LTP/LTD (Validación Cuantitativa)', fontsize=14, fontweight='bold', color='#1f77b4')
    
    # Compute quantitative metrics
    G_norm_at_exp = np.interp(exp_pulses, pulse_sim, G_norm)
    mae_100na = np.mean(np.abs(G_norm_at_exp - exp_I))
    rmse_100na = np.sqrt(np.mean((G_norm_at_exp - exp_I)**2))
    
    if len(exp_I) > 1 and np.std(exp_I) > 1e-9 and np.std(G_norm_at_exp) > 1e-9:
        r2 = np.corrcoef(exp_I, G_norm_at_exp)[0, 1]**2
    else:
        r2 = 0.0

    ax.legend(fontsize=10, facecolor='white', framealpha=0.9, loc='upper left')

    text_str = f"$R^2 = {r2:.4f}$\nMAE = {mae_100na*100:.2f} nA\nRMSE = {rmse_100na*100:.2f} nA"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    out_path = Path(__file__).parent / 'figuras' / 'fig_14b_ltp_ltd_jo2010.png'
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print(f"Computed Metrics for Jo 2010 LTP/LTD:")
    print(f"R2: {r2:.4f}, MAE: {mae_100na*100:.2f} nA")

if __name__ == "__main__":
    plot_stdp()
    plot_ltp_ltd()
    print("Figuras de validación Jo 2010 en blanco generadas correctamente.")
