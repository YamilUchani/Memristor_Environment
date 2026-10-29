"""
verify3.py
==========
Verificación del Sensor Neuromórfico Repetible (Memristor Volátil en Serie R_S).
"""
import numpy as np
from neurolab.core.config import DeviceIdentity, ElectricalConfig, StrukovConfig
from neurolab.core.memristor import Memristor
from neurolab.devices.models.strukov import StrukovMathModel
from neurolab.devices.realism.volatile import VolatileDecayModifier
from neurolab.devices.realism.window import BiolekWindowModifier
from neurolab.neurons.config import LIFConfig
from neurolab.neurons.lif import LIFNeuron

def run_series_sensor_test():
    # Memristor en Serie (Sensor): R_ON = 100 Ω, R_OFF = 16 kΩ, x0 = 0.50
    # Volátil: tau_relax = 0.5 s, x_eq = 0.05, mu_v = 1e-14
    mem = Memristor(
        math_model=StrukovMathModel(),
        electrical=ElectricalConfig(r_on=100.0, r_off=16000.0, initial_state=0.50),
        identity=DeviceIdentity(device_name="series_sensor", device_family="ox", model_name="strukov"),
        model_config=StrukovConfig(D=10e-9, mu_v=1e-14),
        modifiers=[
            BiolekWindowModifier(p=3),
            VolatileDecayModifier(tau_relax=0.5, x0_override=0.05)
        ],
        clip_x=True,
    )

    neuron = LIFNeuron(LIFConfig(
        c_m=100e-9, r_leak=1e6, r_series=100e3,
        v_rest=0.0, v_th=1.0, v_reset=0.0, t_ref=20e-3,
    ))

    dt = 1e-5
    t = np.arange(0.0, 5.0, dt)
    v_in = np.where((t * 40.0) % 1.0 < 0.2, 5.0, 0.0)

    spikes = []
    r_series_hist = []

    for k in range(len(t)):
        v_m = neuron.v_membrane
        v_drop_s = max(v_in[k] - v_m, 0.0)
        i_in = mem.step(v_drop_s, dt)
        r_s = mem.resistance
        r_series_hist.append(r_s)
        i_leak = (v_m - neuron.config.v_rest) / neuron.config.r_leak

        neuron.t += dt
        neuron.has_spiked = False
        if neuron.refractory_time_left > 0.0:
            neuron.refractory_time_left -= dt
            neuron.v_membrane += (-i_leak / neuron.config.c_m) * dt
        else:
            neuron.v_membrane += ((i_in - i_leak) / neuron.config.c_m) * dt
            if neuron.v_membrane >= neuron.config.v_th:
                neuron.spike_times.append(neuron.t)
                spikes.append(t[k])
                neuron.v_membrane = neuron.config.v_reset
                neuron.refractory_time_left = neuron.config.t_ref

    print("=" * 70)
    print(f"=== Sensor Neuromórfico Repetible en Serie (R_S) ===")
    print(f"Total Spikes producidos: {len(spikes)}")
    print(f"R_S inicial: {r_series_hist[0]/1e3:.1f} kOhm")
    print(f"R_S mínima alcanzada: {min(r_series_hist)/1e3:.1f} kOhm")
    print(f"R_S final (relajada): {r_series_hist[-1]/1e3:.1f} kOhm")
    print("=" * 70)

if __name__ == "__main__":
    run_series_sensor_test()
