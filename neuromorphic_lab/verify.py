import numpy as np
from neurolab.core.config import DeviceIdentity, ElectricalConfig, StrukovConfig
from neurolab.core.memristor import Memristor
from neurolab.devices.models.strukov import StrukovMathModel
from neurolab.devices.realism.volatile import VolatileDecayModifier
from neurolab.neurons.config import LIFConfig
from neurolab.neurons.lif import LIFNeuron

mem = Memristor(
    math_model=StrukovMathModel(),
    electrical=ElectricalConfig(r_on=1e3, r_off=1e6, initial_state=0.99),
    identity=DeviceIdentity(device_name="leak", device_family="ox", model_name="strukov"),
    model_config=StrukovConfig(D=10e-9, mu_v=1e-16),
    modifiers=[VolatileDecayModifier(tau_relax=0.10, x0_override=0.05)],
    clip_x=True,
)

neuron = LIFNeuron(LIFConfig(
    c_m=100e-9, r_leak=1e6, r_series=100e3,
    v_rest=0.0, v_th=1.0, v_reset=-0.20, t_ref=0.0,
))

dt = 1e-5
t = np.arange(0.0, 0.5, dt)
v_in = np.where((t * 40.0) % 1.0 < 0.2, 5.0, 0.0)

v_m_arr = np.zeros(len(t))
R_arr = np.zeros(len(t))
spikes = []

for k in range(len(t)):
    v_m = neuron.v_membrane
    mem.step(v_m - neuron.config.v_rest, dt)
    R_leak = mem.resistance
    i_in = max(v_in[k] - v_m, 0.0) / neuron.config.r_series
    i_leak = (v_m - neuron.config.v_rest) / R_leak

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

    v_m_arr[k] = neuron.v_membrane
    R_arr[k] = R_leak

def flat_ms(k0):
    thr = neuron.config.v_reset + 0.05
    for j in range(k0, len(t)):
        if v_m_arr[j] > thr:
            return (t[j] - t[k0]) * 1e3
    return float("nan")

print("=" * 60)
print("Spikes:", len(spikes))
print(f"{'#':<4}{'t (ms)':<12}{'flat (ms)':<12}{'R_leak (kOhm)':<14}")
print("-" * 60)
for i, ts in enumerate(spikes):
    k = int(round(ts / dt))
    print(f"{i:<4}{ts*1e3:<12.1f}{flat_ms(k):<12.1f}{R_arr[k]/1e3:<14.1f}")
print("=" * 60)
