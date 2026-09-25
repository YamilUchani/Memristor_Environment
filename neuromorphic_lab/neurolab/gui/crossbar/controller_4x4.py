
import numpy as np
from typing import Dict, Any, Optional
from neurolab.gui.crossbar_elements import VisualElement, get_G_matrix, compute_currents
from neurolab.crossbar.plasticity import STDPRule, RSTDPRule, STDPConfig, RSTDPConfig
from neurolab.neurons.lif import LIFNeuron
from neurolab.neurons.config import LIFConfig

class Crossbar4x4Controller:
    """
    Controlador separado (Patrón MVC) para aislar la lógica del motor físico,
    la integración de neuronas LIF, el algoritmo WTA y la plasticidad (STDP).
    """
    def __init__(self, elements: Dict[str, VisualElement], crossbar_core):
        self.elements = elements
        self.crossbar = crossbar_core
        self.sim_time = 0.0

        # STDP con tasas realistas
        stdp_cfg = STDPConfig(
            A_plus=0.1e-6, A_minus=0.08e-6, G_min=1e-6, G_max=500e-6,
            tau_plus=20e-3, tau_minus=20e-3
        )
        self.stdp_rule = STDPRule(stdp_cfg)
        self.stdp_rule.reset(4, 4)

        rstdp_cfg = RSTDPConfig(
            A_plus=0.1e-6, A_minus=0.08e-6, G_min=1e-6, G_max=500e-6, R=0.0
        )
        self.rstdp_rule = RSTDPRule(rstdp_cfg)
        self.rstdp_rule.reset(4, 4)

        self.spike_pre = np.zeros(4)
        self.spike_post = np.zeros(4)

        # WTA first-to-fire
        self.winner_j = None
        self.winner_lock_time = 0.0
        self.WINNER_LOCK_DURATION = 0.20
        self.refractory_time = np.zeros(4)
        self.REFRACTORY_DURATION = 0.15

        # Motor Físico Neuronas LIF Reales
        lif_cfg = LIFConfig(
            c_m=100e-9, r_leak=1e6, v_th_base=1.0, v_th=1.0,
            v_adapt_inc=0.15, tau_adapt=0.05, t_ref=0.15, v_rest=0.0, v_reset=0.0
        )
        self.lif_neurons = [LIFNeuron(lif_cfg) for _ in range(4)]
        
        # Factores de escala con unidades explícitas
        # Ref: modelo fenomenológico dW/dt = k*I - W/tau (Strukov 2008 adaptado)
        self.MEM_VOLATILE_K = 1.6e4     # [1/(A·s)]  ganancia calibrada (I=200uA, tau=0.3s -> x ~ 1.0)
        self.MEM_VOLATILE_A = 1.6e4     # Alias para compatibilidad
        self.MEM_VOLATILE_TAU = 1e-3    # [s]        tiempo de relajación
        self.TIA_R = 50e3               # [Ω]        transimpedancia (50 kΩ)
        self.LIF_R_IN = 1e6             # [Ω]        resistencia de entrada LIF
        self.LIF_I_scale = self.TIA_R / self.LIF_R_IN
        self.WTA_GAIN = 0.5

        # Weight decay
        self.G_base = 69.4e-6
        self.G_decay_rate = 0.005
        self._gui_dirty = True

    def mark_gui_dirty(self):
        """Marca que la GUI ha actualizado conductancias manualmente."""
        self._gui_dirty = True

    def reset(self):
        self.sim_time = 0.0
        self.winner_j = None
        self.winner_lock_time = 0.0
        self.refractory_time = np.zeros(4)
        for neuron in self.lif_neurons:
            neuron.reset()
        self.stdp_rule.reset(4, 4)
        self.rstdp_rule.reset(4, 4)
        self._gui_dirty = True

    def set_reward(self, reward: float):
        self.rstdp_rule.set_reward(reward)

    def apply_programming_pulse(self, tg_r: int, tg_c: int, v_pulse: float, dt_pulse: float = None):
        """Aplica un pulso delegando a Crossbar y actualizando la UI."""
        if dt_pulse is None:
            dt_pulse = 0.03
        dt_pulse = max(1e-6, float(dt_pulse))

        for i in range(4):
            for j in range(4):
                m = self.elements.get(f'M{i+1}{j+1}')
                if m: self.crossbar.set_conductance(i, j, float(m.params.get('G', 69.4e-6)))

        self.crossbar.program_V2(
            tg_r, tg_c, V_program=v_pulse, dt=dt_pulse,
            isolate_half_select=False
        )

        G_mat = self.crossbar.G_matrix
        for i in range(4):
            for j in range(4):
                m = self.elements.get(f'M{i+1}{j+1}')
                if m:
                    G_new = G_mat[i, j]
                    m.params['G'] = G_new
                    m.params['G_11'] = G_new
                    R_new = 1.0 / max(1e-12, G_new)
                    m.params['R'] = R_new
                    RON = float(m.params.get('RON', 2000.0))
                    ROFF = float(m.params.get('ROFF', 16000.0))
                    x_calc = (ROFF - R_new) / (ROFF - RON) if ROFF != RON else 0.1
                    m.params['x'] = float(np.clip(x_calc, 0.01, 0.99))

    def step(self, dt: float, mode: str, plasticity_mode: str, is_animating: bool, V_rows: np.ndarray, V_cols: np.ndarray, spike_pre: Optional[np.ndarray] = None) -> Dict[str, Any]:
        self.sim_time += dt

        if spike_pre is None:
            spike_pre_arr = (V_rows > 0.5).astype(float)
        else:
            spike_pre_arr = np.asarray(spike_pre, dtype=float)

        # 1. Sincronizar conductancias GUI -> Core solo si la GUI fue editada manualmente o al inicio
        if self._gui_dirty or mode == "program_v2":
            for i in range(4):
                for j in range(4):
                    mem = self.elements.get(f'M{i+1}{j+1}')
                    if mem:
                        self.crossbar.set_conductance(i, j, float(mem.params.get('G', 69.4e-6)))
            self._gui_dirty = False

        # 2. Aplicar voltajes al backend físico del Crossbar
        self.crossbar.V_rows = V_rows
        self.crossbar.V_cols = V_cols
        V_matrix = V_rows[:, None] - V_cols[None, :]
        v_th_val = float(getattr(self.crossbar.cfg, 'V_th', 0.5))
        has_high_voltage = np.any(np.abs(V_matrix) >= v_th_val)

        if mode == "program_v2" or has_high_voltage:
            if mode == "program_v2":
                active_rows = np.flatnonzero(np.abs(V_rows) > 1e-12)
                active_cols = np.flatnonzero(np.abs(V_cols) > 1e-12)
                if len(active_rows) and len(active_cols):
                    self.crossbar.programming_target = (int(active_rows[0]), int(active_cols[0]))
            self.crossbar.update_memristors(dt)
            self.crossbar.programming_target = None

        # 3. LECTURA SIEMPRE A TRAVÉS DEL CORE (ÚNICA FUENTE DE VERDAD)
        if np.all(V_cols == 0):
            I_cols = self.crossbar.read(V_rows)
        else:
            I_cols = self.crossbar.read(V_rows - V_cols)
        self.crossbar.V_rows = V_rows
        self.crossbar.V_cols = V_cols

        # 4. Sincronizar conductancias y estado Core -> GUI
        G_mat = self.crossbar.G_matrix
        for i in range(4):
            for j in range(4):
                mem = self.elements.get(f'M{i+1}{j+1}')
                if mem:
                    G_new = G_mat[i, j]
                    mem.params['G'] = G_new
                    mem.params['G_11'] = G_new
                    R_new = 1.0 / max(1e-12, G_new)
                    mem.params['R'] = R_new
                    mem.params['R_11'] = R_new
                    RON = float(mem.params.get('RON', 2000.0))
                    ROFF = float(mem.params.get('ROFF', 16000.0))
                    x_calc = (ROFF - R_new) / (ROFF - RON) if ROFF != RON else 0.1
                    mem.params['x'] = float(np.clip(x_calc, 0.01, 0.99))
        for j in range(4):
            mv = self.elements.get(f'M_v{j+1}')
            if not mv: continue
            p_v = mv.params
            tau_rel = float(p_v.get('volatile_tau_relax', self.MEM_VOLATILE_TAU))
            x_v = float(p_v.get('x', 0.05))
            x_eq = float(p_v.get('x0', 0.05))

            # FIX: Restaurar hacia x_eq con ganancia calibrada k = 1.6e4
            dxdt_v = self.MEM_VOLATILE_K * I_cols[j] - ((x_v - x_eq) / max(1e-4, tau_rel))
            p_v['x'] = float(np.clip(x_v + dxdt_v * dt, 0.001, 1.0))

        self.spike_pre = spike_pre_arr
        
        # Si estamos en modo de programación V/2, los LIF y la WTA están DESCONECTADOS
        if mode == "program_v2":
            self.winner_j = None
            spike_post = np.zeros(4, dtype=bool)
            for j in range(4):
                LIF_ui = self.elements[f'LIF_{j+1}']
                Act = self.elements[f'Act_{j+1}']
                LIF_ui.params['is_winner'] = False
                Act.params['is_winner'] = False
                Act.params['action'] = '⚙️ PROG V/2'
            return {
                "I_cols": I_cols,
                "spike_post": spike_post,
                "winner_j": None,
            }

        # Soft WTA e Inhibición Lateral Analógica Continuos
        winner_active = (self.winner_j is not None and self.sim_time < self.winner_lock_time)

        # LIF & WTA delegando al motor neuronal LIFNeuron
        spike_post = np.zeros(4, dtype=bool)
        spikes_this_tick = np.zeros(4)

        for j in range(4):
            LIF_ui = self.elements[f'LIF_{j+1}']
            neuron = self.lif_neurons[j]
            mv = self.elements.get(f'M_v{j+1}')
            x_v = float(mv.params.get('x', 0.05)) if mv else 1.0

            # Inhibición lateral suave: inyección de corriente negativa (sin reset duro de V_m)
            if winner_active and j != self.winner_j:
                I_inhib = -self.WTA_GAIN * (I_cols[self.winner_j] * self.TIA_R / self.LIF_R_IN)
            else:
                I_inhib = 0.0

            # Inyectar corriente física modulada por la celda volátil M_v en serie + inhibición
            I_col_eff = I_cols[j] * (0.1 + 0.9 * x_v)
            V_TIA = I_col_eff * self.TIA_R          # [V]
            I_syn = (V_TIA / self.LIF_R_IN) + I_inhib
            has_spiked = neuron.step(current_input=I_syn, dt=dt, t=self.sim_time)

            # Sincronizar UI con el motor físico
            LIF_ui.params['V_m'] = neuron.V_m
            LIF_ui.params['V_th'] = neuron.v_th

            if has_spiked:
                spikes_this_tick[j] = 1.0
                spike_post[j] = True
                LIF_ui.params['spike_count'] = int(LIF_ui.params.get('spike_count', 0)) + 1
                self.refractory_time[j] = self.sim_time + neuron.config.t_ref

        # Actualizar ganador según la neurona que acaba de disparar con mayor corriente
        recent_spikers = np.where(spikes_this_tick > 0.5)[0]
        if len(recent_spikers) > 0:
            self.winner_j = int(recent_spikers[np.argmax([I_cols[j] for j in recent_spikers])])
            self.winner_lock_time = self.sim_time + 0.050  # Ventana fluida de 50 ms
        elif self.sim_time >= self.winner_lock_time:
            self.winner_j = None

        for j in range(4):
            LIF = self.elements[f'LIF_{j+1}']
            Act = self.elements[f'Act_{j+1}']
            if self.winner_j is not None and j == self.winner_j:
                LIF.params['is_winner'] = True
                Act.params['is_winner'] = True
                Act.params['action'] = '🏆 GANADOR'
            else:
                LIF.params['is_winner'] = False
                Act.params['is_winner'] = False
                is_refractory = self.sim_time < self.refractory_time[j]
                if is_refractory:
                    Act.params['action'] = '⏳ REFRACTARIO'
                elif winner_active:
                    Act.params['action'] = '🚫 INHIBIDO'
                else:
                    Act.params['action'] = 'listo'

        # Plasticity STDP / R-STDP
        if plasticity_mode in ("stdp", "rstdp") and mode == "read":
            G_matrix = get_G_matrix(self.elements, 4, 4)
            self.spike_post = spikes_this_tick.copy()
            # En STDP competitivo (WTA), la inhibición lateral suprime los spikes post-sinápticos de las columnas no ganadoras
            if self.winner_j is not None:
                for j in range(4):
                    if j != self.winner_j:
                        self.spike_post[j] = 0.0

            if plasticity_mode == "rstdp":
                dG = self.rstdp_rule.apply(G_matrix, self.spike_pre, self.spike_post, dt)
            else:
                dG = self.stdp_rule.apply(G_matrix, self.spike_pre, self.spike_post, dt)

            G_min = self.stdp_rule.G_min
            G_max = self.stdp_rule.G_max
            
            for i in range(4):
                for j in range(4):
                    if abs(dG[i, j]) > 1e-12:
                        mem = self.elements.get(f'M{i+1}{j+1}')
                        if mem:
                            G_old = float(mem.params.get('G', self.G_base))
                            saturation = max(0.0, (G_max - G_old) / (G_max - G_min)) if dG[i, j] > 0 else max(0.0, (G_old - G_min) / (G_max - G_min))
                            G_new = float(np.clip(G_old + dG[i, j] * saturation, G_min, G_max))
                            R_new = 1.0 / max(1e-12, G_new)
                            RON = float(mem.params.get('RON', 2000.0))
                            ROFF = float(mem.params.get('ROFF', 16000.0))
                            x_calc = float(np.clip((ROFF - R_new) / (ROFF - RON), 0.01, 0.99)) if ROFF != RON else 0.1
                            
                            mem.params['G'] = G_new
                            mem.params['G_11'] = G_new
                            mem.params['R'] = R_new
                            mem.params['R_11'] = R_new
                            mem.params['x'] = x_calc
                            mem.params['x0'] = x_calc
                            self.crossbar.set_conductance(i, j, G_new)

        # Weight Decay
        if plasticity_mode in ("stdp", "rstdp") and self.winner_j is not None:
            for i in range(4):
                for j in range(4):
                    if j != self.winner_j:
                        mem = self.elements.get(f'M{i+1}{j+1}')
                        if mem:
                            G_old = float(mem.params.get('G', self.G_base))
                            G_new = G_old + self.G_decay_rate * (self.G_base - G_old)
                            mem.params['G'] = float(G_new)
                            R_new = 1.0 / max(1e-12, G_new)
                            RON = float(mem.params.get('RON', 2000.0))
                            ROFF = float(mem.params.get('ROFF', 16000.0))
                            x_calc = float(np.clip((ROFF - R_new) / (ROFF - RON), 0.01, 0.99)) if ROFF != RON else 0.1
                            mem.params['x'] = x_calc
                            mem.params['R'] = R_new
                            mem.params['G_11'] = G_new
                            mem.params['R_11'] = R_new
                            self.crossbar.set_conductance(i, j, G_new)

        return {
            "I_cols": I_cols,
            "spike_post": spike_post,
            "winner_j": self.winner_j,
        }
