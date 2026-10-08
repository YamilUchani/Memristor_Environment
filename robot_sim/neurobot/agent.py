"""
Agente neuromórfico con R-STDP en forma REINFORCE.

Regla de aprendizaje:
    ΔX[i,j] = η · E[i,j] · (R_ep - b)
    E[i,j] = pre_i · (1_{j == col} - π_j)

donde:
    pre_i   ∈ {0,1}  → fila activa en el encoder
    col              → columna elegida en este episodio
    π_j              → probabilidad softmax de cada columna
    R_ep             → retorno del episodio
    b                → media móvil de los últimos K retornos

Esto es la forma exacta del gradiente de política para una
política softmax, con la elegibilidad confinada a un paso
(el que eligió la acción).
"""
from collections import deque

import numpy as np

from neurobot.encoder import SensorEncoder
from neurobot.decoder import ActionDecoder
from neurobot.crossbar_brain import CrossbarBrain


class NeuromorphicAgent:
    def __init__(self,
                 obs_dim: int,
                 n_actions: int,
                 N: int = 8,
                 M: int = 4,
                 V_read: float = 0.1,
                 dt: float = 1e-4,
                 k_wta: int = 1,
                 steps_per_action: int = 1,
                 seed: int = 0,
                 # --- aprendizaje ---
                 eta: float = 1e-3,
                 temp_init: float = 1.0,
                 temp_min: float = 0.3,
                 temp_decay: float = 0.999,
                 baseline_window: int = 50,
                 epsilon: float = 0.0,
                 json_memristor_nv: str = "last_session.json",
                 json_memristor_v: str = "memristor_volatile.json",
                 json_lif: str = "lif_config.json"):
        
        # 1. Cerebro principal (Memristores No Volátiles para Pesos a largo plazo)
        self.brain = CrossbarBrain(N=N, M=M, seed=seed, scheme="V2",
                                   json_memristor=json_memristor_nv, json_lif=json_lif)
        
        # 2. Matriz de Trazas (Memristores Volátiles para Elegibilidad a corto plazo)
        from neurobot.lab_bridge import make_memristor
        self.volatile_trace = [[make_memristor(filename=json_memristor_v) for _ in range(M)] for _ in range(N)]
        
        self.encoder = SensorEncoder(obs_dim=obs_dim, N=N,
                                     V_read=V_read, seed=seed)
        self.decoder = ActionDecoder(n_actions=n_actions, M=M,
                                     sample=True,
                                     temperature=temp_init)

        self.n_actions = int(n_actions)
        self.dt = float(dt)
        self.k_wta = int(k_wta)
        self.steps_per_action = int(steps_per_action)

        self.eta = float(eta)
        self.temp_init = float(temp_init)
        self.temp_min = float(temp_min)
        self.temp_decay = float(temp_decay)
        self.epsilon = float(epsilon)

        self.reward_history = deque(maxlen=int(baseline_window))
        self.baseline = 0.0
        self.episode_count = 0
        self._E_accum = np.zeros((N, M))

        # Diagnóstico
        self.stats = {
            "dX_abs_mean": [],
            "signal": [],
            "R_ep": [],
        }

        self._last = {}

    # ------------------------------------------------------------------
    def act(self, obs):
        """
        Un ciclo perceptivo. Guarda lo necesario para learn().
        """
        V_rows = self.encoder.encode(obs)
        pre = self.encoder.last_pre_spikes()

        # Brain: acumular sobre steps_per_action si > 1
        spikes_acc = np.zeros(self.brain.M, dtype=bool)
        I_acc = np.zeros(self.brain.M)
        sneak_acc = 0.0
        winners = np.array([], dtype=int)

        for _ in range(self.steps_per_action):
            spikes, I_col, winners, sneak = self.brain.step(
                V_rows, dt=self.dt, k_wta=self.k_wta
            )
            spikes_acc |= spikes
            I_acc += I_col
            sneak_acc += sneak

        # ε-greedy opcional sobre columnas activas
        if self.epsilon > 0.0 and np.random.rand() < self.epsilon:
            active = np.where(spikes_acc)[0]
            pool = active if active.size > 0 else np.arange(self.brain.M)
            col = int(np.random.choice(pool))
            action = int(self.decoder.col_to_action[col])
            probs = np.zeros(self.brain.M)
            probs[col] = 1.0
        else:
            action = self.decoder.decode(I_acc)
            col = self.decoder.last_col()
            probs = self.decoder.last_probs()

        # --- Elegibilidad REINFORCE de este paso ---
        indicator = np.zeros(self.brain.M)
        indicator[col] = 1.0
        delta = indicator - probs                    # (M,)
        
        # 1. Update matemático tradicional (IMPRESCINDIBLE para que REINFORCE converja)
        self._E_accum += np.outer(pre.astype(float), delta)
        
        # 2. Update FÍSICO con Memristores Volátiles (Fase 9 de la Tesis)
        # Solo actualizamos si la señal es significativa para evitar 576 llamadas en vano.
        # Reducimos a muestreo: actualizamos 1 de cada 4 pasos (trade-off rendimiento/fidelidad)
        if self.episode_count % 4 == 0:
            for i in range(self.brain.N):
                for j in range(self.brain.M):
                    v_app = float(pre[i]) * float(delta[j]) * 2.0
                    self.volatile_trace[i][j].update(voltage=v_app, dt=self.dt)

        self._last = {
            "V_rows": V_rows,
            "pre": pre,
            "spikes": spikes_acc,
            "winners": winners,
            "I_col": I_acc,
            "sneak": sneak_acc / self.steps_per_action,
            "action": action,
            "col": col,
            "probs": probs,
        }
        return action

    # ------------------------------------------------------------------
    def learn(self, R_ep: float):
        """
        Una actualización por episodio. REINFORCE con baseline robusto.
        Usa la elegibilidad acumulada sobre TODOS los act() del episodio.
        """
        R_ep = float(R_ep)

        # Baseline: media móvil
        self.reward_history.append(R_ep)
        self.baseline = float(np.mean(self.reward_history))
        signal = R_ep - self.baseline

        # Si el episodio fue casi perfecto (con la nueva recompensa centrada, el máximo
        # real es ~430-460 en lugar de 500), no actualizamos pesos.
        if R_ep >= 430.0:
            return

        # Update con la elegibilidad acumulada
        dX = -self.eta * self._E_accum * signal
        self._last_signal = signal  # Guardar señal para el estrés
        
        # En lugar de clipping elemento a elemento (que destruye la suma cero de softmax),
        # usamos escalado global (Gradient Norm Clipping) para preservar la dirección.
        max_grad = float(np.max(np.abs(dX)))
        if max_grad > 0.05:
            dX = dX * (0.05 / max_grad)
        
        self.brain.X = np.clip(self.brain.X + dX, 0.0, 1.0)
        
        # DEBUG: Imprimir si la matriz X y G se mueven realmente
        if self.episode_count % 10 == 0:
            print(f"[DEBUG Ep {self.episode_count}] max|dX|: {np.max(np.abs(dX)):.2e}, std(X): {np.std(self.brain.X):.2e}, std(G): {np.std(self.brain.G):.2e}, signal: {signal:.2f}")

        # Logs
        self.stats["dX_abs_mean"].append(float(np.abs(dX).mean()))
        self.stats["signal"].append(float(signal))
        self.stats["R_ep"].append(R_ep)

    # ------------------------------------------------------------------
    def end_episode(self):
        """Se llama tras learn(). Decae temperatura, resetea LIF y Volátiles."""
        self.episode_count += 1
        
        # Temperatura con decaimiento suave.
        # Si ocurre un fallo severo, se añade un pequeño toque de exploración (+0.02)
        # acotado a 0.25 para salir de trampas locales sin destruir la política aprendida.
        T_curr = self.decoder.temperature
        if hasattr(self, '_last_signal') and self._last_signal < -30:
            T = min(0.25, T_curr + 0.02)
        else:
            T = max(self.temp_min, T_curr * self.temp_decay)
            
        self.decoder.set_temperature(T)
        self.brain.reset(keep_weights=True)
        self._E_accum.fill(0.0)
        
        # Reset de los memristores volátiles a su estado de equilibrio
        for i in range(self.brain.N):
            for j in range(self.brain.M):
                self.volatile_trace[i][j].x = self.volatile_trace[i][j].model_config.x0
                
        self._last = {}

    # ------------------------------------------------------------------
    def last_step(self):
        return self._last

    def current_temperature(self):
        return self.decoder.temperature
