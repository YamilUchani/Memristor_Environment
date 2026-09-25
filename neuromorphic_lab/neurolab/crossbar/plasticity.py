"""
neurolab/crossbar/plasticity.py
================================
Reglas de plasticidad sináptica para crossbar.

Implementa:
- Trace: traza de elegibilidad con decaimiento exponencial
- STDP: Spike-Timing-Dependent Plasticity
- R-STDP: Reward-modulated STDP

Todos los modos calculan Δw LOCALMENTE por celda.
No hay decodificador de dirección.
"""
import numpy as np
from dataclasses import dataclass


# =====================================================================
# TRAZA DE ELEGIBILIDAD
# =====================================================================

class Trace:
    """
    Traza de elegibilidad con decaimiento exponencial.
    
    En hardware real, esto es un capacitor que se carga cuando hay
    un spike y se descarga lentamente a través de una resistencia.
    
    dTrace/dt = -Trace/τ + spike
    Trace(t+dt) = Trace(t)·exp(-dt/τ) + spike(t)
    """
    
    def __init__(self, n_channels: int, tau: float = 20e-3):
        """
        Parameters
        ----------
        n_channels : int
            Número de trazas (filas o columnas).
        tau : float
            Constante de tiempo (s). Biológico: 20 ms.
        """
        self.n = n_channels
        self.tau = tau
        self.values = np.zeros(n_channels)
    
    def step(self, spikes: np.ndarray, dt: float):
        """
        Actualiza las trazas con decaimiento + nuevos spikes.
        dt se clampa a [1μs, 1s] para estabilidad numérica.
        """
        if dt <= 0.0:
            return
        dt = float(np.clip(dt, 1e-6, 1.0))
        decay = np.exp(-dt / self.tau)
        self.values = self.values * decay + np.asarray(spikes, dtype=float)
    
    def reset(self):
        """Reinicia todas las trazas a 0."""
        self.values[:] = 0.0


# =====================================================================
# REGLA STDP
# =====================================================================

@dataclass
class STDPConfig:
    """Configuración de STDP."""
    A_plus: float = 0.1e-6       # Amplitud LTP (0.1 μS por evento)
    A_minus: float = 0.08e-6     # Amplitud LTD (0.08 μS por evento)
    tau_plus: float = 20e-3     # Constante pre (s)
    tau_minus: float = 20e-3    # Constante post (s)
    eta: float = 1.0            # Tasa de aprendizaje
    G_min: float = 1e-6         # Límite inferior (S)
    G_max: float = 500e-6       # Límite superior (S)


class STDPRule:
    """
    Regla STDP implementada con trazas de elegibilidad.
    """
    
    def __init__(self, config: STDPConfig = None):
        self.cfg = config or STDPConfig()
        self.trace_pre = None
        self.trace_post = None
        self.G_min = self.cfg.G_min
        self.G_max = self.cfg.G_max
        self.threshold_product = 0.01
        self.threshold_dG = 1e-9
    
    def reset(self, n_rows: int, n_cols: int):
        """Reinicia trazas."""
        self.trace_pre = Trace(n_rows, self.cfg.tau_plus)
        self.trace_post = Trace(n_cols, self.cfg.tau_minus)
    
    def compute_delta_G(self, spike_pre: np.ndarray,
                         spike_post: np.ndarray) -> np.ndarray:
        """
        Calcula dG (S) por evento.
        IMPORTANTE: spike_pre y spike_post DEBEN ser eventos binarios del
        MISMO tick. Si son niveles sostenidos, la traza satura y el
        aprendizaje degenera.
        """
        if self.trace_pre is None or self.trace_post is None:
            self.reset(len(spike_pre), len(spike_post))

        pre = np.asarray(spike_pre, dtype=float)
        post = np.asarray(spike_post, dtype=float)

        term_ltp = np.outer(self.trace_pre.values, post)
        term_ltd = np.outer(pre, self.trace_post.values)

        term_ltp[term_ltp < self.threshold_product] = 0.0
        term_ltd[term_ltd < self.threshold_product] = 0.0

        dG = self.cfg.eta * (self.cfg.A_plus * term_ltp - self.cfg.A_minus * term_ltd)
        dG[np.abs(dG) < self.threshold_dG] = 0.0
        return dG
    
    def apply(self, G_matrix: np.ndarray,
              spike_pre: np.ndarray,
              spike_post: np.ndarray,
              dt: float) -> np.ndarray:
        if self.trace_pre is None or self.trace_post is None:
            self.reset(len(spike_pre), len(spike_post))

        self.trace_pre.step(spike_pre, dt)
        self.trace_post.step(spike_post, dt)
        dG = self.compute_delta_G(spike_pre, spike_post)
        return dG


# =====================================================================
# REGLA R-STDP (REWARD-MODULATED STDP)
# =====================================================================

@dataclass
class RSTDPConfig(STDPConfig):
    """Configuración de R-STDP (extiende STDPConfig)."""
    R: float = 0.0              # Recompensa (+1, 0, -1)
    use_reward: bool = True     # Activar modulación por recompensa
    tau_eligibility: float = 1.0  # Persistencia de elegibilidad para recompensa retrasada (s)


class RSTDPRule(STDPRule):
    """
    Regla R-STDP: STDP modulada por recompensa global R.
    """

    def __init__(self, config: RSTDPConfig = None):
        super().__init__(config or RSTDPConfig())
        self.cfg: RSTDPConfig = self.cfg
        self.eligibility = None
        self._pending_reward = 0.0

    def reset(self, n_rows: int, n_cols: int):
        """Reinicia trazas temporales y la elegibilidad por sinapsis."""
        super().reset(n_rows, n_cols)
        self.eligibility = np.zeros((n_rows, n_cols), dtype=float)
        self._pending_reward = 0.0

    def set_reward(self, R):
        """Establece la señal de recompensa (global escalar o vectorial por columna)."""
        if isinstance(R, (list, tuple, np.ndarray)):
            self.cfg.R = np.asarray(R, dtype=float)
        else:
            self.cfg.R = float(R)

    def apply(self, G_matrix, spike_pre, spike_post, dt):
        """
        Aplica R-STDP modulado por coincidencia temporal de trazas y recompensa.
        """
        if self.trace_pre is None or self.trace_post is None:
            self.reset(len(spike_pre), len(spike_post))

        # 1. Actualizar trazas primero
        self.trace_pre.step(spike_pre, dt)
        self.trace_post.step(spike_post, dt)

        # 2. Calcular dG base con trazas pobladas y conservar elegibilidad.
        dG_base = self.compute_delta_G(spike_pre, spike_post)
        if self.eligibility is None or self.eligibility.shape != dG_base.shape:
            self.eligibility = np.zeros_like(dG_base)
        tau_e = max(1e-6, float(self.cfg.tau_eligibility))
        self.eligibility *= np.exp(-float(np.clip(dt, 1e-6, 1.0)) / tau_e)
        self.eligibility += dG_base

        # 3. Modular por recompensa
        R_val = np.asarray(self.cfg.R)
        if np.all(np.abs(R_val) < 1e-6):
            R_val = np.asarray(self._pending_reward)
            if np.all(np.abs(R_val) < 1e-6):
                return np.zeros_like(G_matrix)

        # Una recompensa pulsada desde la GUI puede llegar antes del spike
        # post. Se conserva hasta que haya elegibilidad que modular.
        if not np.any(np.abs(self.eligibility) > 1e-15):
            self._pending_reward = R_val.item() if R_val.ndim == 0 else R_val.copy()
            self.cfg.R = 0.0
            return np.zeros_like(G_matrix)

        if R_val.ndim > 0 and len(R_val) == G_matrix.shape[1]:
            dG_final = self.eligibility * R_val[None, :]
        else:
            dG_final = float(self.cfg.R) * self.eligibility

        # Consumir la recompensa para que actúe en este tick
        self.cfg.R = 0.0
        self._pending_reward = 0.0
        self.eligibility.fill(0.0)

        return dG_final


