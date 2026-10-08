"""
Crossbar N×M como cerebro del agente.
Ata: solver MNA del lab + LIFNeuron del lab + estado memristivo propio.
Añade lo que el lab NO tiene: WTA, esquemas V/2 y V/3, persistencia de G.
"""
import json
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from neurobot import lab_bridge

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)


class CrossbarBrain:
    """
    Estado primario: matriz `X` en [0,1] (variable de estado memristiva).
    La conductancia se deriva: G(x) = 1 / (r_on + x·(r_off - r_on)).
    """

    def __init__(self,
                 N: int,
                 M: int,
                 R_wire: float = 1e-3,
                 R_sense: float = 1e2,
                 wta_inhibit: float = 5e-3,
                 scheme: str = "V2",
                 seed: Optional[int] = None,
                 json_memristor: str = "last_session.json",
                 json_lif: str = "lif_config.json"):

        if scheme not in ("V2", "V3"):
            raise ValueError("scheme debe ser 'V2' o 'V3'")

        self.N, self.M = int(N), int(M)
        self.R_wire = float(R_wire)
        self.R_sense = float(R_sense)
        self.wta_inhibit = float(wta_inhibit)
        self.scheme = scheme

        # Config del lab usando los JSON elegidos
        self.elec = lab_bridge.get_electrical_config(filename=json_memristor)
        
        # Leemos el kind del json o asumimos strukov por defecto (simplificado para RoboSim)
        kind = "prezioso" if "prezioso" in json_memristor.lower() else "strukov"
        self.dev_cfg = lab_bridge.get_device_config(kind=kind, filename=json_memristor)

        # Estado: X en [0,1], G derivada
        rng = np.random.default_rng(seed)
        self.X = np.clip(
            self.elec.initial_state + rng.normal(0.0, 0.01, (self.N, self.M)),
            0.0, 1.0,
        )

        # Un memristor por celda, con el modelo del lab (para update físico)
        self.cells = [[lab_bridge.make_memristor(filename=json_memristor)
                       for _ in range(self.M)]
                      for _ in range(self.N)]

        # Una LIF por columna
        self.neurons = [lab_bridge.make_lif(filename=json_lif) for _ in range(self.M)]

        # Historial (maxlen evita memory leak en simulaciones largas)
        _H = 2000
        self.history = {
            "I_col":       deque(maxlen=_H),
            "V_rows":      deque(maxlen=_H),
            "spikes":      deque(maxlen=_H),
            "sneak_ratio": deque(maxlen=_H),
            "G_mean":      deque(maxlen=_H),
        }

    def save_weights(self, filepath: str):
        """Guarda la matriz de estado (pesos) en la ruta completa especificada."""
        np.save(filepath, self.X)
        print(f"[ok] Pesos (X) guardados en: {filepath}")

    def load_weights(self, filepath: str):
        """Carga la matriz de estado (pesos) desde un archivo .npy"""
        if Path(filepath).exists():
            data = np.load(filepath)
            if data.shape == self.X.shape:
                self.X = data
                print(f"[ok] Pesos (X) cargados desde: {filepath}")
            else:
                print(f"[error] Error de dimensiones al cargar pesos: {data.shape} != {self.X.shape}")
        else:
            print(f"[error] No se encontró el archivo de pesos: {filepath}")

    # ----------------------------------------------------------------
    # G(x) — conductancia derivada
    # ----------------------------------------------------------------
    @property
    def G(self) -> np.ndarray:
        r_on = self.elec.r_on
        r_off = self.elec.r_off
        R = r_on + self.X * (r_off - r_on)
        return 1.0 / R

    @property
    def G_min(self) -> float:
        return 1.0 / self.elec.r_off

    @property
    def G_max(self) -> float:
        return 1.0 / self.elec.r_on

    # ----------------------------------------------------------------
    # Lectura (MNA con sneak paths)
    # ----------------------------------------------------------------
    def read(self, V_rows: np.ndarray):
        V_rows = np.asarray(V_rows, dtype=float)
        G = self.G

        I_col = lab_bridge.solve_mna(
            G=G, V_rows=V_rows,
            R_wire=self.R_wire, R_sense=self.R_sense,
        )

        # Ideal para sneak_ratio: I_ideal_j = Σ_i G_ij · V_row_i
        I_ideal = G.T @ V_rows
        denom = np.linalg.norm(I_ideal) + 1e-30
        sneak_ratio = float(np.linalg.norm(I_col - I_ideal) / denom)

        self.history["I_col"].append(I_col.copy())
        self.history["V_rows"].append(V_rows.copy())
        self.history["sneak_ratio"].append(sneak_ratio)
        self.history["G_mean"].append(float(G.mean()))

        return I_col, sneak_ratio

    # ----------------------------------------------------------------
    # Paso completo: read + LIF + WTA
    # ----------------------------------------------------------------
    def step(self, V_rows: np.ndarray, dt: float = 1e-4, k_wta: int = 1):
        I_col, sneak = self.read(V_rows)

        spikes = np.zeros(self.M, dtype=bool)
        for j, n in enumerate(self.neurons):
            spikes[j] = n.step(current_input=float(I_col[j]), dt=dt)

        winners = self._apply_wta(spikes, I_col, k=k_wta)
        self.history["spikes"].append(spikes.copy())

        return spikes, I_col, winners, sneak

    # ----------------------------------------------------------------
    # WTA suave: los perdedores reciben inhibición lateral
    # ----------------------------------------------------------------
    def _apply_wta(self, spikes: np.ndarray, I_col: np.ndarray, k: int):
        active = np.where(spikes)[0]
        if active.size == 0:
            return np.array([], dtype=int)

        # La(s) k columnas con mayor corriente ganan
        winners = active[np.argsort(I_col[active])[-k:]]
        losers = np.setdiff1d(np.arange(self.M), winners)

        # Inhibición: bajón de potencial en las LIF perdedoras
        for j in losers:
            self.neurons[j].v_membrane -= self.wta_inhibit

        return winners

    # ----------------------------------------------------------------
    # Escritura (STDP base + esquema V/2 o V/3)
    # Nota: R-STDP con recompensa vendrá en Fase 9.
    # ----------------------------------------------------------------
    def write(self,
              pre_spikes: np.ndarray,
              post_spikes: np.ndarray,
              dt: float = 1e-3,
              A_plus: float = 0.01,
              A_minus: float = 0.01):
        pre = np.asarray(pre_spikes, dtype=bool)
        post = np.asarray(post_spikes, dtype=bool)

        # Voltaje de escritura según esquema
        V_write = 2.0
        V_half = V_write / 2.0 if self.scheme == "V2" else V_write / 3.0

        # Todas las celdas half-selected reciben V_half (disturb)
        V_cells = np.full((self.N, self.M), V_half)

        pre_idx = np.where(pre)[0]
        post_idx = np.where(post)[0]

        # Celdas fully-selected: LTP si pre∩post, LTD si solo pre
        dX = np.zeros((self.N, self.M))
        for i in pre_idx:
            for j in post_idx:
                V_cells[i, j] = V_write
                dX[i, j] = +A_plus
            for j in np.setdiff1d(np.arange(self.M), post_idx):
                dX[i, j] = -A_minus

        # Actualización física celda por celda con el memristor del lab
        for i in range(self.N):
            for j in range(self.M):
                self.cells[i][j].update(voltage=float(V_cells[i, j]), dt=dt)

        # Actualización de estado interno X
        self.X = np.clip(self.X + dX, 0.0, 1.0)

    # ----------------------------------------------------------------
    # Reset (LIF y estado memristivo a initial_state)
    # ----------------------------------------------------------------
    def reset(self, keep_weights: bool = True):
        for n in self.neurons:
            n.reset()
        if not keep_weights:
            self.X.fill(self.elec.initial_state)

    # ----------------------------------------------------------------
    # Persistencia (en robot_sim/outputs/, NUNCA en el lab)
    # ----------------------------------------------------------------
    def save_state(self, name: str = "crossbar_state.json"):
        payload = {
            "N": self.N, "M": self.M,
            "scheme": self.scheme,
            "R_wire": self.R_wire, "R_sense": self.R_sense,
            "wta_inhibit": self.wta_inhibit,
            "X": self.X.tolist(),
            "G": self.G.tolist(),
        }
        (OUT_DIR / name).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def load_state(self, name: str = "crossbar_state.json"):
        p = OUT_DIR / name
        if not p.exists():
            raise FileNotFoundError(p)
        d = json.loads(p.read_text(encoding="utf-8"))
        X = np.asarray(d["X"], dtype=float)
        if X.shape != (self.N, self.M):
            raise ValueError(f"Shape X {X.shape} != ({self.N},{self.M})")
        self.X = np.clip(X, 0.0, 1.0)
