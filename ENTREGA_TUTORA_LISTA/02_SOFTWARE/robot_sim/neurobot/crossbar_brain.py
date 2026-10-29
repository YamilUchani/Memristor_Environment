"""
Crossbar N×M como cerebro del agente.
Ata: solver MNA del lab + LIFNeuron del lab + estado memristivo propio.
Añade lo que el lab NO tiene: WTA, esquemas V/2 y V/3, persistencia de G.
"""
import json
import warnings
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from neurobot import lab_bridge

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)


class CrossbarBrain:
    """
    Estado primario: matriz `X` en [0,1] (fracción no dopada).
    La conductancia se deriva: G(X) = 1 / (r_on + X·(r_off - r_on)).
    """

    def __init__(self,
                 N: int,
                 M: int,
                 R_wire: float = 1e-3,
                 R_sense: float = 1e2,
                 wta_inhibit: float = 5e-3,
                 scheme: str = "V2",
                 seed: Optional[int] = None,
                 json_memristor_nv: str = "last_session.json",
                 json_memristor_v: str = "memristor_volatile.json",
                 json_lif: str = "lif_config.json",
                 json_memristor: Optional[str] = None,
                 initial_X: Optional[float] = None,
                 d2d_enabled: bool = False,
                 c2c_enabled: bool = False,
                 d2d_sigma: float = 0.08,
                 c2c_sigma: float = 0.05,
                 c2c_theta: float = 1.0,
                 c2c_dt: float = 0.02,
                 lif_current_gain: float = 1.0,
                 record_history: bool = True,
                 compute_sneak_ratio: Optional[bool] = None):

        if scheme not in ("V2", "V3"):
            raise ValueError("scheme debe ser 'V2' o 'V3'")

        self.N, self.M = int(N), int(M)
        self.R_wire = float(R_wire)
        self.R_sense = float(R_sense)
        self.wta_inhibit = float(wta_inhibit)
        self.scheme = scheme
        self.seed = None if seed is None else int(seed)

        # Compat: el nombre legacy json_memristor equivale a json_memristor_nv
        if json_memristor is not None:
            json_memristor_nv = json_memristor
        self.json_memristor_nv = json_memristor_nv
        self.json_memristor_v = json_memristor_v
        self.d2d_enabled = bool(d2d_enabled)
        self.c2c_enabled = bool(c2c_enabled)
        self.d2d_sigma = float(d2d_sigma)
        self.c2c_sigma = float(c2c_sigma)
        self.c2c_theta = float(c2c_theta)
        self.c2c_dt = float(c2c_dt)
        self.lif_current_gain = float(lif_current_gain)
        if not np.isfinite(self.lif_current_gain) or self.lif_current_gain <= 0.0:
            raise ValueError("lif_current_gain must be finite and positive")
        # The device state and crossbar equations only change after writes, so
        # reuse the exact linear-network response until the next state update.
        # Per-timestep history is useful for circuit analysis, but copying
        # arrays into a bounded deque on every neural tick is unnecessary in
        # a training run. Keep it enabled for direct CrossbarBrain users.
        self.record_history = bool(record_history)
        self.compute_sneak_ratio = bool(
            self.record_history if compute_sneak_ratio is None
            else compute_sneak_ratio
        )
        if (
            min(self.d2d_sigma, self.c2c_sigma) < 0.0
            or self.c2c_theta <= 0.0
            or self.c2c_dt <= 0.0
        ):
            raise ValueError(
                "Variability sigmas must be nonnegative; c2c_theta and c2c_dt must be positive"
            )

        # Config del lab usando los JSON elegidos
        self.elec = lab_bridge.get_electrical_config(filename=json_memristor_nv)
        # The JSON state is x; an explicit agent X0 can select a programmed regime.
        self._initial_X = (
            1.0 - self.elec.initial_state
            if initial_X is None
            else float(initial_X)
        )
        if not 0.0 <= self._initial_X <= 1.0:
            raise ValueError(f"initial_X must be in [0, 1], got {self._initial_X}")

        # Leemos el kind del json o asumimos strukov por defecto (simplificado para RoboSim)
        kind = "prezioso" if "prezioso" in json_memristor_nv.lower() else "strukov"
        self.dev_cfg = lab_bridge.get_device_config(kind=kind, filename=json_memristor_nv)

        # Estado: X en [0,1], G derivada
        rng = np.random.default_rng(seed)
        self.X = np.clip(
            self._initial_X + rng.normal(0.0, 0.01, (self.N, self.M)),
            0.0, 1.0,
        )

        seed_sequence = np.random.SeedSequence(seed)
        cell_seeds = [
            int(child.generate_state(1, dtype=np.uint32)[0])
            for child in seed_sequence.spawn(self.N * self.M)
        ]
        self._d2d_modifiers = np.empty((self.N, self.M), dtype=object)
        self._c2c_modifiers = np.empty((self.N, self.M), dtype=object)
        self._cell_electrical = np.empty((self.N, self.M), dtype=object)
        self._mna_factorization = None
        self._factorized_G = None
        self._mna_response_cache = None
        for index, cell_seed in enumerate(cell_seeds):
            i, j = divmod(index, self.M)
            d2d, c2c = lab_bridge.make_variability_modifiers(
                seed=cell_seed,
                d2d_sigma=self.d2d_sigma,
                c2c_sigma=self.c2c_sigma,
                c2c_theta=self.c2c_theta,
                c2c_dt=self.c2c_dt,
            )
            self._d2d_modifiers[i, j] = d2d if self.d2d_enabled else None
            self._c2c_modifiers[i, j] = c2c if self.c2c_enabled else None
            self._cell_electrical[i, j] = (
                d2d.apply_to_electrical(self.elec)
                if self.d2d_enabled else self.elec
            )
        self._cell_r_on = np.asarray(
            [[self._cell_electrical[i, j].r_on for j in range(self.M)]
             for i in range(self.N)], dtype=float
        )
        self._cell_r_off = np.asarray(
            [[self._cell_electrical[i, j].r_off for j in range(self.M)]
             for i in range(self.N)], dtype=float
        )

        # Un memristor por celda, con el modelo del lab (para update físico)
        self.cells = [[lab_bridge.make_memristor(filename=json_memristor_nv)
                       for _ in range(self.M)]
                      for _ in range(self.N)]
        # Una LIF por columna
        self.neurons = [lab_bridge.make_lif(filename=json_lif) for _ in range(self.M)]

        # Etapa de salida en modo corriente: una celda volátil por LIF. Su estado
        # modula la corriente transferida; la tensión de programación se limita
        # al voltaje medido en su propia columna de salida.
        self.volatile_devices = [
            lab_bridge.make_memristor(filename=json_memristor_v)
            for _ in range(self.M)
        ]
        from neurolab.devices.realism.volatile import VolatileDecayModifier
        if any(not any(isinstance(modifier, VolatileDecayModifier)
                       for modifier in device.modifiers)
               for device in self.volatile_devices):
            raise ValueError(
                f"{json_memristor_v!r} no habilita el modelo volátil; "
                "selecciona un perfil con enable_volatile=true"
            )
        self._volatile_reference_resistance = np.asarray(
            [device.resistance for device in self.volatile_devices], dtype=float
        )
        self.last_volatile_voltage = np.zeros(self.M, dtype=float)

        # Historial (maxlen evita memory leak en simulaciones largas)
        _H = 2000
        self.history = {
            "I_col":       deque(maxlen=_H),
            "I_volatile":  deque(maxlen=_H),
            "V_volatile":  deque(maxlen=_H),
            "V_rows":      deque(maxlen=_H),
            "spikes":      deque(maxlen=_H),
            "sneak_ratio": deque(maxlen=_H),
            "G_mean":      deque(maxlen=_H),
        }

    def save_weights(self, filepath: str):
        """Guarda la matriz de estado (pesos) en la ruta completa especificada."""
        np.save(filepath, self.X)
        metadata = {
            "N": self.N,
            "M": self.M,
            "crossbar_seed": self.seed,
            "convention": "X = 1 - x",
            "initial_X": self._initial_X,
            "json_memristor_nv": self.json_memristor_nv,
            "variability": {
                "d2d_enabled": self.d2d_enabled,
                "d2d_sigma": self.d2d_sigma,
                "c2c_enabled": self.c2c_enabled,
                "c2c_sigma": self.c2c_sigma,
                "c2c_theta": self.c2c_theta,
                "c2c_dt": self.c2c_dt,
            },
        }
        metadata_path = Path(filepath).with_name(Path(filepath).name + ".meta.json")
        metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"[ok] Pesos (X) guardados en: {filepath}")

    def load_weights(self, filepath: str):
        """Carga la matriz de estado (pesos) desde un archivo .npy"""
        if Path(filepath).exists():
            data = np.load(filepath)
            metadata_path = Path(filepath).with_name(Path(filepath).name + ".meta.json")
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("convention") != "X = 1 - x":
                    raise ValueError(
                        "Weights use an incompatible or unspecified state "
                        f"convention: {metadata.get('convention')!r}"
                    )
                if (metadata.get("N"), metadata.get("M")) != (self.N, self.M):
                    raise ValueError("Weight metadata dimensions do not match this crossbar")
            else:
                warnings.warn(
                    f"Weights {filepath} have no convention metadata; assuming X = 1 - x.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.set_weights(data)
            print(f"[ok] Pesos (X) cargados desde: {filepath}")
        else:
            raise FileNotFoundError(f"Weights file not found: {filepath}")

    def set_weights(self, weights: np.ndarray) -> None:
        """Set crossbar weights while keeping physical cells and caches in sync."""
        weights = np.asarray(weights, dtype=float)
        if weights.shape != self.X.shape:
            raise ValueError(
                f"Weight shape {weights.shape} != expected {self.X.shape}"
            )
        if not np.isfinite(weights).all():
            raise ValueError("Weights must contain only finite values")
        self.X = np.clip(weights, 0.0, 1.0).copy()
        for i in range(self.N):
            for j in range(self.M):
                self.cells[i][j].x = float(1.0 - self.X[i, j])
        self._invalidate_network_cache()

    # ----------------------------------------------------------------
    # G(X) — conductancia derivada
    # ----------------------------------------------------------------
    @property
    def G(self) -> np.ndarray:
        X = np.clip(self.X, 0.0, 1.0)
        R = self._cell_r_on + X * (self._cell_r_off - self._cell_r_on)
        return 1.0 / R

    @property
    def G_min(self) -> float:
        return 1.0 / self.elec.r_off

    @property
    def G_max(self) -> float:
        return 1.0 / self.elec.r_on

    def _invalidate_network_cache(self):
        """Invalidate conductance and MNA caches after any state update."""
        self._mna_factorization = None
        self._factorized_G = None
        self._mna_response_cache = None

    # ----------------------------------------------------------------
    # Lectura (MNA con sneak paths)
    # ----------------------------------------------------------------
    def read(self, V_rows: np.ndarray):
        V_rows = np.asarray(V_rows, dtype=float)
        G = self.G

        if self._mna_factorization is None:
            self._mna_factorization = lab_bridge.factor_mna(
                G=G,
                R_wire=self.R_wire,
                R_sense=self.R_sense,
            )
            self._factorized_G = G
        if self._mna_response_cache is None:
            response = np.empty((self.M, self.N), dtype=float)
            basis = np.zeros(self.N, dtype=float)
            for row in range(self.N):
                basis.fill(0.0)
                basis[row] = 1.0
                response[:, row] = self._mna_factorization.solve(basis)
            self._mna_response_cache = response
        I_col = self._mna_response_cache @ V_rows

        # Ideal para sneak_ratio: I_ideal_j = Σ_i G_ij · V_row_i
        if self.compute_sneak_ratio:
            I_ideal = G.T @ V_rows
            denom = np.linalg.norm(I_ideal) + 1e-30
            sneak_ratio = float(np.linalg.norm(I_col - I_ideal) / denom)
        else:
            sneak_ratio = 0.0

        if self.record_history:
            self.history["I_col"].append(I_col.copy())
            self.history["V_rows"].append(V_rows.copy())
            self.history["sneak_ratio"].append(sneak_ratio)
            self.history["G_mean"].append(float(G.mean()))

        return I_col, sneak_ratio

    # ----------------------------------------------------------------
    # Paso completo: read + LIF + WTA
    # ----------------------------------------------------------------
    def step(self, V_rows: np.ndarray, dt: float = 1e-4, k_wta: int = 1,
             wta_temperature: float = 0.0, rng=None):
        I_col, sneak = self.read(V_rows)

        # El dispositivo volátil recibe una lectura con cumplimiento de tensión
        # igual al rail de entrada del crossbar. La corriente de salida conserva
        # la transferencia acotada hacia la LIF, pero no se convierte en cientos
        # de voltios multiplicándola por la resistencia de estado OFF del volátil.
        # Available output voltage is the measured current through the column
        # sense resistor, not the maximum input-row voltage.
        voltage_compliance = np.abs(I_col) * self.R_sense
        I_lif = np.empty(self.M, dtype=float)
        V_volatile = np.empty(self.M, dtype=float)
        for j, device in enumerate(self.volatile_devices):
            r_initial = self._volatile_reference_resistance[j]
            # Keep the existing calibrated, state-dependent current transfer:
            # gain=1 at the initial resistance and approaches 2 at Ron.
            current_gain = 2.0 * r_initial / (r_initial + device.resistance)
            current_before_update = float(I_col[j]) * current_gain
            unconstrained_voltage = current_before_update * device.resistance
            voltage_device = float(np.clip(
                unconstrained_voltage,
                -voltage_compliance[j], voltage_compliance[j],
            ))
            device.update(voltage=voltage_device, dt=dt)
            I_lif[j] = current_before_update
            V_volatile[j] = voltage_device

        self.last_volatile_voltage = V_volatile.copy()

        spikes = np.zeros(self.M, dtype=bool)
        for j, n in enumerate(self.neurons):
            # Optional transimpedance/front-end gain between crossbar current
            # readout and the LIF input. The crossbar solve and conductances
            # remain unchanged; gain=1 preserves the original direct coupling.
            spikes[j] = n.step(
                current_input=float(I_lif[j]) * self.lif_current_gain, dt=dt
            )

        winners = self._apply_wta(
            spikes, I_col, k=k_wta,
            temperature=wta_temperature, rng=rng,
        )
        if self.record_history:
            self.history["spikes"].append(spikes.copy())
            self.history["I_volatile"].append(I_lif.copy())
            self.history["V_volatile"].append(V_volatile.copy())

        return spikes, I_col, winners, sneak

    # ----------------------------------------------------------------
    # WTA suave: los perdedores reciben inhibición lateral
    # ----------------------------------------------------------------
    def _apply_wta(self, spikes: np.ndarray, I_col: np.ndarray, k: int,
                   temperature: float = 0.0, rng=None):
        active = np.where(spikes)[0]
        if active.size == 0:
            return np.array([], dtype=int)

        k = max(1, min(int(k), int(active.size)))
        if temperature > 0.0 and active.size > k:
            # Stochastic WTA: only neurons that actually spiked compete, and
            # higher read currents remain more likely to win.
            rng = rng if rng is not None else np.random.default_rng()
            logits = np.asarray(I_col[active], dtype=float)
            logits = (logits - logits.mean()) / (logits.std() + 1e-8)
            logits = logits / max(float(temperature), 1e-8)
            logits -= logits.max()
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum()
            winners = rng.choice(
                active, size=k, replace=False, p=probabilities
            )
        else:
            # Deterministic top-k remains the default for existing callers.
            winners = active[np.argsort(I_col[active])[-k:]]

        losers = np.setdiff1d(np.arange(self.M), winners)

        # Inhibición: bajón de potencial en las LIF perdedoras
        for j in losers:
            self.neurons[j].v_membrane -= self.wta_inhibit

        return winners

    def apply_learning_update(self, delta_X: np.ndarray) -> np.ndarray:
        """Apply a learning-rule state update with enabled per-cell variability."""
        delta_X = np.asarray(delta_X, dtype=float)
        if delta_X.shape != self.X.shape:
            raise ValueError(
                f"delta_X shape {delta_X.shape} != crossbar shape {self.X.shape}"
            )

        effective_delta = delta_X.copy()
        for i in range(self.N):
            for j in range(self.M):
                rate = effective_delta[i, j] / self.c2c_dt
                electrical = self._cell_electrical[i, j]
                common = {
                    "state": 1.0 - self.X[i, j],
                    "voltage": 0.0,
                    "current": 0.0,
                    "electrical": electrical,
                    "model_config": self.dev_cfg,
                }
                d2d = self._d2d_modifiers[i, j]
                if d2d is not None:
                    rate = d2d.modify_dxdt(dxdt=rate, **common)
                c2c = self._c2c_modifiers[i, j]
                if c2c is not None:
                    rate = c2c.modify_dxdt(dxdt=rate, **common)
                effective_delta[i, j] = rate * self.c2c_dt

        self.X = np.clip(self.X + effective_delta, 0.0, 1.0)
        self._invalidate_network_cache()
        return effective_delta

    # ----------------------------------------------------------------
    # Reset (LIF y estado memristivo a initial_state)
    # ----------------------------------------------------------------
    def reset(self, keep_weights: bool = True):
        for n in self.neurons:
            n.reset()
        self.last_volatile_voltage.fill(0.0)
        if not keep_weights:
            self.X.fill(self._initial_X)
            for i in range(self.N):
                for j in range(self.M):
                    self.cells[i][j].x = 1.0 - self._initial_X
            self._invalidate_network_cache()

    # ----------------------------------------------------------------
    # Persistencia (en robot_sim/outputs/, NUNCA en el lab)
    # ----------------------------------------------------------------
    def save_state(self, name: str = "crossbar_state.json"):
        payload = {
            "N": self.N, "M": self.M,
            "scheme": self.scheme,
            "R_wire": self.R_wire, "R_sense": self.R_sense,
            "wta_inhibit": self.wta_inhibit,
            "variability": {
                "d2d_enabled": self.d2d_enabled,
                "d2d_sigma": self.d2d_sigma,
                "c2c_enabled": self.c2c_enabled,
                "c2c_sigma": self.c2c_sigma,
                "c2c_theta": self.c2c_theta,
                "c2c_dt": self.c2c_dt,
            },
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
        for i in range(self.N):
            for j in range(self.M):
                self.cells[i][j].x = float(1.0 - self.X[i, j])
        self._invalidate_network_cache()
