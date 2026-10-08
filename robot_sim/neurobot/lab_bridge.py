"""
Único punto de contacto entre robot_sim/ y neuromorphic_lab/.
Ningún otro archivo de neurobot/ debe importar de neurolab.
"""
import sys
import json
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

# --- path al lab (robot_sim/ y neuromorphic_lab/ son hermanos) ---
LAB_ROOT = Path(__file__).resolve().parents[2] / "neuromorphic_lab"
if not LAB_ROOT.exists():
    raise RuntimeError(f"No encuentro neuromorphic_lab en {LAB_ROOT}")
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

# --- imports del lab (verificados por sonda) ---
from neurolab.core.config import ElectricalConfig, StrukovConfig, PreziosoConfig
from neurolab.core.memristor import Memristor
from neurolab.devices.models.strukov import StrukovMathModel
from neurolab.devices.models.prezioso import PreziosoMathModel
from neurolab.neurons.config import LIFConfig
from neurolab.neurons.lif import LIFNeuron
from neurolab.crossbar.nodal_solver import solve_crossbar_nodal

CONFIG_DIR = LAB_ROOT / "configs"


# ------------------------------------------------------------------
# JSON helpers
# ------------------------------------------------------------------
def _read_json(name: str) -> dict:
    p = CONFIG_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Falta {p}")
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------
# Conversión de unidades (robusta a 'Ω', 'Ohm', mojibake UTF-8/latin-1)
# ------------------------------------------------------------------
_UNIT_SCALE = {
    # capacitancia
    "f": 1.0, "mf": 1e-3, "uf": 1e-6, "µf": 1e-6, "nf": 1e-9, "pf": 1e-12,
    # resistencia (contexto memristor: 'M' = mega, no mili)
    "ohm": 1.0, "ω": 1.0,
    "kohm": 1e3, "kω": 1e3,
    "mohm": 1e6, "mω": 1e6,
    # longitud
    "m": 1.0, "nm": 1e-9, "um": 1e-6, "µm": 1e-6,
}


def _normalize_unit(unit: str) -> str:
    """Normaliza unicode, repara mojibake de Ω, pasa a minúsculas."""
    if not unit:
        return ""
    u = unit.strip()
    # 'Î©' es el resultado de leer bytes UTF-8 de Ω como latin-1
    u = u.replace("Î©", "Ω").replace("Î\u00a9", "Ω")
    u = unicodedata.normalize("NFKC", u)
    return u.lower()


def _scaled(value: float, unit: str) -> float:
    """(valor, unidad) → SI. Unidad desconocida → asume SI."""
    key = _normalize_unit(unit)
    if key in _UNIT_SCALE:
        return float(value) * _UNIT_SCALE[key]
    return float(value)


# ------------------------------------------------------------------
# API pública
# ------------------------------------------------------------------
def get_electrical_config(filename: str = "last_session.json") -> ElectricalConfig:
    """Parámetros eléctricos comunes desde un archivo JSON (por defecto last_session.json)."""
    sess = _read_json(filename)
    r_on = _scaled(sess.get("r_on", 100.0), sess.get("r_on_unit", "Ohm"))
    r_off = _scaled(sess.get("r_off", 16000.0), sess.get("r_off_unit", "kOhm"))
    return ElectricalConfig(
        r_on=float(r_on),
        r_off=float(r_off),
        initial_state=float(sess.get("initial_state", 0.1)),
    )


def get_device_config(kind: str = "strukov", filename: str = "last_session.json") -> Any:
    """
    Devuelve la dataclass específica del modelo físico.
    kind: 'strukov' | 'prezioso'
    """
    sess = _read_json(filename)
    if kind == "strukov":
        return StrukovConfig(
            D=_scaled(sess.get("D_nm", 10.0), "nm"),
            mu_v=float(sess.get("mu_v", 1e-14)),
        )
    if kind == "prezioso":
        return PreziosoConfig(
            A_p=float(sess.get("A_p", 0.001)),
            A_m=float(sess.get("A_m", 0.001)),
            t_p=float(sess.get("t_p", 1e-3)),
            t_m=float(sess.get("t_m", 1e-3)),
            a_p=float(sess.get("a_p", 1.0)),
            a_m=float(sess.get("a_m", 1.0)),
            b_p=float(sess.get("b_p", 1.0)),
            b_m=float(sess.get("b_m", 1.0)),
        )
    raise ValueError(f"kind desconocido: {kind}")


def make_memristor(filename: str = "last_session.json") -> Memristor:
    """
    Memristor configurado desde el lab (usando el archivo JSON especificado).
    """
    sess = _read_json(filename)
    electrical = get_electrical_config(filename)
    
    # Extraer el modelo real que se usó en el JSON
    model_str = sess.get("model_name", "strukov").lower()
    
    if "custom_equation" in model_str:
        from neurolab.devices.models.custom import CustomMathModel
        from neurolab.core.config import CustomConfig
        math_model = CustomMathModel()
        cfg = CustomConfig(
            equation_dxdt=sess.get("eq_dxdt", "0.0"),
            equation_i=sess.get("eq_i", "v / (R_on * x + R_off * (1 - x))"),
            equation_window=sess.get("eq_window", "1.0"),
            D=_scaled(sess.get("D_nm", 10.0), "nm"),
            mu_v=float(sess.get("mu_v", 1e-14)),
            tau_relax=float(sess.get("tau_relax", 50.0)),
            x_eq=float(sess.get("x_eq", 0.05))
        )
    elif "prezioso" in model_str or "yakopcic" in model_str:
        from neurolab.devices.models.prezioso import PreziosoMathModel
        math_model = PreziosoMathModel()
        cfg = PreziosoConfig(
            A_p=float(sess.get("A_p", 0.001)), A_m=float(sess.get("A_m", 0.001)),
            t_p=float(sess.get("t_p", 1e-3)), t_m=float(sess.get("t_m", 1e-3)),
            a_p=float(sess.get("a_p", 1.0)), a_m=float(sess.get("a_m", 1.0)),
            b_p=float(sess.get("b_p", 1.0)), b_m=float(sess.get("b_m", 1.0)),
        )
    else:
        # Por defecto Strukov / Ideal / Volátil
        from neurolab.devices.models.strukov import StrukovMathModel
        math_model = StrukovMathModel()
        cfg = StrukovConfig(
            D=_scaled(sess.get("D_nm", 10.0), "nm"),
            mu_v=float(sess.get("mu_v", 1e-14))
        )
    
    modifiers = []
    
    # Modificador 1: Ventana de Frontera
    win_type = sess.get("window_type", "Sin Ventana")
    if win_type == "Biolek":
        from neurolab.devices.realism.window import BiolekWindowModifier
        modifiers.append(BiolekWindowModifier(p=int(sess.get("biolek_p", 1))))
    elif win_type == "Joglekar":
        from neurolab.devices.realism.window import JoglekarWindowModifier
        modifiers.append(JoglekarWindowModifier(p=int(sess.get("biolek_p", 1))))

    # Modificador 2: Ruido Térmico
    if sess.get("enable_noise", False):
        from neurolab.devices.realism.d2d import ThermalNoiseModifier
        modifiers.append(ThermalNoiseModifier(noise_std=float(sess.get("noise_std", 0.05))))

    # Modificador 3: Modo Volátil (Memristor Difusivo)
    if sess.get("enable_volatile", False) or "volatil" in model_str:
        from neurolab.devices.realism.volatile import VolatileDecayModifier
        tau = float(sess.get("tau_relax", 50.0))
        x_eq = float(sess.get("x_eq", 0.05))
        modifiers.append(VolatileDecayModifier(tau_relax=tau, x0_override=x_eq))
        
    return Memristor(math_model=math_model, electrical=electrical, model_config=cfg, modifiers=modifiers)


def get_neuron_config(filename: str = "lif_config.json") -> LIFConfig:
    """
    LIFConfig desde el JSON especificado (por defecto lif_config.json).
    El JSON trae (valor, unidad) separados; aquí se pasa todo a SI.
    """
    d = _read_json(filename)
    return LIFConfig(
        c_m=_scaled(d.get("c_m_value", 100.0), d.get("c_m_unit", "nF")),
        r_leak=_scaled(d.get("r_leak_value", 1.0), d.get("r_leak_unit", "MOhm")),
        r_series=_scaled(d.get("r_series_value", 100.0), d.get("r_series_unit", "kOhm")),
        v_rest=float(d.get("v_rest", 0.0)),
        v_th=float(d.get("v_th", 1.0)),
        v_reset=float(d.get("v_reset", 0.0)),
        t_ref=float(d.get("t_ref", 0.0)) * 1e-3,
    )


def make_lif(filename: str = "lif_config.json") -> LIFNeuron:
    """Fábrica de LIFNeuron ya configurada desde el JSON del lab."""
    return LIFNeuron(config=get_neuron_config(filename))


def solve_mna(G, V_rows, R_wire=1e-3, R_sense=1e-12, R_driver=1e-12) -> np.ndarray:
    """
    Wrapper del solver MNA del lab. Devuelve I_col.
    Sneak paths e IR drops emergen del sistema completo.
    """
    return solve_crossbar_nodal(
        np.asarray(G, dtype=float),
        np.asarray(V_rows, dtype=float),
        R_wire=R_wire,
        R_sense=R_sense,
        R_driver=R_driver,
    )


def get_last_session() -> dict:
    """Acceso crudo al JSON de sesión (inspección / debug)."""
    return _read_json("last_session.json")
