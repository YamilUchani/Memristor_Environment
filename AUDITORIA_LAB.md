# Auditoría de neuromorphic_lab

## 1. Árbol del proyecto
```text
neuromorphic_lab/
├── run_app.py
├── requirements.txt
├── configs/
│   ├── last_session.json
│   ├── lif_config.json
│   ├── memristor_hfo2_neuron.json
│   ├── memristor_prezioso.json
│   ├── memristor_volatile.json
│   ├── strukov_biolek.json
│   ├── strukov_ideal.json
│   └── strukov_stochastic.json
├── neurolab/
│   ├── __init__.py
│   ├── core/
│   │   ├── config.py
│   │   └── memristor.py
│   ├── devices/
│   │   ├── config.py
│   │   └── models/
│   ├── neurons/
│   │   ├── base.py
│   │   ├── config.py
│   │   └── lif.py
│   ├── synapses/
│   │   ├── base.py
│   │   ├── configs.py
│   │   ├── memristive_synapse.py
│   │   ├── plasticity.py
│   │   └── stdp.py
│   ├── crossbar/
│   │   ├── addressing.py
│   │   ├── configs.py
│   │   ├── core.py
│   │   ├── gui_integration.py
│   │   ├── line_resistance.py
│   │   ├── nodal_solver.py
│   │   ├── plasticity.py
│   │   ├── programming.py
│   │   ├── sneak_paths.py
│   │   └── validation.py
│   └── gui/
└── validaciones/
    ├── 01_strukov_2008.py
    ├── ...
    └── 24_jo2010_white.py
```

## 2. Punto de entrada
- **Existe `run_app.py`:** Sí. Ruta exacta: `neuromorphic_lab/run_app.py`.
- **Qué importa y qué hace al arrancar:** 
  Importa `sys` y `os`. Inserta la ruta del directorio actual al inicio del `sys.path`. Importa `main` desde `neurolab.gui.app`. Ejecuta `main()` si se llama como script principal.
- **¿Genera/lee `configs/last_session.json`?** 
  Sí, la inicialización de la GUI lee los perfiles guardados, y al evento de cerrado (closeEvent en `main_window.py`) llama a `_save_last_session()` para actualizar `last_session.json`.

## 3. Dataclasses de configuración
### Archivo: `neurolab/core/config.py`
- **ElectricalConfig**
  - **Campos**: `r_on: float = 100.0`, `r_off: float = 16000.0`, `initial_state: float = 0.1`
  - **Métodos**: `__post_init__` (asegura `r_on < r_off` y `0.0 <= initial_state <= 1.0`)
- **StrukovConfig**
  - **Campos**: `D: float = 10e-9`, `mu_v: float = 1e-14`
- **PreziosoConfig**
  - **Campos**: `A_p, A_m, t_p, t_m, a_p, a_m, b_p, b_m` (todos `float` con valores por defecto)
- **PreziosoVirginConfig**
  - **Campos**: Parámetros vírgenes análogos a `PreziosoConfig`

### Archivo: `neurolab/devices/config.py`
- **DeviceConfig**
  - **Campos**: `name: str = "Strukov 2008 - Figure 2b"`, `family: str = "oxide_memristor"`, `model: str = "strukov"`, `metadata: Dict[str, Any]`
- **StateConfig**
  - **Campos**: `x_init: float = 0.1`, `x_min: float = 0.0`, `x_max: float = 1.0`, `normalized: bool = True`
  - **Métodos**: `__post_init__` (valida bounds de `x_init`)

### Archivo: `neurolab/neurons/config.py`
- **LIFConfig**
  - **Campos**: `c_m: float = 100e-9`, `r_leak: float = 1e6`, `r_series: float = 100e3`, `v_rest: float = 0.0`, `v_th: float = 1.0`, `v_th_base: float = 1.0`, `v_adapt_inc: float = 0.0`, `tau_adapt: float = 0.05`, `v_reset: float = 0.0`, `t_ref: float = 0.0`
  - **Métodos**: `__post_init__` (Validaciones de positividad de resistencias y capacidades), propiedades `r_eq`, `tau`, `has_afterhyperpolarization`.

## 4. Modelos físicos de dispositivos (`neurolab/devices/`)
- **Nombres de Clases**: `StrukovMathModel`, `PreziosoMathModel`
- **Firma literal del método de actualización**: Para el envoltorio `Memristor` se usa `update(self, V_applied: float, dt: float) -> None` (determinado de uso general, aunque el MathModel posee un `dxdt(self, state, v_t)`).
- **Unidades y convenciones**: `V_applied` asume Voltios (V). `dt` asume segundos (s). `D` en metros, `mu_v` en $m^2 V^{-1} s^{-1}$.
- **Dependencia de config**: Reciben instancias de `StrukovConfig` o `PreziosoConfig`, más `ElectricalConfig`.

## 5. Neuronas (`neurolab/neurons/`)
- **¿Existe LIF?**: Sí. Nombre de la clase: `LIFNeuron`. Archivo: `neuromorphic_lab/neurolab/neurons/lif.py`.
- **Firma literal**: `def step(self, current_input: float = 0.0, dt: float = 1e-4, voltage_input: float = None, t: float = None) -> bool:`
- **Campos esperados**: Instancia de `LIFConfig` provista en su `__init__(self, config: LIFConfig = None)`.
- **Estado interno y reset**: Posee `self.v_membrane`, `self.v_th`, `self.refractory_time_left`, `self.t`, `self.spike_times`, `self.has_spiked`. Se resetean invocando su método `def reset(self) -> None:`.

## 6. Sinapsis / STDP (`neurolab/synapses/`)
- **Clases y funciones**: `MemristiveSynapse`, `STDPRule`, `AntiSTDPRule`.
- **Firma literal de STDP**: En `neurolab/synapses/stdp.py`, `def apply(self, synapse: Synapse, dt: float) -> float:`
- **Parámetros STDP**: Sí usa A+ y A-. Sus campos exactos en el constructor son: `A_plus`, `A_minus`, `tau_plus`, `tau_minus`, `max_percent_change`, `w_min`, `w_max`, `w_0`.
- **¿Integra recompensa?**: NO ENCONTRADO. Solo es STDP puro guiado por $\Delta t$.

## 7. Crossbar (`neurolab/crossbar/`)
- **Clases públicas**: `Crossbar` (en `core.py`).
- **Solución de Kirchhoff**: Utiliza MNA vía la función `solve_crossbar_nodal` (en `nodal_solver.py`), que emplea `np.linalg.solve(A, b)`. Si hay error, recurre a `np.linalg.lstsq`.
- **Firma literal de lectura/MNA**: `def solve_crossbar_nodal(G: np.ndarray, V_rows: np.ndarray, R_wire: float = 1e-3, R_sense: float = 1e-12, R_driver: float = 1e-12) -> np.ndarray:`
- **Sneak paths / IR drops**: Se modelan explícitamente llenando una gran matriz $A$ de $2NM \times 2NM$ nodos y resolviéndola con `R_wire` entre las celdas adyacentes.
- **WTA**: NO ENCONTRADO en el lab (se asume que es lógica de la aplicación exterior o post-procesamiento).
- **Esquemas V/2 y V/3**: NO ENCONTRADO como lógica o clase empaquetada dentro de los archivos principales auditados del crossbar.

## 8. Contenido de configs/*.json
### `configs/last_session.json`
- **Ruta**: `neuromorphic_lab/configs/last_session.json`
- **Claves superiores**: `_meta`, `device_name`, `material`, `device_family`, `model_name`, `r_on`, `r_off`, `initial_state`, `D_nm`, `mu_v`, `seed`, `enable_volatile`, `signal`, etc.
- **Claves anidadas (`signal`)**: `realtime_enabled`, `waveform`, `v0`, `f0`, `duration`, `dt_ms`.
- **Ejemplo**:
  - `"device_name": "Strukov TiO2 (Paper Fig 2b)"`
  - `"model_name": "strukov"`
  - `"r_on": 100.0`

### `configs/lif_config.json`
- **Ruta**: `neuromorphic_lab/configs/lif_config.json`
- **Claves superiores**: `_meta`, `c_m_value`, `c_m_unit`, `r_leak_value`, `r_leak_unit`, `r_series_value`, `r_series_unit`, `v_rest`, `v_th`, `v_reset`, `t_ref`, `signal_panel`.
- **Ejemplo**:
  - `"c_m_value": 100.0`
  - `"r_leak_value": 1.0`
  - `"v_th": 1.0`

## 9. API pública (superficie exportada)
En general, el diseño del lab depende de las importaciones profundas absolutas:
- **`neurolab.io.profile_manager`**: Exporta `ProfileManager`.
- **`neurolab.crossbar.nodal_solver`**: Exporta `solve_crossbar_nodal`.
- **`neurolab.neurons.lif`**: Exporta `LIFNeuron`.
- **`neurolab.synapses.stdp`**: Exporta `STDPRule`.
- No hay re-exportaciones generalizadas en módulos `__init__.py` superiores que concentren la API.

## 10. Convenciones globales
- **Unidades de tiempo**: El núcleo físico espera `dt` en segundos (s). Las configs JSON de señales usan `dt_ms` (milisegundos) para la GUI.
- **Unidades de voltaje**: Voltios (V).
- **Estado primario del memristor**: Parámetro `x` normalizado (0 a 1) mapeado a conductancia (G) y resistencia (R). Para MNA, usa Siemens (G).
- **Inicialización de G**: Opcional. No se imponen restricciones fuertes a nivel núcleo, las GUIs asumen `initial_state`.
- **Guardado de estado**: Uso manual de `ProfileManager.save()` o `save_last_session()`.

## 11. Scripts de validación (`validaciones/`)
Hay 24 scripts que replican "papers" o validan subcomponentes físicos:
- `24_jo2010_white.py`: Importa `MemristiveSynapse`, `STDPRule`, `LTPRule`. Genera `fig_15b_stdp_jo2010.png` y `fig_14b_ltp_ltd_jo2010.png`. Demuestra ventanas STDP empíricas.
- `01_strukov_2008.py`: Valida histeresis en T de Strukov.
- `03_lif_analitica.py`: Valida respuesta escalón de LIFNeuron comparada con Euler.
- `04_convergencia.py`: Comprueba estabilidad de paso de integración.

## 12. Dependencias externas
- **`requirements.txt`**:
  - `numpy>=1.20.0`
  - `matplotlib>=3.3.0`
  - `pandas>=1.2.0`
  - `PySide6>=6.4.0`

## 13. Gaps y ambigüedades detectadas
- No hay un "Cerebro" (Brain) unificado ni capa envoltorio para la arquitectura de red; MNA y las LIF son componentes aislados en el lab.
- No existe matriz de elegibilidad $E$ ni mecanismos "Reward-modulated" nativos en las sinapsis (requiere capa adicional).
- Winner-Takes-All (WTA) no existe. El lab es netamente físico (un crossbar o un dispositivo).
- Regímenes de escritura de memoria (V/2 o V/3) no están empaquetados como funciones, deben ser simulados enviando `V_rows` manuales al solver.
- `last_session.json` mezcla configuración del dispositivo con configuración de la "señal de interfaz de usuario".

## 14. Contrato mínimo para robot_sim
Para consumir este lab, `lab_bridge.py` en `robot_sim` DEBE:
- **a) Cargar config de dispositivo**: Importar `ProfileManager` de `neurolab.io.profile_manager`, usar `load()` para procesar JSONs o usar lectura directa de dicts sobre `configs/last_session.json`.
- **b) Cargar config de neurona**: Importar `LIFConfig` de `neurolab.neurons.config` y llenarla de `configs/lif_config.json`.
- **c) Leer matriz G**: NO ENCONTRADO como soporte estandarizado en el lab. Se debe implementar `np.load` local o JSON persistente propio.
- **d) Resolver MNA**: Importar e invocar literalmente `solve_crossbar_nodal(G, V_rows, R_wire, ...)` de `neurolab.crossbar.nodal_solver`.
- **e) Aplicar STDP**: Reimplementar o extender manualmente para R-STDP, ya que `neurolab.synapses.stdp` no soporta recompensa o elegibilidad.
- **f) Guardar estado**: Ejecutar su propio logueo en un `outputs/` separado, ya que el lab no provee logging para matrices $G$ completas o episodios de RL.
- **Rutas literales de JSON**: `neuromorphic_lab/configs/last_session.json` y `neuromorphic_lab/configs/lif_config.json`.

---
**VEREDICTO**: El laboratorio base es altamente maduro y preciso a nivel puramente físico (MNA de Kirchooff, LIF física, dinámica memristiva), pero NO contiene andamiaje de Machine Learning o RL. Carece de WTA, trazas de elegibilidad, y abstracciones de red $N \times M$ conjuntas. **Está listo para ser consumido**, bajo la estricta condición de que `robot_sim/` implemente una capa superior (`CrossbarBrain` / `NeuromorphicAgent`) que ate el MNA y las `LIFNeuron` juntas y sobreescriba su propio lazo de pesos y recompensas.
