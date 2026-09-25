# INFORME TÉCNICO EXHAUSTIVO — CROSSBAR 4×4

> **Alcance:** revisión de enfermizo detalle de TODO lo relacionado con el Crossbar 4×4 en el repositorio `Memristor_Environment`.
> **Fecha:** 2026-09-25 · **Rama:** `dev` (HEAD `05ca3e0`) · **Windows (win32).**
> **Nota metodológica:** se revisaron 100 % de los archivos del paquete `neurolab.crossbar`, la vista/controlador/canvas 4×4 de la GUI, los elementos visuales, las neuronas LIF usadas, las 20 pruebas de validación 4×4, los tests pytest dedicados, el script generador de figuras, la documentación LaTeX/Markdown y el legacy de los pasos 17–23. Se documentan también los **cambios sin commitear** (working tree) que afectan directamente al 4×4.

---

## 1. Resumen ejecutivo

El Crossbar 4×4 es **la matriz por defecto** de todo el proyecto: `CrossbarConfig.n_rows = n_cols = 4`. Existe en dos planos:

1. **Núcleo numérico** (`neurolab/crossbar/*.py`): clase unificada `Crossbar` con 16 celdas `MemristorStrukov`, lectura `I = G^T·V`, programación selectiva (Row/1T1R/V/2/V/3), no idealidades (sneak paths, IR-drop de línea), variabilidad D2D/C2C y plasticidad STDP/R-STDP. En este plano el 4×4 es simplemente el caso `N=4` (default) de un crossbar N×M.
2. **Capa interactiva GUI** (`neurolab/gui/crossbar_4x4_view.py` + `controller_4x4.py`): un circuito robótico-esquemático con 4 sensores → 16 memristores → 4 memristores volátiles HfO₂ → 4 neuronas LIF → 4 actuadores, con modos **Lectura / Programación V/2 / STDP / R-STDP**, inhibición lateral WTA first-to-fire, animación en tiempo real (timer de 30 ms) y telemetría matricial.

Además hay un **ecosistema de 20 pruebas de validación** (`draw_4x4_01…20`) que generan figuras publicables (`validaciones_4x4/fig_*.png`) y alimentan el catálogo de la GUI (`catalog.py` → `4x4_01…20`), más tests pytest dedicados (incluidos 2 archivos nuevos sin commitear).

**Veredicto resumido:** arquitectura madura pero con (a) un bug de diseño en las pruebas de no idealidades (`draw_4x4_08/09` no activan los flags `enable_sneak_paths`/`enable_line_resistance`, por lo que calculan el caso ideal), (b) duplicidad de implementaciones STDP, (c) una reescritura reciente (sin commitear) de la vista y del R-STDP con elegibilidad diferida. Todo ello se detalla en la sección 8.

---

## 2. Inventario de archivos relacionados con el Crossbar 4×4

### 2.1 Núcleo físico (`neuromorphic_lab/neurolab/crossbar/`)

| Archivo | Bytes | Responsabilidad |
|---|---|---|
| `__init__.py` | 1906 | Exports y alias de retrocompatibilidad `CrossbarIdeal = CrossbarSneak = CrossbarLine = Crossbar`; `__version__ = '2.0.0'` |
| `configs.py` | 3395 | `ProgrammingMode` (ROW/1T1R/V2/V3) y `CrossbarConfig` (dataclass; default **4×4**) |
| `core.py` | 19825 | Clase `Crossbar` unificada N×M: lectura, programación, D2D/C2C, no idealidades |
| `addressing.py` | 7831 | `AddressDecoder`, `RowDriver`, `ColumnDriver`, `ReadConfig`, `WriteConfig` |
| `line_resistance.py` | 3298 | `LineResistanceModel`: caída IR por líneas H/V (punto fijo) |
| `sneak_paths.py` | 2789 | `SneakPathModel`: corrientes parásitas (modelo de vecinos) |
| `programming.py` | 2271 | Helpers puros de matrices de voltaje por esquema + `classify_cells` |
| `plasticity.py` | 7283 | `Trace`, `STDPConfig`, `STDPRule`, `RSTDPConfig`, `RSTDPRule` |
| `gui_integration.py` | 8369 | `CrossbarGUIHelper` (+ `COLORS`, `V_READ_THRESHOLD=0.49 V`) |
| `validation.py` | 8076 | `validate_read_operation`, `validate_programming_selectivity`, `validate_line_resistance`, `validate_scalability` (4×4→128×128), `validate_sneak_paths`, `validate_d2d_c2c_variability`, `validate_all` |

### 2.2 GUI (vista 4×4, controlador y elementos)

| Archivo | Bytes | Rol |
|---|---|---|
| `neurolab/gui/crossbar_4x4_view.py` | 66930 | **Vista principal 4×4** (1429 líneas): `Crossbar4x4View`, `Crossbar4x4Canvas`, `BatchVolatileDialog`, `BatchGridDialog` |
| `neurolab/gui/crossbar/controller_4x4.py` | 12489 | **NUEVO (sin commitear):** `Crossbar4x4Controller`, patrón MVC, motor LIF+STDP/WTA |
| `neurolab/gui/crossbar/crossbar_4x4_view.py` | 255 | Re-export de `Crossbar4x4View`/`Crossbar4x4Canvas` |
| `neurolab/gui/crossbar/__init__.py` | 191 | Solo re-exporta `Crossbar2x2View` (el 4×4 no está en `__all__`) |
| `neurolab/gui/crossbar_view.py` | 22256 | `CrossbarView`: contenedor con sub-tabs 1×1/2×2/**4×4** + panel de validación |
| `neurolab/gui/base_view.py` | 20941 | `BaseCrossbarCanvas`: zoom/pan, primitivas de dibujo, `paintEvent` en capas |
| `neurolab/gui/crossbar_elements/*.py` | — | `VisualElement`, `SensorElement`, `MemristorElement`, `VolatileMemristorElement`, `NeuronElement`, `ActuatorElement`, `matrix_helpers` |
| `neurolab/gui/config_dialogs.py` | 14636 | `ConfigDialog` (edición por elemento) + presets |
| `neurolab/gui/read_write_config_dialog.py` | 10990 | `ReadWriteConfigDialog` (configura explícita lectura/escritura V/2…) |
| `neurolab/gui/main_window.py` | 73259 | Dock «🔷 Crossbar 1×1, 2×2 & 4×4 (Fase 4)» + sincronización Pestaña 1→5 |
| `neurolab/gui/widgets/validation_tests_panel.py` | — | Panel de pruebas; `_4x4_draw_map` = `run_4x4_01…20` |
| `neurolab/gui/tests/catalog.py` | — | Catálogo de pruebas: categoría **«Crossbar 4×4»** (entradas `4x4_01…20`) |

### 2.3 Pruebas y scripts de validación 4×4

| Ruta | Rol |
|---|---|
| `neurolab/gui/tests/tests_4x4/test_0X_*.py` (20) | Funciones `draw_4x4_01…20` (cada una retorna `{'status':'PASS','metrics':…}`) |
| `neurolab/gui/tests/tests_4x4/__init__.py` | Re-exports de las 20 `draw_4x4_XX` |
| `neuromorphic_lab/tests/crossbar/test_crossbar_4x4.py` | Pytest del 4×4 (helpers, 20 draw, elementos GUI, integración de física) |
| `neuromorphic_lab/tests/crossbar/test_crossbar_4x4_stress.py` | **NUEVO (sin commitear):** 1000 ticks, STDP/R-STDP stress, relajación volátil, switching de modos, filas V=0 |
| `neuromorphic_lab/tests/gui/test_crossbar_4x4_gui.py` | **NUEVO (sin commitear):** init GUI, no-inflación en STDP con filas a 0 V, lectura no destructiva |
| `experiments/scripts/generate_4x4_validation_figures.py` | Genera las 20 figuras (`validaciones_4x4/fig_01…fig_20.png`) + CSV/JSON |
| `neuromorphic_lab/validaciones_4x4/fig_01…fig_20.png` | **20 PNG** generados presentes en el repo |

### 2.4 Documentación y legacy

| Ruta | Contenido 4×4 |
|---|---|
| `docs/07_CROSSBAR_PLASTICIDAD_Y_SNEAK_PATHS.md` | Conceptos crossbar, sneak paths, modelo V/2 (`R_sneak = R2+R3+R4`), plasticidad, pasos 16–24 |
| `docs/latex/capitulos/06_fase4_crossbar.tex` | Etapa 4.1; **validación cuantitativa del modelo 4×4 frente a Prezioso 2015** (virgen, Fig S3a/b/c; `V_read=0.1 V`, `G≈0.45 µS`) |
| `docs/markdown/Documento_Maestro_Tesis_Neuromorfica.md` | Tablas: `Isneak 4×4 = 0.23 µA` vs `1.45 µA` (84.1 %), `NM 4×4 = 0.95` vs `0.91` (4.3 %) |
| `legacy/memristor_simulator/models/physical_crossbar.py` | `PhysicalCrossbarArray` N×N de `StrukovMemristor` |
| `legacy/.../paso17_red_crossbar.py` | Red crossbar simple |
| `legacy/.../paso18_crossbar_escalable.py` | Escalable **8×4** (32 memristores), clasificación de patrones |
| `legacy/.../paso19_caracterizacion_crossbar.py` | Caracterización **10×8** virgen (Prezioso S3) |
| `legacy/.../paso22_analisis_sneak_paths.py` | Sneak paths **4×4 → 64×64** |
| `legacy/.../paso23_analisis_noise_margin.py` | Noise margin **4×4 → 64×64** |

---

## 3. Configuración por defecto del Crossbar 4×4

`neurolab/crossbar/configs.py` — `CrossbarConfig` (dataclass). **El default es 4×4:**

| Parámetro | Valor | Unidades |
|---|---|---|
| `n_rows`, `n_cols` | **4, 4** | — |
| `R_on` | **2000.0** (antes 100.0; cambio sin commitear) | Ω (≈ G_max = 500 µS) |
| `R_off` | 16000.0 | Ω (≈ G_min = 62.5 µS) |
| `x0` | 0.10 | adim. |
| `D` | 10e-9 | m |
| `mu_v` | 1e-14 | m²/(V·s) |
| `clip_x` | True | — |
| `line_resistance` | 0.0 | Ω |
| `enable_volatile` | False / `tau_relax` 0.5 s / `x_eq` 0.05 | — |
| `G_mean` / `G_sigma` | 70e-6 / 8e-6 | S |
| `G_min` / `G_max` | 1e-6 / 500e-6 | S |
| `V_read` | 0.20 | V |
| `V_th` | **1.20** (antes 0.50; cambio sin commitear) | V |
| `V_program` / `V_reset` | 2.00 / −2.00 | V |
| `dt_pulse` / `dt_physics` | 1e-3 / 0.02 | s |
| `mode` / `programming_mode` | `ProgrammingMode.ROW` / `'V2'` | — |
| `d2d_enabled` / `enable_d2d` | **True** / True (σ=0.08) | — |
| `enable_c2c` | False | — |
| `enable_sneak_paths` | **False** (🔒 desactivado por defecto) | — |
| `enable_line_resistance` | **False** | — |
| `R_line_H` / `R_line_V` | 0.0 / 0.0 | Ω |
| `R_sneak_factor` | 0.1 | — |
| `seed` | 42 | — |

`__post_init__` valida: `n_rows,n_cols ≥ 1`, `G_mean > 0`, `V_th > 0` y la coherencia `V_read < V_th ≤ |V_program|`.

**`ProgrammingMode` (enum):** `ROW="row"`, `ONE_T_ONE_R="1T1R"`, `V2="V2"`, `V3="V3"`.

**`ReadConfig` / `WriteConfig` (`addressing.py`):**
- `ReadConfig`: `V_read=0.20 V`, `V_th=1.20 V` (cambio sin commitear, antes 0.50), `t_settle=1e-3 s`, `V_col=0.0 V`; `is_safe()` exige `V_read < V_th`.
- `WriteConfig`: `V_program=2.00 V`, `V_th=1.20 V` (cambio sin commitear), `scheme='V2'`, `n_pulses=1`, `t_pulse=1e-3 s`, C2C opcional. `V_row_active()` → `+V_program/2` (V/2), `+2V/3` (V/3), si no `+V_program`; `V_col_active()` → `−V_program/2`, `−V_program/3`, `0`.

---

## 4. Núcleo de simulación — `neurolab/crossbar/core.py`

### 4.1 Inicialización (`Crossbar.__init__`, líneas 26–58)

- Crea `rng = np.random.default_rng(config.seed)`.
- `_d2d_factors` (matriz 4×4 de factores fijos por celda): `f = 1 + (G_sigma/G_mean)·N(0,1)`, **recortado a [0.5, 1.5]**. Con G_mean=70 µS y G_sigma=8 µS, σ del factor ≈ 11.4 %.
- `cells`: `MemristorStrukov(StrukovConfig(RON=R_on, ROFF=R_off, x0, D, mu_v, clip_x))`. Si `enable_d2d`/`d2d_enabled` (default True), aplica `G_0 = G_mean·d2d_factor` vía `_set_cell_conductance`.
- `line_model = LineResistanceModel(R_H=config.R_line_H|line_resistance, R_V=…)`.
- Estado: `programming_history`, `last_operation`, `V_rows`, `V_cols`, `I_out`, `programming_target=None` (**nuevo, sin commitear**).

### 4.2 Propiedades y API de conductancia (líneas 135–181)

- `G_matrix` (S), `R_matrix` (Ω = 1/G), `G_matrix_uS`.
- `set_conductance(i,j,G)` / `get_conductance(i,j)` / `set_uniform_conductance(G)` / `set_matrix_conductance(G_mat)` (valida forma (n_rows,n_cols)).
- `_set_cell_conductance`: si el dispositivo expone `set_conductance` lo usa; si no, despeja `x` de la ley lineal `R(x)=R_on·x+R_off·(1−x)` → `x=(R_off−R)/(R_off−R_on)`, clamped a [0,1]; último recurso `cell.G=G`.
- `memristors` **property** reconstruye un `ndarray` de objetos 4×4 en cada acceso (ineficiente, no cachea).

### 4.3 Lectura (líneas 187–283) — el corazón de la operación matricial

```
I_col = G^T · V      (I_j = Σ_i G[i,j]·V_rows[i])
```

- `read_ideal(V)`: `I_out = G_matrix.T @ V` (exacto, vectorizado O(N·M)).
- `read_with_sneak_paths(V)`: **modelo de vecinos** — `alpha = R_sneak_factor` (0.1), `V_mean = mean(|V|)`, y por cada fila i suma `alpha·G[i,j]·V_mean·n_vecinos` donde `n_vecinos = (i>0) + (i<n_rows-1)` (1 en bordes, 2 en interior). `I_out = I_target + I_sneak`.
- `read_with_line_resistance(V)`: delega en `line_model.solve()` (iteración de punto fijo, máx. 10 iteraciones).
- `read(V)`: **despacho gateado por flags** → `enable_sneak_paths` > `enable_line_resistance` > ideal. **Estos flags están desactivados por defecto** (ver hallazgo H1).
- `read_currents()` = `read(V_applied)` (alias).
- `read_single(V)`: solo válido para 1×1 (lanza `ValueError` si no).
- `read_objective_only` / `reference_currents`: solo corriente ideal.
- `sneak_error_pct(V)`: `‖I_sneak − I_ideal‖/‖I_ideal‖·100` (norma L2). Es el método que sí fuerza el modelo de sneak (usado por `draw_4x4_20`).
- `_effective_voltage`: caída acumulada `V_drop = I_avg·R_line·posición`.

### 4.4 Actualización y programación (líneas 285–426)

- `apply_voltages(V)`: setter de `V_applied` (valida longitud).
- `update_memristors(dt)`: construye `V_matrix = V_rows[:,None] − V_cols[None,:]`; **si `programming_target` está activo, aplica máscara de selectividad**: solo se actualizan celdas con `|V_matrix| ≥ V_th` más la celda objetivo. (Mecanismo nuevo sin commitear; protege contra disturbios half-select.)
- `_apply_voltages(V_matrix, dt)`: llama `cell.update(V_cell, dt)` solo si `|V_cell| > 1e-9`.
- Modos de programación:
  - `program_row(i)`: fila entera a +V (V_col=0). Útil para ROW; en 4×4 satura los 4 memristores de la fila.
  - `program_1T1R(i,j)`: solo la celda (i,j) a +V (aislamiento perfecto).
  - `program_V2(i,j, isolate_half_select=False)`: **La Cruz** — fila a +V/2, columna a −V/2 → la celda objetivo ve V completo y las half-selected ven V/2 (±1 V con V_program=2). Con `isolate_half_select=True` se usa la máscara de `V_th` (1.2 V): half-selected (1.0 V) **no** modifican G → selectividad perfecta.
  - `program_V3(i,j)`: objetivo V, fila 2V/3, columna −V/3.
  - `program(i,j)`: despacho según `cfg.mode`/`cfg.programming_mode`.
- `reset()`: re-semilla `rng` y reconstruye D2D + celdas.

### 4.5 Ejemplo concreto 4×4 (Modo V/2, target M₁₁)

Con `V_program=2.0 V`: fila 1 = +1.0 V, columna 1 = −1.0 V → `V_matrix[0,0]=2.0` (objetivo), `V_matrix[0,j]=1.0` y `V_matrix[i,0]=1.0` (cruz, half-selected), resto 0.0. Con `isolate_half_select=True` y `V_th=1.2`, solo M₁₁ evoluciona. **Este es exactamente el comportamiento que verifica `test_crossbar_4x4_core_physics_integration`** (`V_rows[0]=1.0`, `V_cols[0]=−1.0`).

---

## 5. No idealidades, addressing y plasticidad

### 5.1 `line_resistance.py` — `LineResistanceModel`

- Params: `R_H`, `R_V` (Ω/segmento), `max_iter=10`, `tol=1e-6`.
- `solve(G_matrix, V_rows)`: **iteración de punto fijo** — primero corrientes ideales `I_cols = G^T·V`; en cada iteración computa caída acumulada desde el origen: `drop_H = Σ_k I_cols[j]·R_H·k` y `drop_V = Σ_k I_cols[j]·R_V·k` → `V_eff[i,j]=V[i]−drop_H−drop_V`; recalcula `I_new[j]=Σ_i G[i,j]·V_eff[i,j]`; converge si `allclose`. **No garantiza convergencia** (max_iter=10) y usa `I_cols[j]` (corriente de la columna) tanto para la caída H como para la V — modelo simplificado.
- `compute_error`: error relativo porcentual por columna vs ideal.

### 5.2 `sneak_paths.py` — `SneakPathModel`

- `alpha` (factor de acoplamiento, default 0.1).
- `compute_sneak_current(G, V, target_col)` y `compute_all_sneak_currents`: suman `α·G[i,j]·V_rows[i]` sobre **todas las celdas de columnas ≠ objetivo** (acoplamiento global, no topológico).
- `read_error`: error relativo por columna de `I_total = I_ideal + I_sneak` vs ideal.

> ⚠️ Nota: el `SneakPathModel` no se usa en las pruebas 4×4 de figura; el `core.read_with_sneak_paths` sí (modelo de vecinos). El `CrossbarGUIHelper` tampoco lo referencia.

### 5.3 `addressing.py` — hardware de dirección

- `AddressDecoder(n_lines=4)`: `n_bits=2`; `decode(address)` → **vector one-hot** (valida rango); `address_to_binary()` para el badge del canvas.
- `RowDriver(n_rows)` / `ColumnDriver(n_cols)`: aplican `V_active` a la línea activa y `V_inactive` al resto; `ColumnDriver` está pensado para V/2 (columna con voltaje negativo).
- `ReadConfig` / `WriteConfig` descritos en §3 (umbrales cambiados a 1.2 V sin commitear).

### 5.4 `programming.py` — build de matrices puras

- `build_V_matrix_row`, `build_V_matrix_1T1R`, `build_V_rowscols_V2`, `build_V_matrix_V2`, `build_V_matrix_V3`.
- `classify_cells(n, m, i_target, j_target)` → `ndarray` de roles **`'target' | 'half' | 'idle'`** (cruz inclusiva: fila O columna = 'half'). Usado por `gui_integration._classify_roles` para colorear el esquema (🔴 target / 🟡 half / ⚪ idle).

### 5.5 `plasticity.py` — STDP y R-STDP

- `Trace(n, tau=20 ms)`: traza de elegibilidad exponencial `T ← T·exp(−dt/τ) + spike` (dt clamp [1 µs, 1 s]).
- `STDPConfig`: `A_plus=0.1e-6 S`, `A_minus=0.08e-6 S`, `tau_plus=tau_minus=20e-3 s`, `eta=1.0`, `G_min=1e-6`, `G_max=500e-6`.
- `STDPRule`:
  - `reset(n_rows, n_cols)` crea `trace_pre` (filas) y `trace_post` (columnas).
  - `compute_delta_G(spike_pre, spike_post)`: **eventos binarios por tick** (los niveles sostenidos saturan la traza); `dG = A_plus·trace_pre·spike_post − A_minus·spike_post·trace_pre` (anti-simétrico), umbral `|dG| < 1e-9 → 0`.
  - `apply(...)`: (tras cambio sin commitear) primero `trace_pre.step` + `trace_post.step` y **después** `compute_delta_G` — orden corregido.
- `RSTDPConfig(STDPConfig)`: añade `R=0.0`, `use_reward=True`, **`tau_eligibility=1.0 s`** (nuevo).
- `RSTDPRule` (**rework completo sin commitear**, ver §9): mantiene una **matriz de elegibilidad temporal** `eligibility (4,4)` que decae con `τ_eligibility` y acumula `dG_base` en cada tick; la recompensa `R` (escalar o vector por columna) modula `dG_final = R·eligibility` **solo cuando hay elegibilidad acumulada**; si la recompensa llega antes que los spikes, se guarda en `_pending_reward`; la recompensa se **consume en 1 tick** (`cfg.R = 0`). Garantiza que `R=0` no produce cambios.

### 5.6 `validation.py` — validadores oficiales

- `validate_read_operation`: 100 tensiones aleatorias, `max|I_sim − G^T·V| < 1e-12` → PASS.
- `validate_programming_selectivity(mode)`: programa (1,1) y reporta `dG` por región (target/half/idle); PASS si `ΔG_target > 0.1 µS`.
- `validate_line_resistance`: barrido `R_H ∈ {0,0.1,1,5,10,50} Ω` con el flag **activado/restaurado** correctamente.
- `validate_sneak_paths`: activa `enable_sneak_paths` temporalmente (a diferencia de las pruebas de figura, ver H1).
- `validate_scalability`: tamaños `(4,4),(8,8),(16,16),(32,32),(64,64),(128,128)` con tiempos de lectura/programación.
- `validate_d2d_c2c_variability` y `validate_all` (construye `Crossbar(n_rows=4, n_cols=4)`).

### 5.7 `gui_integration.py` — `CrossbarGUIHelper`

- `COLORS`: target `#ef4444`, half `#eab308`, idle `#1e293b`, read `#06b6d4`, write `#ec4899`.
- `V_READ_THRESHOLD = 0.49 V` (clasifica badge `⚡ ESCRITURA` vs `📖 LECTURA`).
- `get_display_state(V_rows, V_cols, mode, i_target, j_target)` → estructura de celdas {G_uS, V_cell, role, color, badge, text} + corrientes (`currents_uA`).
- Acciones: `action_read` (clip a [0,0.49]), `action_write_row`, `action_write_cell_V2`, `action_write_cell_1T1R`, `action_ltp_pulse` (+2 V), `action_ltd_pulse` (−2 V), `action_reset`.

---

## 6. GUI interactiva 4×4

### 6.1 `Crossbar4x4View` (`crossbar_4x4_view.py`, líneas 539–1429)

**Layout matricial físicamente completo** (docstring, líneas 6–17):

```
S1 (V1) ──[M11]──[M12]──[M13]──[M14]──► Fila 1 (PRE)
S2 (V2) ──[M21]──[M22]──[M23]──[M24]──► Fila 2 (PRE)
S3 (V3) ──[M31]──[M32]──[M33]──[M34]──► Fila 3 (PRE)
S4 (V4) ──[M41]──[M42]──[M43]──[M44]──► Fila 4 (PRE)
Columna j: M1j → M2j → M3j → M4j ──► M_vj ──► LIF_j ──► Act_j
```

**Inicialización (542–636):**
- `crossbar = Crossbar(CrossbarConfig(n_rows=4, n_cols=4, enable_d2d=False))` — el core inicia **sin D2D** (la dispersión se deja a la capa de elementos GUI).
- `controller = Crossbar4x4Controller(self.elements, self.crossbar)`.
- Decoders/drivers: `row_decoder`, `col_decoder` (`AddressDecoder(n_lines=4)`), `row_driver`, `col_driver`.
- `read_cfg = ReadConfig()`, `write_cfg = WriteConfig()` (esquema V/2).
- Estado: `mode ∈ {'read','program_v2'}`, `target_cell=(0,0)`, `V_rows=np.full(4,0.2)`, `V_cols=np.zeros(4)`.
- Plasticidad: `plasticity_mode ∈ {'off','stdp','rstdp'}`; **recrea** `controller.stdp_rule`/`controller.rstdp_rule` (líneas 580/590) duplicando la creación del controlador (ver H4). RNG Poisson `seed=12345`.
- Sensores Poisson: `SENSOR_MAX_RATE_HZ=100`, `SENSOR_MIN_RATE_HZ=5`, `SENSOR_V_REF=1.0`.
- WTA: `WINNER_LOCK_DURATION=0.20 s`, `REFRACTORY_DURATION=0.15 s`.
- Constantes LIF declaradas en la vista: `LIF_C_m=10e-9`, `LIF_R_leak=100e6`, `LIF_V_th_base=1.0`, `LIF_V_adapt_inc=0.5`, `LIF_tau_adapt=0.05`, `LIF_I_scale=1e-4` — **ojo: el controlador usa otros valores** (H4).
- `G_base=69.4e-6`, `G_decay_rate=0.005`.
- Secuencia de filas: `sequence_period_ticks=40` (200 ms).
- `anim_timer` QTimer **30 ms** → `_anim_tick`.

**Elementos construidos (`_build_elements`, 640–685):** 4 `SensorElement` (S1..S4, `V_out=0.8/0.4/0.3/0.2`), 16 `MemristorElement` (M11..M44, x = 300 + j·180, y = 150 + i·160), 4 `VolatileMemristorElement` (M_v1..M_v4, τ_relax=0.3), 4 `NeuronElement` (LIF_1..LIF_4, C_m=100 nF, R_leak=1 MΩ, V_th=1.0, V_adapt_inc=0.15, tau_adapt=0.05), 4 `ActuatorElement` (Act_1..Act_4, acciones `girar_izq/girar_der/avanzar/retroceder`).

**UI (`_init_ui`, 692–930):** botones de modos (Lectura/V/2/STDP/R-STDP/OFF), pulsos ➕LTP/➖LTD (conectados a `_apply_programming_pulse(±write_cfg.V_program)`), recompensa ✅+1/❌−1, spinboxes de sensores, selector de escala de tiempo (`time_scale_factor` ∈ {1,6,30}), zoom/pan, canvas y barra de estado.

**Bucle `_anim_tick` (1242–1312):**
1. `dt = interval_ms/1000 / time_scale_factor` (0.03 s wall por defecto; slow-motion explícito).
2. Genera `spike_pre` por **Poisson**: `rate_hz = SENSOR_MIN + (SENSOR_MAX−SENSOR_MIN)·min(|V_i|/V_ref,1)`; `p_spike = rate·dt`; 0 Hz si `|V_i| < 0.05` (subumbral → cero spikes).
3. Modo `program_v2`: `V_rows[tg_r]=+V_program/2`, `V_cols[tg_c]=−V_program/2`; en read/STDP/R-STDP: `V_rows` = voltajes de S1..S4, `V_cols = read_cfg.V_col`.
4. Llama `controller.step(...)` y pinta telemetría `I=[...] µA`, conteos de spikes, `t_sim`.

**Modos (939–1006):**
- `_set_mode_1_read` → `mode="read"`, `plasticity_mode="off"`, resetea recompensa.
- `_set_mode_2_v2` → `mode="program_v2"` y `_sync_sensors_to_mode()` (fila target = 1 V, resto 0).
- `_set_mode_3_stdp` → `mode="read"`, `plasticity_mode="stdp"`, reset de `stdp_rule`, arranca tiempo real si estaba pausado.
- `_set_mode_4_rstdp` → `mode="read"`, `plasticity_mode="rstdp"`, reset + recompensa 0.

**Acciones de usuario:** `_apply_programming_pulse` (delega en `controller.apply_programming_pulse` repitiendo `write_cfg.n_pulses` veces con `write_cfg.t_pulse` — 1 ms por pulso), `_apply_reward` (solo setea R en el `rstdp_rule`; la actualización real de G ocurre en el siguiente tick), `_on_simulate` (ejecuta 1 `_anim_tick`), `_on_reset`, `_on_reset_matrix` (nuevas semillas), `_on_show_matrix` (heatmap textual), `_on_config_volatiles` (`BatchVolatileDialog`), `_on_config_grid` (`BatchGridDialog`), `_on_open_rw_config` (`ReadWriteConfigDialog` → sincroniza drivers), `_on_config_voltages`, `_on_uniform` (programa toda la matriz a G µS vía `set_G_matrix`).

**Diálogos:**
- `BatchVolatileDialog` (57–150): aplica modelo (HfO₂ volátil / Ag:SiO₂ difusivo / NbOₓ threshold switch), `τ_relax`, R_ON, R_OFF y x₀ a los 4 M_v.
- `BatchGridDialog` (153–276): programación por sub-matriz con presets **1×1, 2×2, 1×3, 3×3, 4×4, custom**; modelos Strukov/Prezioso/Jo 2010; G objetivo (µS) y x₀; recalcula `x` de cada celda desde `R_target=1/G`, `x=(R_target−ROFF)/(RON−ROFF)`.

### 6.2 `Crossbar4x4Controller` (`crossbar/controller_4x4.py`, nuevo)

Patrón MVC para aislar física + LIF + WTA + plasticidad. Recibe `elements` y `crossbar_core`:

- **Config STDP/R-STDP:** A₊=0.1 µS, A₋=0.08 µS, G_min=1 µS, G_max=500 µS, τ=20 ms; R-STDP con `R=0`.
- **WTA first-to-fire:** `winner_j`, lock 200 ms, `REFRACTORY_DURATION=0.15`.
- **Motor LIF real:** `LIFConfig(c_m=100 nF, r_leak=1 MΩ, v_th=1.0 V, v_adapt_inc=0.15, tau_adapt=0.05, t_ref=0.15 s)` → 4 `LIFNeuron`. **`LIF_I_scale=0.02`**: convierte ~70 µA del crossbar a ~1.4 µA efectivos en la LIF (justificación documentada en el código).
- **`apply_programming_pulse(tg_r, tg_c, v_pulse, dt_pulse=0.03)`:** ① sincroniza cada M_ij del core con `elements['Mij'].params['G']`; ② `crossbar.program_V2(..., isolate_half_select=True)`; ③ escribe de vuelta G del core a los elementos.
- **`step(dt, mode, plasticity_mode, is_animating, V_rows, V_cols, spike_pre)`:** el motor central:
  1. `spike_pre` por defecto = `(V_rows > 0.5)`.
  2. En `program_v2`: sincroniza core ← elementos y llama `crossbar.update_memristors(dt)` con `programming_target = (fila activa, columna activa)` (selectividad V/2 + V_th).
  3. Sincroniza elementos ← core (G, R, x) tras escribir.
  4. **Corrientes reales por columna:** `I_cols[j] = Σ_i G[i,j]·(V_rows[i] − V_cols[j])` (respeta V_col, a diferencia del helper `compute_currents`).
  5. **Memristores volátiles:** `dx_v/dt = 100·I_cols[j] − (x_v − x_eq)/τ_relax` (FIX: relaja hacia `x_eq`, no hacia 0), clamp x ∈ [0.001, 1].
  6. **LIF + WTA:** si hay ganador activo, las demás neuronas se ponen a `V_m=0` (inhibición lateral); `I_syn = I_cols[j]·LIF_I_scale`; `neuron.step(...)`; al disparar suma `spike_count`, marca `spike_post` y fija `refractory_time`.
  7. **STDP/R-STDP** (solo en `mode=="read"`): `G_matrix = get_G_matrix(elements)`; `dG = rule.apply(G, spike_pre, spike_post, dt)` con **escalado de saturación**: `sat = (G_max−G_old)/(G_max−G_min)` para LTP, `(G_old−G_min)/(G_max−G_min)` para LTD; actualiza `G`, `R`, `x` del elemento y llama `crossbar.set_conductance(i,j,G_new)`.
  8. **Weight decay:** en plasticidad, las columnas no ganadoras decaen `G ← G + 0.005·(G_base − G)` con `G_base=69.4 µS`.

### 6.3 `Crossbar4x4Canvas` (`crossbar_4x4_view.py`, 279–537)

- `MIN_WIDTH=1100`, `MIN_HEIGHT=1000`.
- `_draw_wires`: filas PRE en azul (`sensor`), columnas POST en verde (`neuron`), LIF→Actuador; **destello neón** de la columna ganadora (glow 9 px + línea 5 px); animación de partículas (offset basado en `anim_time`).
- Etiquetas: `FILA i (V=±x.xx V)`, `COL j (V=…)`, `🏆 COL j (GANADORA)` / `(INHIBIDA)`.
- `_draw_extra`: badges de los **decoders** (`🎯 Decoder ROW/COL [binario] → Addr n`, solo en `program_v2`), **drivers** (`⚡ Driver ROW (WL)/COL (BL)`), título central «CROSSBAR 4×4 — MATRIZ DE 16 MEMRISTORES STRUKOV», **bus de inhibición lateral WTA** en rosa discontinuo, y la **ecuación matricial en vivo**: `I = G^T · V` con `I_j` en µA (extrae `V` del view; en `program_v2` reconstruye `V[tg_r]=V_program/2`).

### 6.4 Elementos visuales (`crossbar_elements/`)

- `VisualElement`: `element_id`, `x`, `y`, `selected`, `hover`, `params`; `hit_test()` abstracto; `update_params()`.
- `MemristorElement` (**Strukov, no volátil**): params con `RON=2000`, `ROFF=16000`, `x0=0.10`, `D=10 nm`, `mu_v=1e-14`, ventana Biolek p=2, semilla por celda, `chk_d2d=True` σ=0.08, `chk_c2c=True` σ=0.05, `chk_noise=False`, estado `x=0.10`, `R=14410 Ω`, `G=69.4 µS`. `on_params_changed` recalcula **factor D2D por semilla**: `sample=clip(N(0,σ_d2d), −2σ, +2σ)`, `d2d_factor=1+sample`, `G = (1/R_nominal)·d2d_factor`.
- `VolatileMemristorElement`: HfO₂ (`RON=10 kΩ`, `ROFF=500 kΩ`, `x0=0.05`, `τ_relax=0.30 s`, `x_eq=0.05`); `R = RON·x + ROFF·(1−x)`, `G=1/R`.
- `NeuronElement`: LIF (C_m=100 nF, R_series=100 kΩ, R_leak=1 MΩ, V_th=2.5 V, τ_ref=2 ms); recalcula `tau_m` y `tau_eq`.
- `SensorElement` y `ActuatorElement`: parámetros de robot (IR, f_max, distancias; acciones `avanzar/retroceder/girar_*`).
- `matrix_helpers.py`: `get_G_matrix(elements,4,4)` (usa `params['G']` o `'G_11'`), `get_V_vector`, `set_G_matrix` (aplica x desde G), `compute_currents = G.T @ V`, `matrix_to_heatmap_string`.

### 6.5 `BaseCrossbarCanvas` (`base_view.py`)

- `widget_to_canvas` (pan/zoom), `set_zoom` (clamp 0.4×–3×), `zoom_in/out`, `reset_view`, pan con mouse.
- `paintEvent`: capas `_draw_grid` → `_draw_wires` → **elementos por `type(elem).__name__`** → `_draw_extra`.
- Primitivas: `_draw_sensor`, `_draw_memristor`, `_draw_volatile_memristor`, `_draw_neuron`, `_draw_actuator` (con estados 🏆GANADOR / 🚫INHIBIDO).

### 6.6 Integración en la ventana principal (`main_window.py`)

- Línea 725–731: `CrossbarView` dentro del dock «🔷 Crossbar 1×1, 2×2 & 4×4 (Fase 4)».
- Línea 886+: `_sync_memristor_to_synapse` propaga modelo/`r_on`/`r_off`/`x0`/volatilidad desde la Pestaña 1 hacia `crossbar_view.elements`.
- `crossbar_view.py` (línea 337): `self.view_4x4 = Crossbar4x4View(self)` como sub-tab «🔷 Matriz Esquema 4×4».
- `validation_tests_panel.py`: `_4x4_draw_map = {'run_4x4_01'…'run_4x4_20'}` y el catálogo `catalog.py` define las 20 entradas con id `4x4_XX`, categoría **«Crossbar 4×4»** y sus parámetros configurables (p. ej. `4x4_02` V_min/V_max, `4x4_03` G_min/G_max, `4x4_05/06` n_pulses, `4x4_09` line_R, `4x4_10` duración, `4x4_11` max_percent, `4x4_14` G_target).

---

## 7. Las 20 pruebas de validación 4×4

Ubicadas en `neurolab/gui/tests/tests_4x4/test_0X_*.py` como funciones `draw_4x4_XX(gui, **kwargs)` que devuelven `{'status':'PASS','metrics':{…}}`. Todas usan `CrossbarConfig(n_rows=4, n_cols=4)` salvo indicación. La figura correspondiente (`validaciones_4x4/fig_XX_*.png`) la genera `experiments/scripts/generate_4x4_validation_figures.py`.

| # | Prueba (archivo) | Qué valida | Escenario clave | Métricas devueltas |
|---|---|---|---|---|
| 01 | `test_01_operacion.py` | **Operación matricial I = G^T·V** | G heterogénea (100–250 µS) fijada con `set_conductance`; V = [0.8,0.4,0.6,0.3] V | MAE, RMSE, err. máx (%), R² |
| 02 | `test_02_barrido_v.py` | **Barrido de voltaje y linealidad** por columna | V₁ ∈ [−0.5, +0.5] V (41 pts) con V₂=0.2, V₃=0.1, V₄=0 | R² por columna; pendiente dI_j/dV₁ ≈ G₁ⱼ |
| 03 | `test_03_barrido_g.py` | **Barrido G₁₁: ideal vs real (IR-drop)** | G₁₁ ∈ [62.5, 500] µS; R_line=50 Ω usando `CrossbarLine` ⚠️ **H1** | caída V₁ (mV), caída I₂ (µA) |
| 04 | `test_04_no_destructiva.py` | **Lectura no destructiva** | G uniforme (200 µS) vs heterogénea; V_pico=50 mV senoidal ×50 muestras | variación G = 0.000 % (ambas) |
| 05 | `test_05_ltp_selectivo.py` | **LTP selectivo** en M₁₁ | x0=0.30 (G≈130 µS); 40 pulsos +1 V **solo sobre `memristors[0,0].update(...)`** (bug corregido: antes `update_memristors` tocaba toda la fila) | ΔG M11, ΔG M12≈0, aislamiento 100 % |
| 06 | `test_06_ltd_selectivo.py` | **LTD selectivo** en M₄₄ | x0=0.70 (G≈400 µS); 40 pulsos −1 V solo M₄₄ | ΔG M44, ΔG M41≈0, aislamiento 100 % |
| 07 | `test_07_ciclo.py` | **Ciclo reversible LTP→LTD→LTP** | x0=0.30; 50+50+50 pulsos ±1 V en M₁₁; concordancia LTP1 vs LTP2 | error repetibilidad (%), G máx/mín, rango |
| 08 | `test_08_sneak.py` | **Sneak paths** ideal vs `CrossbarSneak` | G aleatoria seed 42 (50–500 µS); V=0.5 V ⚠️ **H1** (usa `read_currents()` gateado) | error medio/máx (%) |
| 09 | `test_09_line_resistance.py` | **Atenuación por R_line** | R ∈ [0,200] Ω (21 pts), G uniforme 300 µS ⚠️ **H1** | caída (%) Col 1 y Col 4 |
| 10 | `test_10_lif_dinamico.py` | **4 neuronas LIF dinámicas** | G heterogénea (100–450 µS); señales senoidales 10–30 Hz; LIF calibrado (c_m=1 µF, r_leak=10 kΩ, v_th=1 V, t_ref=2 ms) | spikes por neurona, V_m máx |
| 11 | `test_11_stdp.py` | **Ventana STDP con escalado físico** | Δt ∈ [−50,+50] ms; G₀=200 µS; `max_percent=0.5 %` ⚠️ **H2** (usa `neurolab.synapses.stdp.STDPRule`) | ΔG LTP/LTD máx |
| 12 | `test_12_cross_talk.py` | **Cross-talk / aislamiento** | x0=0.30; 50 pulsos +1.5 V solo M₁₁ (programación selectiva) | cross-talk máx (≈0 µS) |
| 13 | `test_13_escalabilidad.py` | **Escalabilidad 1×1 / 2×2 / 4×4** | 1000 lecturas + 1000 updates por tamaño | t_read (µs), ops/s |
| 14 | `test_14_uniformidad.py` | **Uniformidad** | `set_uniform_conductance(300 µS)` | media, σ, CV (%) |
| 15 | `test_15_matriz_identidad.py` | **Matriz identidad** | G_diag=500, G_off=62 µS; V=[1.0,0.8,0.6,0.4]; verificación analítica `I_j = G_diag·V_j + G_off·Σ_{i≠j}V_i` | I₁..I₄ vs analítica |
| 16 | `test_16_matriz_diagonal.py` | **Matriz diagonal con gradiente** | G_diag=[500,400,300,200] µS, G_off=62; misma V | I₁..I₄ vs analítica |
| 17 | `test_17_patron_X.py` | **Patrón X** (diag + anti-diag) | 400 µS activas, 62.5 µS fondo, V=0.5 V | I₁..I₄ (simetría) |
| 18 | `test_18_patron_T.py` | **Patrón T** (fila 1 + col 2) | 450 µS activas, fondo 62.5, V=0.5 V | I₂ dominante (tallo) |
| 19 | `test_19_patron_4x4.py` | **Clasificación del carácter «H»** | plantilla 480/62.5 µS; V=0.8 V | I por columna (correlación) |
| 20 | `test_20_comparacion.py` | **Comparación consolidada 1×1/2×2/4×4** | capacidades [160,400,1600] µA; IR-drop % **hardcodeado** [0,3.5,9.8] ⚠️ **H5**; sneak calculado dinámicamente con `sneak_error_pct()` | errores sneak 0/10/20 % aprox. |

### 7.1 Notas por prueba

- **04:** como `set_conductance` sobreescribe el estado, la «no destructividad» se comprueba contra G fijada (no contra la física); las variaciones reportadas son 0.000 % por construcción.
- **05/06/07/12:** el docstring documenta explícitamente el bug histórico corregido → `cb.update_memristors(dt)` aplicaba voltaje a TODA la fila (la programación selectiva ahora toca la celda objetivo directamente). La prueba 12 explica que «el crossbar ideal **no** tiene cross-talk»: el aislamiento es un resultado del modelo.
- **08:** el objetivo original era demostrar el efecto sneak con `alpha=0.1`; ver H1 (el efecto no se activa).
- **10:** choca conceptualmente con los parámetros reales del controller GUI (que usa c_m=100 nF, r_leak=1 MΩ, LIF_I_scale=0.02); la prueba calibra su propio LIF para que dispare a ~100 Hz.
- **20:** `ir_drop_pct = [0.0, 3.5, 9.8]` está fijo en el código (no se calcula), mientras que `sneak_error_pct` sí es dinámico (el docstring «v2» corrigió ese bug).

---

## 8. Hallazgos, riesgos y bugs (con rutas y líneas exactas)

### H1 ⚠️ BUG funcional — `draw_4x4_08` y `draw_4x4_09` (y 03) no activan las no idealidades

`CrossbarSneak` y `CrossbarLine` **son alias de `Crossbar`** (`crossbar/__init__.py:49–51`), y `Crossbar.read()` (`core.py:225–234`) despacha según `cfg.enable_sneak_paths` / `cfg.enable_line_resistance`, **ambos `False` por defecto** (`configs.py:76–77`). Las pruebas de figura usan `read_currents()` (ideal), por lo que:

- `draw_4x4_08` (`tests_4x4/test_08_sneak.py:30–31`): `I_sneak = cb_sneak.read_currents()` → **corriente ideal** → `err_rel = 0 %` y la figura 08 no muestra parásitas.
- `draw_4x4_09` (`test_09_line_resistance.py:26`): `CrossbarLine(CrossbarConfig(line_resistance=R))` + `read_currents()` → ideal → **curva plana** (la «mejor» figura solo puede mostrar un achatamiento).
- `draw_4x4_03` (`test_03_barrido_g.py:50–51`): `cb_real.read_currents()` → ideal → la «caída nodal» del panel 2 es idéntica a la ideal (el texto de la figura la explica, pero los datos no la producen).

Esto explica por qué `sneak_error` en la prueba 20 (que sí llama a `sneak_error_pct()`) es el único sitio donde el efecto sneak aparece realmente. **El resto de validadores oficiales sí activan los flags correctamente** (`validation.py:98–100, 136–143`).

**Sugerencia:** en las draw_4x4_03/08/09 activar los flags (`cfg.enable_sneak_paths=True`, `cfg.enable_line_resistance=True`) o llamar a `read_with_sneak_paths`/`read_with_line_resistance`/`sneak_error_pct` directamente.

### H2 ⚠️ Duplicidad de implementaciones STDP

- `draw_4x4_11` importa `from neurolab.synapses.stdp import STDPRule` (rama de sinapsis clásica, con `delta_w(Δt)`, `A_plus/A_minus`, `max_percent_change`).
- El controller 4×4 y el paquete crossbar usan `neurolab.crossbar.plasticity.STDPRule` (con trazas de elegibilidad por fila/columna y matriz dG 4×4).
- También `validation_tests_panel.py` importa `STDPRule` de `neurolab.synapses` (línea 33). Dos modelos y dos semánticas de «ΔG» → resultados no intercambiables.

### H3 ⚠️ «No destructividad» de `draw_4x4_04` es trivial

La prueba programa `set_conductance`/`set_matrix_conductance` (que fijan G) y luego solo lee; el 0.000 % de variación es por construcción y no ejercita la física Strukov (`dx/dt`). No hay ningún pulso sobre-umbral que pueda perturbar el estado.

### H4 ⚠️ Duplicación e inconsistencias en la vista/controller

- `Crossbar4x4View.__init__` **recrea** `controller.stdp_rule` y `controller.rstdp_rule` (`crossbar_4x4_view.py:580–591`) después de que el controller ya los creó (`controller_4x4.py:24–31`). Duplica configuración (misma), pero viola el desacoplamiento MVC.
- `LIF_I_scale` existe **dos veces con valores distintos**: `view.LIF_I_scale=1e-4` (línea 620, sin uso) y `controller.LIF_I_scale=0.02` (usado en `controller_4x4.py:53`). El comentario del controller justifica el 0.02.
- Constantes LIF de la vista (líneas 615–620: C=10 nF, R_leak=100 MΩ, V_adapt=0.5) nunca se pasan al controller (que usa C=100 nF, R_leak=1 MΩ, v_adapt=0.15).

### H5 ℹ️ Valores hardcodeados en `draw_4x4_20`

`max_current_uA=[160,400,1600]` e `ir_drop_pct=[0.0,3.5,9.8]` son literales (solo `sneak_error_pct` es dinámico). Si la física de línea cambiara, la figura 20 quedaría desactualizada.

### H6 ℹ️ Modelos simplificados de no idealidades

- Sneak paths del core: **modelo de vecinos** (`neighbors = (i>0)+(i<n-1)`, `α·G·V_mean`) — no resuelve el circuito nodal de Kirchhoff completo; escala «lineal en vecinos» (~20 % en 4×4, ~30 % en 8×8, etc. según la prueba 20).
- IR-drop: **iteración de punto fijo** `max_iter=10` sin garantía de convergencia; usa la misma `I_cols[j]` para caídas H y V.
- `SneakPathModel` (`sneak_paths.py`) está implementado pero **no se referencia** ni desde el core ni desde la GUI (código muerto de facto).

### H7 ℹ️ Dos sistemas D2D paralelos

- Core: `enable_d2d=True` por defecto con `f = 1 + (G_sigma/G_mean)·N(0,1)` clip [0.5,1.5].
- GUI: la vista crea el core con `enable_d2d=False`, pero cada `MemristorElement` aplica su propio `d2d_factor` por semilla (`memristor_element.py:91–104`, σ=0.08 clip ±2σ).
- Resultado: si se instancia `Crossbar` con config por defecto, la D2D está activa; dentro de la GUI, la dispersión la pone la capa de elementos. Ambas coexisten sin conflicto pero duplican el concepto.

### H8 ℹ️ Rendimiento y estilo

- `memristors` es una **property que reconstruye el array** de objetos en cada acceso → O(4×4) por consulta.
- `gui_integration.get_display_state` (modo `program_V2`) asume que `V_rows[i_target]` ya vale `V/2` para derivar la columna (`gui_integration.py:74–78`); `_draw_extra` de la vista **reasigna la variable local `view`** en la línea 516 después de usarla en 446 (sombreado confuso, aunque funcional).
- Los botones de acción (LTP/LTD/±1) quedan siempre habilitados y los modos STDP/R-STDP se pueden activar desde cualquier estado (los tests lo exigen).

### H9 ✅ Comportamientos correctos que conviene preservar (verificados)

- **Selectividad V/2 real:** con `isolate_half_select=True` y `V_th=1.2 V`, la cruz half-selected (1.0 V) no modifica G → test `core_physics_integration` (V_rows[0]=1.0, V_cols[0]=−1.0).
- **Filas con V=0 no potencian:** test de estrés y `test_crossbar_4x4_gui` verifican que M2*/M4* quedan < 100–150 µS tras 100 ticks de STDP (el bug de «inflación falsa» quedó cubierto).
- **R-STDP consume la recompensa en 1 tick** y usa elegibilidad diferida (`tau_eligibility=1 s`) — 2 tests nuevos lo cubren (`test_plasticity.py::test_rstdp_delayed_reward_uses_eligibility`, `test_crossbar_4x4.py::test_crossbar_4x4_physical_realism_integration`).
- **Lectura no destructiva en GUI:** `_set_mode_1_read` + 100 ticks no cambian G (±1 µS) — `test_crossbar_4x4_read_mode_does_not_inflate_conductance`.
- **Volátiles:** relajan hacia `x_eq` (no a 0) con `dx/dt = 100·I − (x−x_eq)/τ` — test `volatile_relaxation_stress` (x→0.05 ±0.01 tras 100 ticks).

---

## 9. Cambios sin commitear (working tree, rama `dev`)

`git status --short` muestra 16 archivos modificados + 3 nuevos + 8 PNG sueltos en la raíz (`loop_*.png`) que **parecen residuos de pruebas de ruido**:

### 9.1 Modificados que tocan el 4×4

| Archivo | Difstat (+/−) | Contenido del cambio |
|---|---|---|
| `neurolab/crossbar/addressing.py` | +2/−2 | `ReadConfig.V_th` y `WriteConfig.V_th`: **0.50 → 1.20 V** (evitar disturbio half-select) |
| `neurolab/crossbar/configs.py` | +3/−3 | `R_on` **100 → 2000 Ω** (coherente con G_max=500 µS); `V_th` **0.50 → 1.20 V** |
| `neurolab/crossbar/core.py` | +16/−3 | Añade `programming_target=None`; default `RON=2000` en `_init_cells`; `update_memristors` con máscara de `V_th` + `programming_target` (selectividad) |
| `neurolab/crossbar/plasticity.py` | +62/−27 | `STDPRule.apply` corrige el orden (traces → dG); `RSTDPConfig.tau_eligibility=1.0 s`; `RSTDPRule` con matriz `eligibility`, `_pending_reward` y consumo de R en 1 tick |
| `neurolab/gui/crossbar_4x4_view.py` | **+284/−437** | Reescritura: nueva arquitectura Controller; `_anim_tick`; sincronización core↔elementos; modos y estilos |
| `neurolab/gui/base_view.py` | +12/−3 | Ajustes de dibujo/zoom |
| `neurolab/gui/crossbar_elements/memristor_element.py` | +2/−2 | RON default 2000 (µS coherente) |
| `neurolab/neurons/lif.py` | +6/−3 | Bug diodo ideal `max(V_in−V_m,0)` (sin reflujo) |
| `neurolab/devices/strukov.py` + `models/strukov.py` | +9/−2 y +3/−0 | Ajustes del modelo Strukov (R_on/R_off) |
| `neurolab/gui/tests/tests_2x2/test_07_ciclo.py` | +1/−1 | Bugfix compartido ciclo |
| `neurolab/tests/crossbar/test_crossbar_4x4.py` | **+70/−44** | Sustituye `test_crossbar_4x4_7_bug_fixes` por `test_crossbar_4x4_core_physics_integration` + `test_crossbar_4x4_physical_realism_integration` |
| `neurolab/tests/crossbar/test_crossbar_1x1.py` | +1/−1 | Rango de G: 1000e-6 → **500e-6** (por R_on=2000) |
| `neurolab/tests/crossbar/test_plasticity.py` | +15/−0 | Nuevo `test_rstdp_delayed_reward_uses_eligibility` |
| `neuromorphic_lab/configs/last_session.json` | +3/−3 | Sesión persistida |
| `neuromorphic_lab/neurolab/synapses/memristive_synapse.py` | +1/−1 | Ajuste secundario |

### 9.2 Nuevos (untracked)

| Archivo | Rol |
|---|---|
| `neurolab/gui/crossbar/controller_4x4.py` | Controlador MVC del 4×4 (analizado en §6.2) |
| `tests/crossbar/test_crossbar_4x4_stress.py` | Tests de estrés (§10) |
| `tests/gui/test_crossbar_4x4_gui.py` | Tests GUI (§10) |

### 9.3 Implicación

Los cambios de `V_th` y `R_on` son coherentes entre sí (R_on=2000 ⇒ G_max=500 µS; V_th=1.2 ⇒ mitad de la cruz V/2 a 1.0 V no programa, protegiendo half-select). No se han ejecutado en este informe modificaciones al código ni tests que cambien el repositorio.

---

## 10. Tests pytest dedicados al 4×4

### 10.1 `tests/crossbar/test_crossbar_4x4.py` (6 tests tras el cambio sin commitear)

| Test | Verifica |
|---|---|
| `test_crossbar_4x4_initialization` | `n_rows=n_cols=4`, `memristors.shape==(4,4)`, `G_matrix.shape==(4,4)` |
| `test_crossbar_4x4_matrix_helpers` | `get_G_matrix`, `compute_currents`, `matrix_to_heatmap_string` |
| `test_all_20_draw_functions_4x4` (parametrizado ×20) | Cada `draw_4x4_XX` con `DummyGUI` devuelve `dict` con `status=='PASS'` y `'metrics'` |
| `test_crossbar_4x4_volatile_elements` | La vista contiene `M_v1..M_v4` como `VolatileMemristorElement` |
| `test_crossbar_4x4_core_physics_integration` | El core se usa en `program_v2`: mock de `update_memristors` llamado; `V_rows[0]==1.0`, `V_cols[0]==−1.0`; transiciones STDP/R-STDP limpias |
| `test_crossbar_4x4_physical_realism_integration` | R-STDP consume R en 1 tick y potencia (dG>0); volátil relaja hacia `x0=0.05` sin caer debajo; `I_cols>0` con `V_row[0]=2 V` |

### 10.2 `tests/crossbar/test_crossbar_4x4_stress.py` (NUEVO, 5 tests)

| Test | Duración simulada | Aserciones clave |
|---|---|---|
| `test_crossbar_4x4_continuous_1000_ticks_stress` | 30 s (1000×30 ms) | Sin NaN/Inf; `I_cols ≥ 0`; toda G ∈ [0.9, 500.1] µS |
| `test_crossbar_4x4_plasticity_stdp_rstdp_stress` | 250+250 ticks | G acotada en STDP y R-STDP; `rstdp_rule.cfg.R == 0` (consumo) |
| `test_crossbar_4x4_volatile_relaxation_stress` | 100 ticks | Los 4 M_v relajan `x→0.05 ±0.01` |
| `test_crossbar_4x4_mode_switching_stress` | 50 ciclos × 4 modos | Sin NaN en V_rows/V_cols al alternar modos |
| `test_crossbar_4x4_zero_voltage_rows_must_not_potentiate` | 50 ticks | M2*/M4* (filas V=0) quedan < 150 µS |

### 10.3 `tests/gui/test_crossbar_4x4_gui.py` (NUEVO, 3 tests)

| Test | Escenario |
|---|---|
| `test_crossbar_4x4_gui_initialization` | S1..S4, M11..M44, M_v1, LIF_1, Act_1 presentes |
| `test_crossbar_4x4_zero_voltage_rows_do_not_potentiate_in_stdp` | S1=0.8, S2=0, S3=1.1, S4=0 + 100 ticks de STDP → filas 2 y 4 < 100 µS |
| `test_crossbar_4x4_read_mode_does_not_inflate_conductance` | Modo 1 + 100 ticks → variación de G < 1 µS |

### 10.4 Cobertura de `.pytest_cache`

Los `nodeids` en caché confirman ejecuciones previas de `tests/crossbar/test_crossbar_4x4.py::test_all_20_draw_functions_4x4[draw_4x4_01…20]`, `tests/gui/test_crossbar_4x4_gui.py::*` y `tests/gui/test_crossbar_gui.py::test_crossbar_elements_hit_test`, `test_crossbar_pan_zoom`.

---

## 11. Documentación y referencias

### 11.1 `docs/latex/capitulos/06_fase4_crossbar.tex` (Etapa 4.1)

- Sección «Matriz Crossbar e In-Memory»: define el arreglo M×N de nano-hilos con memristor en cada cruce.
- **Validación 4×4 contra Prezioso 2015:** (1) curva I-V virgen **Fig S3a** (fuga < 10 nA, ramas asimétricas); (2) **censo de red 4×4 (Fig S3b)**: con `V_read=0.1 V` el simulador reporta conductancia media ≈ **0.45 µS** (piso térmico calibrado); (3) **modelado estocástico (Fig S3c)**: CV matricial coherente con la varianza D2D real. Relevancia: «…matriz 4×4 dotado con la misma varianza estocástica D2D observada en el hardware físico… acredita que cualquier red entrenada lidiará con los mismos niveles realistas de ruido espacial que un chip real».
- Figuras citadas: `23a_prezioso_s3a.png`, `23b_prezioso_s3b.png`, `23c_prezioso_s3c.png`, `fig_22_prezioso_v2.png` (asimetría SET/RESET).

### 11.2 `docs/markdown/Documento_Maestro_Tesis_Neuromorfica.md`

- **Sneak paths:** `Isneak 4×4 = 0.23 µA` (simulador) vs `1.45 µA` (paper) → **error 84.1 %**; el documento explica que en matrices pequeñas el modelo teórico simplificado choca con la simulación, pero en 64×64 converge (16.8 %). (Nota: esos valores provienen del legacy `paso22`/`paso27`, no de las draw_4x4.)
- **Noise margin:** `NM 4×4 = 0.95` vs `0.91` (paper) → **error 4.3 %** (precisión excelente en redes pequeñas).

### 11.3 `docs/07_CROSSBAR_PLASTICIDAD_Y_SNEAK_PATHS.md`

- Estados HRS/LRS (`R_off`/`R_on`), crossbar físico (`PhysicalCrossbarArray`), sneak paths con modelo V/2 (`R_sneak = R2+R3+R4`), plasticidad LTP/LTD/STDP, riesgos de escalado, y catálogo de `paso16`–`paso24`. No menciona la vista 4×4 de la GUI (documenta solo la capa legacy).

### 11.4 Legacy (`legacy/memristor_simulator/steps/`)

- `paso18_crossbar_escalable.py`: crossbar **8×4** (N_IN=8, N_OUT=4, 32 memristores) con clasificación de patrones A/B y MAC en hardware.
- `paso19_caracterizacion_crossbar.py`: caracterización **10×8 virgen** (Prezioso S3).
- `paso22_analisis_sneak_paths.py` y `paso23_analisis_noise_margin.py`: estudio de sneak/noise margin **de 4×4 a 64×64** (origen de la tabla del Documento Maestro).

---

## 12. Conclusión

1. El **Crossbar 4×4 es el corazón por defecto** del simulador (`CrossbarConfig` 4×4) y existe como **motor físico** (`neurolab.crossbar`) + **esquema GUI robótico** (`crossbar_4x4_view` + `controller_4x4`), con 20 pruebas de validación, 5 tests de estrés y 3 tests GUI dedicados.
2. El núcleo está **bien fundamentado**: lectura `I=G^T·V` exacta, programación selectiva V/2 con máscara `V_th` (half-select protegido), plasticidad STDP/R-STDP realista (elegibilidad diferida) y validadores oficiales correctos.
3. La **deuda principal** está en las pruebas de figura `draw_4x4_03/08/09` (H1): no activan los flags de no idealidad y por tanto **validan el caso ideal**, no el físico. El resto de hallazgos (H2–H8) son de duplicación/simplificación sin impacto funcional crítico.
4. El working tree contiene **una reescritura sustancial sin commitear** del 4×4 (vista + controller + R-STDP + umbrales), coherente internamente y respaldada por tests nuevos; conviene revisar y commitear, y eliminar los `loop_*.png` residuales de la raíz.

---

*Fin del informe.*