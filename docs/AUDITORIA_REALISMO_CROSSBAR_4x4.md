# AUDITORÍA DE REALISMO FÍSICO — CROSSBAR 4×4
### «Todo lo que está mal cuando se exige que actúe como un crossbar memristivo realista»

> **Objetivo:** no repetir el informe descriptivo anterior (`INFORME_CROSSBAR_4x4.md`). Aquí se evalúa **cada aspecto del Crossbar 4×4 contra lo que hace un crossbar físico real** y se lista, con severidad, todo lo que NO es realista, está mal, es inconsistente o se da por supuesto sin verificación.
> **Fecha:** 2026-09-25 · **Rama `dev`** (HEAD `05ca3e0`, con cambios sin commitear incluidos).
> **Criterio de gravedad:** 🔴 CRÍTICO (físicamente imposible o rompe el propósito) · 🟠 ALTO (desviación física seria) · 🟡 MEDIO (simplificación con impacto medible) · 🔵 BAJO (estilo/consistencia/reproducibilidad).

---

## 1. Criterio de evaluación: ¿qué es un crossbar 4×4 realista?

Un arreglo memristivo 4×4 físico debe cumplir como mínimo:

| R1 | **Celda = memristor real** — no lineal, con umbral de conmutación propio, asimetría SET/RESET (o al menos un modelo documentado), variabilidad D2D/C2C, ruido de lectura y rango dinámico coherente entre R_on y R_off. |
| R2 | **Electrónica de red real** — filas/columnas metálicas con resistencia finita; lectura con columnas a tierra y corriente medida por columna; la red cumple **KCL/KVL nodal** (no fórmulas independientes por celda). |
| R3 | **Lectura no destructiva** — `V_read` muy por debajo del umbral de conmutación; al leer, G no cambia (por umbral físico, no por «no llamar a la función de escritura»). |
| R4 | **Escritura selectiva real** — esquemas 1T1R / V/2 / V/3 con **disturbio half-select y cross-talk explícitos** (no eliminados por artimaña), write-verify y no saturación en un solo pulso. |
| R5 | **Fenómenos parásitos conservativos** — sneak paths que conservan corriente (se resuelven KCL), IR-drop correcto por tramo de línea, ruido de lectura, retención y resistencia (endurance). |
| R6 | **Plasticidad en hardware** — el STDP/R-STDP se materializa **como pulsos de voltaje que modifican físicamente G del memristor**; no parches numéricos directos sobre G. |
| R7 | **Acoplamiento neuromórfico real** — las corrientes de columna alimentan neuronas mediante impedancias físicas reales (sin factores mágicos) y la inhibición es sináptica. |
| R8 | **Reproducibilidad y coherencia** — un parámetro físico tiene el mismo valor en todas las capas; mismas semillas → mismos resultados. |

**Veredicto general:** el código cumple holgadamente **R1-parcial y R2→solo matemática ideal**, y falla o fuerza **R3–R8** en múltiples puntos. Los defectos se detallan a continuación.

---

## 2. Resumen ejecutivo (por severidad)

| Severidad | Cantidad | Ítems |
|---|---|---|
| 🔴 CRÍTICO | 6 | KCL no resuelto; fórmulas de IR-drop físicamente erróneas; sneak no conservativo; STDP sin física; umbrales contradictorios (0.5/1.2 V); saturación de escritura por `dt` por defecto |
| 🟠 ALTO | 12 | x0 ignorado por D2D (premisas falsas de tests); disturbio half-select real sin aislar; R_on inconsistente (100 vs 2000 Ω); G_min inalcanzable; LIF_I_scale mágico; WTA no sináptico; volátil heurístico; etc. |
| 🟡 MEDIO | 10 | reproducibilidad GUI, dispatch `program()` a ROW, tests 08/09 ideales, dead code, etc. |
| 🔵 BAJO | 8 | parámetro muerto, sombreado de variable, docstrings desactualizados, etc. |

Total: **~36 hallazgos**, referenciados como **F01…F36** en las secciones 4–8. Al final, sección 10: checklist «qué le falta para ser un crossbar realista».

---

## 3. Nota metodológica

- Se auditaron los archivos responsables de la física del 4×4: `core/memristor.py`, `core/config.py`, `core/base_device.py`, `devices/strukov.py`, `devices/models/strukov.py`, `devices/realism/*.py`, `crossbar/*.py` (core, line, sneak, configs, plasticity), `gui/crossbar_4x4_view.py`, `gui/crossbar/controller_4x4.py`, `gui/crossbar_elements/*`, `neurons/lif.py`, `synapses/*` y los tests 4×4.
- Todas las afirmaciones numéricas se derivaron de los parámetros del propio código (sección 9 recoge los cálculos).

---

## 4. Defectos del modelo de dispositivo (F01–F08)

### F01 🔴 CRÍTICO — El modelo de celda es el «memristor ideal» lineal, no el de Strukov 2008 no lineal
- **Dónde:** `devices/models/strukov.py:44–45` → `dx/dt = (mu_v·R_on/D²)·I(t)`; `core/memristor.py:138` integra Euler.
- **Qué hace el código:** deriva lineal, simétrica en signo y proporcional a la corriente.
- **Qué haría un crossbar real:** la conmutación en TiO₂ (Strukov 2008) es **exponencial/activada por voltaje** y asimétrica entre SET y RESET (ver `PreziosoConfig`, que sí modela A_p=200 vs A_n=600 y umbrales V_p/V_n, pero **no se usa** por defecto en el 4×4). La celda del 4×4 es simétrica e ideal; LTP y LTD escriben/borran con la misma «velocidad», irreal para óxidos reales.
- **Impacto:** toda física derivada (simetría, tiempos de conmutación, ventanas de operación) es la de un dispositivo ficticio, no de un memristor físico.

### F02 🔴 CRÍTICO — Umbral de conmutación de la celda (0.5 V) ≠ umbral de máscara del crossbar (1.2 V)
- **Dónde:** `models/strukov.py:41` → `getattr(cfg,'V_th',0.5)`; `core/config.StrukovConfig` NO define `V_th` → cae a 0.5 V. En cambio `CrossbarConfig.V_th=1.2` (configs.py:53) se usa solo para la **máscara** de `update_memristors`/`program_V2` (`core.py:297–300`).
- **Qué hace el código:** conviven **dos umbrales**: la celda cambia de estado con |V| ≥ 0.5 V; el crossbar solo protege celdas con |V| ≥ 1.2 V.
- **Consecuencia física:** en `program_V2` **sin** `isolate_half_select=True` (la API pública por defecto), las celdas half-selected reciben 1.0 V → **> 0.5 V** → **DISTURBIO HALF-SELECT REAL**. La protección solo existe en el camino GUI (que pasa `isolate_half_select=True`). Un usuario de la librería que llame a `crossbar.program_V2(i,j)` corrompe toda la cruz.
- **Además:** `ReadConfig.V_th=1.2` y `WriteConfig.V_th=1.2` (addressing.py) son valores **informativos**: ningún código los usa para el umbral físico de la celda.

### F03 🔴 CRÍTICO — Un solo pulso de escritura por defecto satura la celda (escritura en bucle abierto)
- **Dónde:** `core.py:318–319` (y 337–338, 357–358, 385–386): `dt = getattr(cfg,'dt_physics', …)` → `dt_physics = 0.02 s`.
- **Números (sección 9):** con G≈70 µS, a 1 V → dx/dt ≈ 14 s⁻¹; con `dt=0.02 s` → Δx ≈ **0.28 por pulso**. Un pulso a 2 V (dx/dt≈28 s⁻¹) → Δx ≈ **0.56** → la celda casi se satura con un solo `program_V2()` sin pasar `dt`.
- **Qué haría un crossbar real:** la escritura debe ser **pulsada y verificada (write-verify)** para aterrizar en una G objetivo; aquí es un salto en bucle abierto sin control de convergencia ni verificación posterior.
- **Nota:** la GUI disfraza el problema pasando `dt=t_pulse=1e-3` (Δx≈0.014/pulso), pero la librería pura no es utilizable de forma realista.

### F04 🟠 ALTO — `x0` de `CrossbarConfig` es ignorado cuando D2D está activa (default)
- **Dónde:** `core.py:110–112`: si `enable_d2d`/`d2d_enabled` (True por defecto) → `G_0 = G_mean·d2d_factor` y `_set_cell_conductance` despeja `x` **ignorando `cfg.x0`**.
- **Consecuencia:** los tests `draw_4x4_05/06/07/12` configuran `x0=0.30/0.70` «para tener rango dinámico visible (130→160 µS)» y **esa premisa es falsa**: la matriz arranca siempre en **G ≈ G_mean·(1+ε) ≈ 70 µS** (x≈0.106). Los comentarios y métricas de esas pruebas describen un estado de trabajo que el código no produce.
- **Impacto:** la «selección de punto de trabajo» documentada no existe; el usuario que fija `x0` cree controlar el estado inicial y no lo controla.

### F05 🟠 ALTO — Rango dinámico inconsistente: `G_min=1 µS` es inalcanzable con `R_off=16 kΩ`
- **Dónde:** `CrossbarConfig.G_min=1e-6` (configs.py:48) y `STDPConfig.G_min=1e-6` (plasticity.py:75); pero `R_off=16000 Ω` impone **G ≥ 62.5 µS**.
- **Qué hace el código:** el clip STDP y los límites de doc prometen [1, 500] µS; la física solo permite [62.5, 500] µS. Un `set_conductance(i,j, 5e-6)` (vía `_set_cell_conductance`) calcula x=(16000−200000)/14000 **negativo** y lo clampa a 0 → la celda queda en **G=62.5 µS, no en 5 µS**, silenciosamente.
- **Impacto:** violación de la invariante física; cualquier algoritmo que programe G<62.5 µS recibe otro valor sin aviso.

### F06 🟠 ALTO — `R_on` tiene **tres valores distintos** según la capa
- **Dónde:** `CrossbarConfig.R_on=2000` (configs.py:31, reciente) vs `core/config.ElectricalConfig.r_on=100` y `StrukovConfig.RON=100` (core/config.py:15,42) vs `config_dialogs.MEMRISTOR_PRESETS['strukov'].RON=100` (config_dialogs.py:23) vs `validation_tests_panel._get_live_crossbar_config` fallback `r_on=100` vs `MemristorElement.RON=2000`.
- **Consecuencia:** `MemristorStrukov()` aislado → G_max=10 mS; una celda del 4×4 → G_max=500 µS. Las pruebas 1×1 originales (con 1000 µS) y los presets de la GUI (100 Ω) pertenecen a otro dispositivo.
- **Impacto:** métricas, validaciones y figuras dependen de qué constructor se use; no hay una única «ficha técnica» del dispositivo.

### F07 🟠 ALTO — Ventana de Biolek con `p=2` y sin calibración contra curvas reales
- **Dónde:** `devices/strukov.py:39–43` lee `window_type/window_p` de un `StrukovConfig` que **no los define** → cae a `'Biolek'`, `p=2`; `realism/window.py:32` aplica `f = 1 − (x − stp(−i))^(2p)`.
- **Qué haría un crossbar real:** las ventanas de frontera deben ajustar la dinámica para cuadrar las curvas I–V medidas; aquí el efecto no está calibrado contra ningún dato (p=2 arbitrario) y es meramente decorativo.
- **Impacto:** el «realismo de ventana» prometido no está validado.

### F08 🟡 MEDIO — Euler de primer orden sin límite de estabilidad
- **Dónde:** `core/memristor.py:138`: `new_x = x + dxdt·dt`; sin control de `|dxdt·dt|` salvo el clip final.
- **Qué haría un crossbar real:** la integración debe ser estable para `dt` arbitrario; con `dt_physics=0.02 s` y dxdt de decenas por segundo (F03) el paso es inestable y el clip enmascara la pérdida de física (cualquier pulso «satura»).
- **Nota:** el soft-clamp de R ≥ 1 Ω (`memristor.py:76–89`, añadido 2026.09.11) evita explosiones numéricas, pero no corrige la inexactitud del paso grande.

---

## 5. Defectos de la electrónica de red (F09–F15)

### F09 🔴 CRÍTICO — No se resuelve KCL/KVL nodal: la lectura «lineal» ignora la red
- **Dónde:** `core.py:192` `read_ideal` → `I_out = G_matrix.T @ V`. Es **matemática de matriz**, no circuito.
- **Qué haría un crossbar real:** con filas y columnas conductoras, las corrientes se reparten según la red (las columnas a tierra crean acoplamientos); un crossbar físico se resuelve montando el sistema nodal (matriz laplaciana ponderada por G y R_line).
- **Impacto:** la «operación matricial» es una idealización que **coincide** con el caso R_line=0 y columnas a tierra, pero todo el análisis de no idealidades parte de ella sin modelo de red.

### F10 🔴 CRÍTICO — La fórmula de IR-drop en `LineResistanceModel.solve` es físicamente incorrecta
- **Dónde:** `line_resistance.py:66–95`.
  - `drop_H(i,j) = Σ_{k=0..j} I_cols[j]·R_H·k` y `drop_V(i,j) = Σ_{k=0..i} I_cols[j]·R_V·k`.
- **Errores físicos:**
  1. La corriente que circula por un **tramo horizontal k de la fila i** es la suma de las corrientes de las celdas `(i, m)` con `m > k` — **no** `I_cols[j]`. Usar la corriente de la columna j (`I_cols[j]`) para el tramo horizontal de la celda (i,j) rompe la conservación de corriente.
  2. La **caída vertical** también se calcula con `I_cols[j]` — la corriente de una columna que además cae verticalmente; pero la celda (i,j) tiene una sola corriente `G[i,j]·V_eff[i,j]`, no dos.
  3. El sistema no plantea **KCL en ningún nodo**: `I_new[j] = Σ_i G[i,j]·V_eff[i,j]` ignora que la corriente debe entrar/salir por tierra de columna y que las líneas comparten corrientes.
- **Qué haría un crossbar real:** resolver `(G + L)·V = I_fuente` con la matriz laplaciana de la red (filas + columnas + tierra).
- **Impacto:** las figuras 03 y 09 y el validador `validate_line_resistance` cuantifican un efecto «tipo IR-drop» que no es el IR-drop real de la red.

### F11 🔴 CRÍTICO — El modelo de sneak paths no conserva corriente
- **Dónde:** `core.py:205–212`: `I_sneak[j] += alpha·G[i,j]·V_mean·n_vecinos` con `V_mean = mean(|V|)`.
- **Qué hace el código:** añade a cada columna una fuga proporcional a `alpha` y a la **media de voltajes** (una magnitud global sin sentido físico local) y al número de vecinos **de fila** (algo geométrico que no aparece en la física de corriente).
- **Qué haría un crossbar real:** las corrientes parásitas surgen de resolver la red con la columna objetivo leída y el resto de columnas en tierra/flotando (esquema de lectura); se conserva I_total.
- **Impacto:** el «error sneak» resultante (≈ 10–20 %) es un **artefacto de la fórmula**, no una predicción física; crece con N de forma lineal en vecinos, no según la topología real.

### F12 🟠 ALTO — `read()` no permite line + sneak simultáneos (elif)
- **Dónde:** `core.py:229–233`: `if enable_sneak_paths … elif enable_line_resistance … else ideal`.
- **Qué haría un crossbar real:** un chip tiene **ambas** no idealidades a la vez; el modelo obliga a elegir una y descarta la otra (o ninguna).
- **Impacto:** ninguna simulación del 4×4 combina IR-drop y sneak, como en el hardware.

### F13 🟠 ALTO — Los drivers/decoders son decorativos: no participan en la física
- **Dónde:** `crossbar_4x4_view.py:551–554`: `row_decoder/col_decoder/row_driver/col_driver` se instancian y **dibujan**, pero la tensión la fija directamente `V_rows[tg_r]=V/2` (`_anim_tick`), sin pasar por `AddressDecoder.decode()` ni por `RowDriver.drive()`.
- **Qué haría un crossbar real:** el direccionamiento físico (compuertas ONE-HOT + drivers de palabra/bit) **es** el mecanismo que aplica tensión; aquí es presentación.
- **Impacto:** el esquema «hardware de dirección» mostrado en el canvas no opera sobre el modelo.

### F14 🟠 ALTO — Los elementos `M_ij` y el `Crossbar` core se sincronizan solo en `program_v2`
- **Dónde:** `controller_4x4.py:112–117` sincroniza core ← elementos **solo en `mode=="program_v2"`**; en read/STDP usa `get_G_matrix(elements)` y parchea G con `crossbar.set_conductance` solo cuando hay dG.
- **Consecuencia:** tras configurar la matriz con `_on_uniform`/`BatchGridDialog`/clics, el **core queda desincronizado** (G vieja) hasta la próxima escritura V/2; un `view.crossbar.G_matrix` no refleja lo que se ve/pinta.
- **Impacto:** estado «fantasma» duplicado (elementos vs core) sin una fuente de verdad única.

### F15 🟡 MEDIO — Falta el circuito de lectura: sin resistencia de sensor ni ruido de medida
- **Dónde:** lectura = `G.T @ V` directa; `chk_noise=False` en los elementos (`memristor_element.py:50`); no existe bitline resistor.
- **Qué haría un crossbar real:** cada columna se lee con amperímetro/`R_sense` (con su caída y ruido), y hay ruido de lectura.
- **Impacto:** lecturas sin ruido → márgenes de ruido optimistas (los valores «NM 4×4=0.95» del Documento Maestro son ideales).

---

## 6. Defectos del modo interactivo / neuromórfico (F16–F24)

### F16 🟠 ALTO — `LIF_I_scale = 0.02` es un factor mágico sin base circuital
- **Dónde:** `controller_4x4.py:53` (y su gemelo sin uso `LIF_I_scale=1e-4` en `crossbar_4x4_view.py:620`).
- **Qué hace el código:** `I_syn = I_cols·0.02` y se inyecta como corriente directa en la LIF.
- **Qué haría un crossbar real:** las columnas entregan corriente a la neurona a través de una **impedancia física** (R_series) con unidades coherentes; el 0.02 es una atenuación «para que dispare ≈1 V» y no deriva de ningún parámetro eléctrico.
- **Impacto:** la tensión de membrana resultante (V_eq≈1.4 V, sección 9) es un accidente de calibración, no un diseño de circuito.

### F17 🟠 ALTO — La inhibición lateral WTA no es sináptica
- **Dónde:** `controller_4x4.py:176–179`: si hay ganador activo, las demás neuronas se ponen `V_m = 0` directamente.
- **Qué haría un crossbar real:** la inhibición lateral se implementa con **sinapsis inhibitorias** (o corriente inhibitoria hija de la actividad de columnas vecinas); descargar V_m a cero es un reset arbitrario, no inhibición.
- **Impacto:** el «bus de inhibición WTA» que dibuja el canvas (`crossbar_4x4_view.py:497–512`) es decoración: los algoritmos winner-take-all son de **software**, no del crossbar físico.
- **Además:** `is_animating` es un parámetro de `step()` que **nunca se usa** (solo aparece en la firma, `controller_4x4.py:103`).

### F18 🟠 ALTO — El memristor volátil de la GUI no es el modelo físico volátil
- **Dónde:** `controller_4x4.py:151–162`: `dx_v/dt = 100·I_cols[j] − (x_v−x_eq)/τ_relax`, con constante **100 s/A** implícita y dependiente de la **corriente de columna** (no del voltaje en la celda M_v).
- **Qué hace el código:** coexisten **dos** «volátiles»: el físico `VolatileDecayModifier` (`realism/volatile.py`, correcto: `− (x−x_eq)/τ`) y el heurístico del controller, que además carga M_v por la corriente de la columna completa (como si M_v estuviera en serie con toda la columna) **sin** umbral de activación.
- **Qué haría un crossbar real:** un TSM difusivo potencia con pulsos **sobre-umbral** y decae con τ; no con `I_col · 100`.
- **Impacto:** el test `volatile_relaxation_stress` valida el heurístico, no la física del dispositivo volátil.

### F19 🟡 MEDIO — Las corrientes mostradas durante programación son post-escritura, no transitorias
- **Dónde:** `controller_4x4.py:132–151`: en `program_v2` primero se actualiza el core (`update_memristors`) y **después** se calcula `I_cols = Σ G·(V_row−V_col)`. El «pulso» visible en la GUI es la corriente del estado **ya saturado** con G nueva, no la corriente instantánea del pulso (que en la celda objetivo con 2 V y R≈14 kΩ ≈ 143 µA).
- **Impacto:** la telemetría V/2 es engañosa para entender la escritura.

### F20 🟡 MEDIO — Defaults de sensores sobre el umbral de lectura
- **Dónde:** `crossbar_4x4_view.py:653`: `V_out = 0.8/0.4/0.3/0.2 V`; `gui_integration.V_READ_THRESHOLD = 0.49 V`.
- **Consecuencia:** S1 por defecto (0.8 V) entra en «escritura» según la clasificación de badges (se pinta rojo/rosa en lectura pura), aunque el modo lectura no escribe (solo porque la física no se integra en read). Es un ejemplo de que **la no destructividad es accidental** (F23): el diseño permitiría «lectura» a 0.8–2 V sin que el modelo se queje.

### F21 🔵 BAJO — `_on_simulate` declara `dt=0.01` y llama a `_anim_tick()` que usa su propio dt
- **Dónde:** `crossbar_4x4_view.py:1314–1317`: la variable `dt` local no se usa; el «paso» real lo define el intervalo del timer (0.03 s).

### F22 🟡 MEDIO — Reproducibilidad: las celdas de la GUI usan `random.randint` global (no sembrado)
- **Dónde:** `memristor_element.py:43,91,179`: `random.randint(10000,999999)` sobre el RNG global de Python; cada apertura de la vista produce **factores D2D distintos** (las semillas «renovables» al reset lo empeoran); el `seed` del core (42) no tira de estos.
- **Qué haría un crossbar real:** la simulación debe ser reproducible corrida a corrida (misma semilla → misma matriz D2D); aquí no lo es.

### F23 🟡 MEDIO — «Lectura no destructiva» es accidental, no física
- **Dónde:** en `read()`/`read_ideal` nunca se llama a `update()` de las celdas; la prueba `draw_4x4_04` garantiza 0.000 % por construcción (ver INFORME anterior, H3).
- **Qué haría un crossbar real:** la no destructividad la impone `V_read << V_umbral` (R3). Aquí se garantiza porque simplemente no se integra la física en lectura — si alguien llamara `update_memristors` con 0.5 V, la celda **sí** cambiaría (F02), incluso a «voltaje de lectura».

### F24 🔴 CRÍTICO — La «STDP hardware» de la GUI no usa la física del memristor
- **Dónde:** `controller_4x4.py:226–257`: en modo read/STDP, `dG = rule.apply(...)` y luego **se parchea directamente** `mem.params['G']` + `crossbar.set_conductance(i,j,G_new)` con saturación lineal `(G_max−G)/(G_max−G_min)`.
- **Qué haría un crossbar real (R6):** la plasticidad surge de **pulsos de voltaje superpuestos** (pre/post) que hacen evolucionar `x` y por tanto G; aquí la regla STDP es un **ajuste numérico exógeno** y el memristor nunca «ve» esos pulsos.
- **Impacto:** el aprendizaje 4×4 de la GUI es un modelo matemático con plástico verbal, no implementación en hardware; la física Strukov solo participa en el modo V/2 de escritura manual.

---

## 7. Defectos de plasticidad y reglas de aprendizaje (F25–F28)

### F25 🟠 ALTO — Dos STDP distintos y no equivalentes (crossbar vs sinapsis)
- **Dónde:** `crossbar/plasticity.py` (trazas `Trace` + matriz dG) vs `synapses/stdp.py` (`delta_w(Δt)` exponencial + normalización porcentual).
- **Consecuencia:** `draw_4x4_11` importa `synapses.stdp.STDPRule` mientras la GUI ejecuta `crossbar.plasticity.STDPRule`; la «ventana STDP» de la figura 11 usa `A_plus=0.05/A_minus=−0.025/τ=17–34 ms` (¡casi el doble de τ que el de la GUI, 20 ms!) y **no** es la regla que aprende la matriz. Dos curvas, dos escalas de ΔG, ninguna advertencia.
- **Qué haría un crossbar real:** un único mecanismo físico (pulsos) con una sola curva bien definida.

### F26 🟠 ALTO — Las trazas STDP de la GUI con `dt=30 ms` y `τ=20 ms` decaen un 78 % por tick
- **Dónde:** `plasticity.py:54`: `decay = exp(−dt/τ)`; con dt=0.03 s y τ=0.02 s → `exp(−1.5) ≈ 0.223`.
- **Qué hace el código:** la traza de elegibilidad «recuerda» muy poco entre ticks discretos de 30 ms; el aprendizaje efectivo depende fuertemente del paso del timer (si el timer fuera 10 ms, la biología sería distinta). El τ = 20 ms es biológico, pero el paso de simulación (30 ms) es wall-clock del GUI, sin vínculo físico.
- **Impacto:** los resultados del aprendizaje 4×4 dependen de la frecuencia del QTimer, no de la física.

### F27 🟠 ALTO — Weight decay homeostático ad hoc
- **Dónde:** `controller_4x4.py:260–274`: columnas no ganadoras → `G ← G + 0.005·(G_base − G)`.
- **Qué hace el código:** introduce una regla de homeostasis **no documentada físicamente** que compite con STDP (potencia vs decae) y cuyo `G_base=69.4 µS` y `rate=0.005` son arbitrarios.
- **Impacto:** el resultado del aprendizaje mixto STDP+decay no es atribuible a ninguna dinámica física del dispositivo.

### F28 🟡 MEDIO — `RSTDPRule.apply` filtra cambios cuando `R=0`, pero el uso de `_pending_reward` duplica estado
- **Dónde:** `plasticity.py:188–205`: `_pending_reward` + `cfg.R` + `eligibility` acumulan el mismo concepto en tres sitios (recompensa pendiente, recompensa activa y elegibilidad). La lógica es correcta tras la refactorización reciente, pero frágil: si un llamador setea `R` y no llama `apply`, la recompensa queda permanentemente pendiente sin aviso.
- **Nota:** verificado que en la GUI `_apply_reward` setea R y la siguiente `apply` la consume en 1 tick — comportamiento deseado y con test («R se resetea a 0»).

---

## 8. Defectos en pruebas, validación e ingeniería (F29–F36)

### F29 🔴 CRÍTICO — `draw_4x4_08` y `draw_4x4_09` (y 03) validan el caso ideal, no el físico
- **Dónde:** `tests_4x4/test_08_sneak.py`, `test_09_line_resistance.py`, `test_03_barrido_g.py`: usan `read_currents()` → `read()` → despacho por flags (`enable_sneak_paths`/`enable_line_resistance` = **False** por defecto). `CrossbarSneak`/`CrossbarLine` son alias de `Crossbar`.
- **Consecuencia:** el «error sneak» de la figura 08 es **0 %** y la «atenuación por R_line» de la 09 es **plana**; la figura 03 muestra un panel «real» idéntico al ideal. El documento las presenta como validación de no idealidades (falsa).
- **Nota:** el único sitio donde el sneak aparece es `draw_4x4_20` (usa `sneak_error_pct()`), y el validador oficial `validate_sneak_paths`/`validate_line_resistance` sí activa flags — la incoherencia entre «pruebas oficiales» y «pruebas de figura» es total.

### F30 🟠 ALTO — Las premisas de punto de trabajo de los tests 05/06/07/12 son falsas (F04)
- Los docstrings afirman que `x0=0.30 → G_inicial≈130 µS` y `x0=0.70 → G≈400 µS`; con D2D activa el `x0` se ignora y G arranca ≈70 µS. Las métricas «ΔG, rango dinámico» de esas figuras describen otro experimento.

### F31 🟠 ALTO — `test_crossbar_4x4_physical_realism_integration` prueba «realismo» con 2 V en modo lectura
- **Dónde:** `tests/crossbar/test_crossbar_4x4.py:190–193`: `V_rows_active = [2.0, 0, 0, 0]` (el comentario del test dice 0.8) y espera `I_cols > 0` en las 4 columnas (se cumple tristemente porque la fila 1 excita las 4 columnas con G≈70 µS: `I_j = G[0,j]·2 V > 0`).
- **El problema físico:** 2.0 V es **voltaje de programación** (≥ umbral de celda 0.5 V), no de lectura; el test «pasa» solo porque la GUI no integra la física en modo read (F23), no porque la lectura sea segura.
- **Impacto:** el test de «realismo físico» valida un accidente de implementación, no un comportamiento de hardware.

### F32 🟡 MEDIO — `test_crossbar_4x4_continuous_1000_ticks_stress` mete 1000 ticks de 30 ms con V≤2 V en modo read y exige G acotada — pero la física no integra en read
- **Dónde:** `tests/crossbar/test_crossbar_4x4_stress.py:38–59`. El test es un buen *smoke test* de estabilidad numérica, pero **no** verifica física: en read mode el core no integra y los elementos solo cambian si hay dG de STDP (está en off). Las aserciones de acotamiento son casi tautológicas.
- **Qué haría un crossbar real o estrés:** aplicar pulsos de escritura largos y verificar retención/drift y saturation transitoria.

### F33 🟡 MEDIO — Benchmark `draw_4x4_13` hace `update_memristors(1e-3)` con 0.5 V sobre las celdas
- Con umbral intrínseco 0.5 V (F02), `abs(0.5) < 0.5` es False → **las celdas SÍ derivan** (dxdt≈20 s⁻¹ → Δx=0.02/llamada). El benchmark mide «tiempo de actualización» pero, de paso, satura las celdas (cada tamaño usa un `cb` nuevo, así que no corrompe métricas, pero el patrón es incorrecto: lectura de benchmark no debería programar).

### F34 🟡 MEDIO — Código muerto / sin usar en el flujo 4×4
- `SneakPathModel` (`sneak_paths.py`) no se referencia desde core ni GUI.
- `Crossbar._effective_voltage` (core.py:269) y `Crossbar.read_voltages` (279) no se usan en la GUI; `is_animating` en `step()` no se usa (F17); `LIF_*` y `sequence_*` de la vista son parámetros muertos (la secuencia de filas programada no se ejecuta: `_anim_tick` usa el Poisson de sensores).
- La secuencia `sequence_period_ticks=40`/`time_scale_factor` se declaran pero **no** controlan la inyección real de patrones.

### F35 🔵 BAJO — `program()` despacha por defecto a ROW a pesar de `programming_mode='V2'`
- **Dónde:** `core.py:409`: `mode = getattr(cfg,'mode', getattr(cfg,'programming_mode','V2'))`; `cfg.mode = ProgrammingMode.ROW` (default). Llamar `crossbar.program(i,j)` **programa la fila**, ignorando `j`. Un usuario que espera V/2 por la doc (`programming_mode='V2'`) recibe ROW silenciosamente.

### F36 🔵 BAJO — Nombrado y docstrings engañosos
- `validate_programming_selectivity` (validation.py) reporta «PASS si ΔG_target > 0.1 µS» pero con D2D activa el delta incluye ruido; no verifica que half/idle ≈ 0.
- Docstrings de `draw_4x4_05/06` afirman «aislamiento 100 %» porque programan **la celda directa** (`memristors[0,0].update`) — eso es entrenar por atajo, no programación V/2 del crossbar: la prueba no ejercita el esquema de programación selectiva que la GUI sí usa.
- El alias `CrossbarIdeal`/`CrossbarSneak`/`CrossbarLine` sugiere clases distintas; son el mismo objeto con flags comunes: el «nombre» no describe el comportamiento.

---

## 9. Cálculos de soporte (con parámetros del código)

Parámetros: `mu_v=1e-14 m²/(V·s)`, `R_on=2000 Ω`, `D=10 nm`, `R_off=16000 Ω`, `G_mean=70 µS`, `V_th_celda=0.5 V`, `V_th_crossbar=1.2 V`, `dt_physics=0.02 s`, `t_pulse=1e-3 s`, `LIF: C=100 nF, R_leak=1 MΩ, v_th=1 V`.

| Magnitud | Fórmula | Valor con defaults | Referencia |
|---|---|---|---|
| Coeficiente de deriva | `K = mu_v·R_on/D²` | `1e-14·2e3/1e-16 = 2e5 s⁻¹·A⁻¹` | `models/strukov.py:45` |
| R inicial (x≈0.106) | `R = R_on·x + R_off·(1−x)` | ≈ 14 300 Ω → G≈70 µS | `core/memristor.py:88` |
| I a 1 V | `V/R` | ≈ 70 µA | `core/memristor.py:107` |
| dx/dt a 1 V | `K·I` | ≈ 14 s⁻¹ | — |
| Δx con dt=0.02 s (escritura default) | `dx/dt·dt` | **≈ 0.28** (≈ saturación, F03) | `core.py:318` |
| Δx con dt=1e-3 s (GUI) | ídem | ≈ 0.014 | `write_cfg.t_pulse` |
| I del pulso objetivo 2 V | `2/R` | ≈ 143 µA | — |
| Umbral de celda vs cruz V/2 | 1.0 V > 0.5 V | **disturbio half-select real en API pura** | F02 |
| V/3 en fila half | 2/3·2 V = 1.33 V > 1.2 V | disturbio también en V/3 (sin máscara) | F02 |
| G alcanzable mínima | `1/R_off` | **62.5 µS** (no 1 µS de `G_min`) | F05 |
| Decaimiento STDP por tick GUI | `exp(−dt/τ)` con dt=0.03, τ=0.02 | 0.223 (solo queda ~22 %) | F26 |
| V_eq LIF | `I_cols·0.02·R_leak` con 70 µA | 1.4 V > 1.0 V (dispara en ~2–3 ticks) | F16 |
| I_sneak (4×4, V̄=0.5, α=0.1) | `α·ΣG·V̄·n_vecinos` | artefacto sin conservación | F11 |

---

## 10. Checklist: «qué le falta al 4×4 para ser un crossbar realista»

- [ ] **Red nodal:** resolver el circuito eléctrico completo (matriz laplaciana `G + L`) para lectura y escritura (F09, F10, F12).
- [ ] **Un solo umbral físico** por dispositivo, propagado a todas las capas (F02) y una **ficha técnica única** de R_on/R_off/D/µ (F06).
- [ ] **Escritura pulsada con write-verify** y `dt` intencional, no `dt_physics` por defecto (F03).
- [ ] **Disturbio half-select explícito** (cuantificar crosstalk V/2 y V/3) en vez de eliminarlo con máscara (F02, F30).
- [ ] **Sneak paths conservativos** (KCL) — o al menos documentar que el modelo de vecinos es cualitativo (F11).
- [ ] **Retención, endurance y ruido de lectura** a nivel de arreglo (F15) y **asimetría SET/RESET** opcional (F01).
- [ ] **STDP implementado como pulsos** sobre el memristor (F24) y **una sola** regla/con curva (F25).
- [ ] **Acoplamiento LIF por impedancia real** (sin `LIF_I_scale`) (F16) e **inhibición sináptica** (F17).
- [ ] **M_v volátil usando el modelo físico** `VolatileDecayModifier` (F18).
- [ ] **Reproducibilidad:** semillas del RNG global para todos los elementos (F22) y **fuente única de verdad** (F14).
- [ ] **Corregir pruebas:** flags de no idealidad en figs 03/08/09 (F29), premisas de x0 (F30), eliminación de atajos `memristors[i,j].update` (F36), tests de estrés con escritura real (F32).

---

## 11. Conclusión

1. **El 4×4 actual es un simulador matricial ideal con ropaje físico**, no un crossbar realista: la «cirugía» está en la matemática exacta `I=G^T·V` (muy útil para demostraciones y como motor abstracto), pero el **circuito eléctrico real no se resuelve** (F09–F12) y las no idealidades son heurísticas o están desactivadas (F29).
2. Los **defectos críticos** son: células lineales simétricas con doble umbral (F01, F02), escrituras en bucle abierto que saturan (F03), IR-drop y sneak no-conservativos (F10, F11), y una **STDP que no pasa por la física** del dispositivo (F24).
3. La **GUI es un front-end muy pulido** (canvas, WTA, Poisson, telemetría) cuyo realismo se declara pero no se sostiene: factores mágicos (F16), reset arbitrario de membranas (F17), volátil heurístico (F18) y reproducibilidad nula (F22).
4. **Las pruebas validan las decisiones de implementación, no la física** (F29–F33): los PASS actuales no demuestran comportamiento de hardware.
5. La **vía más corta a un crossbar realista** en este código es: (a) resolver la red nodal para `read`/`write`, (b) eliminar los atajos de programación y umbrales duplicados, (c) conectar STDP al memristor con pulsos, y (d) arreglar las figuras de no idealidades. El resto son consolidaciones de parámetros y semillas.

---

## 12. Referencia cruzada con el informe previo

Este documento profundiza cada hallazgo H1–H9 del informe descriptivo (`docs/INFORME_CROSSBAR_4x4.md`), re-clasificándolos contra el criterio de realismo físico:

| Informe previo | Aquí | Mitigado/agravado |
|---|---|---|
| H1 (figs 08/09 ideales) | **F29 🔴** | — |
| H2 (doble STDP) | **F25 🟠** | agravado: τ/amplitudes distintas |
| H3 (no destructiva trivial) | **F23 🟡**, F31 | — |
| H4 (duplicados view/controller) | **F16, F21, F34** | `LIF_I_scale` duplicado |
| H5 (hardcode fig 20) | F36 | — |
| H6 (modelos simplificados) | **F09–F12 🔴** | ahora demostrado no-conservativo |
| H7 (doble D2D) | **F22, F04** | agravado: x0 perdido |
| H8 (rendimiento) | F34 🔵 | — |
| H9 (comportamientos correctos) | (sin contraparte crítica) | nota: correctos dentro del modelo, no del hardware |

---

*Fin de la auditoría de realismo.*