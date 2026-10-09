# Inventario del proyecto — 2026-10-08

> Snapshot verificado con comandos (no inferido de logs). Generado por Cline.
> **Untracked — NO commitear sin OK explícito del usuario.**

## 1. Git

- **Rama**: `dev` @ `102b950` | upstream `origin/dev` | **0 commits adelante (en sincronía)** | **worktree limpio**
- **Otra rama**: `main` @ `c7f2781` ("Commit inicial con .gitignore de Unity" — proyecto ajeno)
- **`.git`**: 165,65 MB → 2.030 objetos sueltos = **160,47 MiB** (casi seguro versiones viejas de `main.pdf`) vs 5,02 MiB en 3 packs → **candidato a `git gc`**

### Tags

| Tag | Hash | Fecha | Mensaje | Ancestro de dev | En origin |
|---|---|---|---|---|---|
| `entrega-final` | `c1ddda8` | 2026-10-08 | cierre: sincronizacion final tesis - valores ablacion A/B/C/D, legend 5 semillas, epsilon renderizable | ✅ | ✅ (hash idéntico) |
| `v0.2.0-physics-validated` | `2c609e7` | 2026-09-25 | fix(crossbar): reconectar MVC, reescribir sneak paths, IR drop, tests | ✅ | ✅ (hash idéntico) |

### Historial (últimos 30)

```
102b950 (HEAD -> dev, origin/dev) fix(neurolab): ecuaciones Strukov al elegir modelo custom + recargar modelo al cargar sesion
fa4fe3f chore: untrack last_session.json (estado de GUI) + ignore logs crudos de robot_sim
eede8c3 docs(3.4): ablaciones con corrida fresca R_off=16k - interpretacion cross-protocolo A/B/C/D
72ce15c robot_sim: scripts de referencia (run_pendulum, run_vision, verify_cartpole_2000)
2550050 docs: capitulo 09 marco practico (3.3.1-3.3.7) + figuras del capitulo
865018c robot_sim: generador de diagrama de arquitectura del agente (Paso 10)
eba6cb3 robot_sim: scripts y figuras de la extension de vision (run_vision)
146c202 robot_sim: entorno de navegacion con obstaculos (gridworld + agente R-STDP)
c1ddda8 (tag: entrega-final) cierre: sincronizacion final tesis - valores ablacion A/B/C/D, legend 5 semillas, epsilon renderizable
530045a fix(gui): update main_window.py with latest adjustments
8896243 fix(physics): set default v_th=0.1V for non-destructive read pulse protection while preserving continuous STDP switching
cbd0dd7 feat(gui): enable real-time automatic plot updates on subtab, spinbox, and combo changes in synapse config panel
60206f3 fix(gui): enable dynamic device switching (Subtabs 1, 2, 3) and set default v_th=0.0 in StrukovMathModel
4e40630 fix(gui): restore Jo 2010 LTP/LTD analytical exponential bell curve (R2=0.9310) and quantitative text box metrics
80213d7 fix(validation): adjust pulse width dt to 150us for Jo 2010 100-pulse LTP/LTD simulation
9dc536d docs(gui): update STDP spinboxes with microSiemens (uS) units and explicit label suffixes
3a9fa77 fix(hysteresis): calibrate IV sweep frequency and dynamic scaling for wide bowtie opening
a03845e fix(gui): expand amplitude spinbox ranges and fix negative stepping behavior
d426763 fix(rstdp): preserve user sensor spinboxes and strictly block memristor modifications in Read/Rest modes
38f02a5 feat(rstdp): interleave read/write/rest phases and add stochastic active sensor pattern selection
60b73a5 feat(rstdp): add Auto-RL Teacher with coherent random target environment
73cbeb7 fix(gui): reset sensor voltages, spinboxes, and winner_j on Reset button click
81420ba fix(stdp): calibrate interactive STDP learning rate A_plus and A_minus for clear visual Hebbian learning
88804ac fix(physics): propagate high sensor voltages >= V_th to physical memristor updates
7b651a1 fix(stdp): mask post-synaptic spikes on inhibited non-winner columns in competitive STDP
e50c77d fix(gui): dynamically sync V_program/2 to sensor spinboxes in V/2 mode
e2258ed fix(gui): restore initial state x0 and sync conductances to core on Reset button click
76e8c14 fix(physics): calibrate V/2 programming pulse step and physical half-select disturb pass-through
fd998c8 fix(crossbar4x4): disconnect LIF/WTA in program_v2 mode and sync core set_conductance on weight decay
95089d0 fix(crossbar): soft WTA inhibicion lateral continua sin reseteo duro V_m
```

## 2. Estructura

### Directorios top-level

| Dir | MB | Contenido |
|---|---|---|
| `scratch/` | 66,52 | binario Tectonic, pruebas sueltas, refs antiguas (ignorado) |
| `neuromorphic_lab/` | 42,03 | 879 archivos: `neurolab/`, `validaciones/`, `experiments/`, `tests/`, `validaciones_2x2/4x4/`, `validaciones_BACKUP_2026/`, `configs/`, `data_validation/`, `graficas/`, `figuras/`, `docs/`, `outputs/` (ignorado), `scratch/` (ignorado) |
| `outputs/` | 35,82 | 85 archivos en `figuras_paper/`, `figuras_verificacion/`, `output_modular/` (ignorado) |
| `docs/` | 33,25 | `latex/` (tesis) + `markdown/` + 13 guías/informes `.md` (161 archivos total) |
| `robot_sim/` | 23,98 | 203 archivos: `neurobot/`, `experiments/`, `legacy/`, `generate_*.py`, `run_*.py`, `outputs/` (ignorado), `figuras/` + `figures/` (21 trackeadas), `capitulo_tesis.tex`, 6 logs `.txt` (ignorados) |
| `archive/` | 5,78 | `fase1_verificacion/` + `fase3_verificacion/` extraídos (trackeados) + 2 `.zip` (ignorados) — 39 archivos |
| `legacy/` | 1,82 | `memristor_simulator/` (76 archivos) |
| `configs/` | ~0 | `last_session.json` (1.071 B) — **trackeado** (raíz) |
| `.pytest_cache/` | ~0 | caché (ignorado) |

### Archivos top-level
`.antigravityrules` (ignorado) · `.editorconfig` · `.gitignore` · `AUDITORIA_LAB.md` (11 KB) · `notas_campos_json.md` · `README.md` (15,5 KB) · `references.bib` (7,5 KB) · `requirements.txt` · `run_app.py` · `run_gui.bat` · **`run_robot.py` (0 B — vacío, trackeado)** · `run_validation.py` (9 KB) · `test_close.py` · `test_gui.py`

### `.gitignore` actual
Python · Pytest · IDE · LaTeX aux (`*.out *.log …`) · Backup/Scratch (`scratch/`, `outputs/`, `neuromorphic_lab/outputs/`, `neuromorphic_lab/scratch/`, `neuromorphic_lab/configs/last_session.json`, `robot_sim/log_*.txt`, `*.zip`) · Sistema · Tectonic.
Ignorados notables: 26 `.aux/.blg` huérfanos de capítulos viejos (`03_metodologia`, `04_validacion_cuantitativa`, `05_sistema_hibrido`, `06_plasticidad`, `07_crossbar`, …).

## 3. Tesis

- **`main.tex` incluye** (en orden): `00_preliminares`, `01_introduccion`, `02_marco_teorico`, `03_fase1_nucleo`, `04_fase2_hibrido`, `05_fase3_plasticidad`, `06_fase4_crossbar`, `07_discusion`, `08_conclusiones`, `09_marco_practico`, `apendices/A_codigo`, `apendices/B_datos`
- ⚠️ **09 va DESPUÉS de 08 Conclusiones** (los resultados del 09 no se citan en 08 ni en 07).

### Capítulos y jerarquía (condensada)

| Cap | Secciones |
|---|---|
| 00 | `*Acrónimos` · `*Símbolos` |
| 01 | Contexto · Problema · Objetivos (general + específicos) |
| 02 | Memristor (Strukov / Yakopcic-Prezioso / volátiles) · LIF · Plasticidad (STDP / R-STDP) |
| 03 | Etapa 1.1 modelo base + I-V · 1.2 variabilidad (cuali/cuanti) |
| 04 | Etapa 2.1 LIF + híbrido (cuali/cuanti) |
| 05 | Etapa 3.1 LTP/LTD + STDP (cuali ×2 + cuanti) |
| 06 | 4.1 crossbar+MNA · 4.2 NeuroBot 24×12 (encoder/MNA/softmax, R-STDP, resultados, ablaciones, heatmap) · Validación 10 experimentos · Síntesis |
| 07 | Discusión · Limitaciones · Reproducibilidad |
| 08 | Conclusiones Principales (4 items, **solo fases 1–4**) · Contribuciones ← *sin refs al 09; `**markdown**` crudo en el `.tex`* |
| 09 | Esquema · Herramientas · Desarrollo (Etapas 1–7) · Resultados y discusión (cross_task, especialización, ablaciones, limitaciones) · **Análisis de costos (§ vacío)** |

### Compilación

| Ítem | Valor |
|---|---|
| `docs/latex/main.pdf` | 13.505.395 B @ 2026-10-08 18:03:52 |
| Errores / refs undefined | **ninguno** |
| Overfull / Underfull | **22 / 3** |

### Notas
- `09_marco_practico.tex` es el **único capítulo sin `\label{ch:...}`** (falta `ch:marco_practico`).
- §3.5 "Análisis de costos" = `% (vacío por ahora)`.
- `robot_sim/capitulo_tesis.tex` (14 KB, trackeado) **no está incluido** en `main.tex`.

## 4. Figuras

| Métrica | Valor |
|---|---|
| Trackeadas en `docs/latex/figuras/` | **59** (idénticas a las 59 en disco — 0 untracked) |
| Referenciadas en `.tex` | **59** (49 en `figuras/` + 10 en `experiments/figures/`) |
| Faltantes (referenciadas sin archivo) | **ninguna** ✅ |
| **Huérfanas** (en disco, no referenciadas) | **7**: `fig2_mna_vs_ideal.png`, `fig3_v2_vs_v3.png`, `fig_15_stdp.png`, `fig_19_iv_hysteresis.png`, `fig_23_s3a.png`, `hfo2_lif_isi_decreciente.pdf`, `probe_100ep_curve.png` |
| `docs/latex/experiments/figures/` | 20 trackeadas = 20 en disco (10 PNG `exp01–10` + 10 JSON) ✅ |
| Duplicadas en `robot_sim/figuras/` | 21 trackeadas (comparativa/curva/fig1–5…) |

### Trazabilidad figura → script (por confianza)

- **✅ Literal** (el script escribe el nombre exacto): `23a/b/c_prezioso_s3` ← `validaciones/23_prezioso_s3.py` · `fig_14b`/`fig_15b` ← `validaciones/24_jo2010_white.py` · `fig_19_iv_hysteresis` ← `validaciones/19_iv_histeresis.py` (único que escribe directo en `docs/latex/figuras/`) · `comparativa_ablaciones_24x12` / `curva_aprendizaje_24x12` / `heatmap_matriz_X_24x12` (png+pdf) ← `robot_sim/generate_thesis_figures_24x12.py` · `agente_arquitectura` ← `generate_agente_arquitectura.py` · `vision_arquitectura` ← `generate_vision_arquitectura.py` · `vision_seeds` ← `generate_vision_seeds.py` · `vision_learning` ← `run_vision.py` · `obstacle_learning/mapa/trayectoria` ← `generate_obstacle_figures.py` · `obstacle_arquitectura` ← `generate_obstacle_arquitectura.py` · huérfanas `probe_100ep_curve` ← `plot_probe_curve.py`, `fig2_mna_vs_ideal`/`fig3_v2_vs_v3` ← `generate_figures.py`
- **🔢 Inferida por número+slug** (`fig_NN_slug` ↔ `validaciones/NN_*.py`, ~24): `fig_01,02,03,05,06,07,08,09,10,11,12,13,14,15,16,17,18,20,21a–d,22` → `01_strukov_2008` … `22_prezioso_v2`. ⚠️ Esos scripts escriben `validacion_*.png` en **su** carpeta → la llegada a `docs/latex/figuras/` es por renombrado/copiado **no automatizado** en el repo.
- **🟡 Mención parcial**: `lif_analitica` ← `run_all_validaciones.py` · `ltp_ltd`/`stdp` ← `14/15/16/17/18/19_*.py` (nombres `validacion_*`) · `mna_vs_ideal` ← `generate_figures.py`/`test_crossbar_suite.py`
- **❌ Sin generador en el repo** (ninguna mención ni en `archive/`/`legacy/`): `cartpole_learning`, `esquema_general`, `sistema_hibrido`, `sneak_ratio_vs_N`, `ir_drops`, `validacion_memristor`, `variabilidad_d2d_c2c` → renombrados manuales o scripts eliminados.

### Scripts con `savefig` (robot_sim)
`generate_thesis_figures_24x12`(6) · `generate_figures`(5) · `plot_results`(4: fig1–4 en dir propio, no tesis) · `generate_obstacle_figures`(3) · `run_vision`(2) · `generate_agente/obstacle/vision_arquitectura`, `generate_vision_seeds`, `plot_probe_curve`(1 c/u).

## 5. Código

| Métrica | Valor |
|---|---|
| `.py` trackeadas (repo) | **379** — **0 `.py` untracked** ✅ |
| `robot_sim/*.py` | **51**: `neurobot/` (9: agent, crossbar_brain, encoder, decoder, env_wrapper, lab_bridge, logger, rl_loop, tasks) · `legacy/` (11) · `experiments/` (3) · `generate_*` (7) · `run_*` (7: run_all, run_robot, run_vision, run_pendulum, run_ablations×2, …) · `obstacle_env/train` · `plot_results/probe_curve` · `probe_*` · `verify_cartpole_2000` · `diag_*` · `eval_yamil` · `test_headless` |
| `neuromorphic_lab/` | **381 archivos trackeados**: `neurolab` 152 · `validaciones` 89 · `experiments` 53 · `tests` 23 · `validaciones_4x4` 20 · `validaciones_2x2` 15 · `configs` 7 · `data_validation` 6 · `graficas` 6 · `figuras` 3 · `docs` 2 · README/requirements/pyproject/test_strukov/run_app 6 |
| Nota | `config_panel.py` (fix GUI de hoy) quedó commiteado en `102b950` ✅ |

## 6. Outputs

| Ubicación | Contenido | Tracking |
|---|---|---|
| `robot_sim/outputs/` | **44 JSON runs**: `run_A_base` (5,4 MB), `run_B_temp01` (5,3 MB), `run_D_hebbian` (2,1 MB), `run_C_noaccum` (2,0 MB), `verify_rstdp` (0,8 MB), `verify_cartpole_2000`, `run_vision_seed0–2`, `run_pendulum_seed0–2`, `run_obstacle_seed0–2` + `obstacle_seedN_paths`, `run_24x12_{A,B,C,D,temps}×seeds`, **`ablations_real_physics_summary.json`**, `verify_{contextual,bandit}`, `last_weights`, `Yamil*` | ignorado (`outputs/`) |
| `robot_sim/outputs/` | **36 `.npy`** (`*_X.npy` de cada run, `cartpole_X.npy`, `last_weights.npy`, …) | ignorado |
| `outputs/` (raíz) | 85 archivos: `output_modular/fig2b_complete.json` (5,2 MB), `fig2b_unity/metadata.json` (4,7 MB), `figuras_paper/`, `figuras_verificacion/` | ignorado |
| `neuromorphic_lab/outputs/` | `fase_4_3/reportes/metricas_consolidadas.json` (4 KB) + históricos | ignorado |
| `robot_sim/log_{pendulum,vision}_s{0,1,2}.txt` | 6 logs crudos de stdout (2 KB c/u) | ignorado (de hoy ✅) |
| `neuromorphic_lab/configs/last_session.json` | estado de GUI | **untracked+ignorado (chore de hoy ✅)** · ⚠️ `configs/last_session.json` de **raíz** sigue **trackeado** |

## 7. Pendientes

1. **Capítulo 08 — actualizar con resultados del 09** (fase C, inspeccionada pero no editada): añadir item 5 (CartPole R̄₂₀₀=77,8 · visión 100% · navegación 49,5–82% · ablaciones cross-protocolo) + `\label{ch:marco_practico}` en 09 + fix de `**markdown**`/bullets crudos en 08 → recompilar → commit con OK.
2. **§3.5 "Análisis de costos"** del 09: **vacío** — requiere datos de energía/área del crossbar (cap 06 o lab); si no existen → "trabajo futuro" (fase B, con search exploratorio previo).
3. **Trazabilidad de figuras**: 7 figuras sin generador localizado; el copiado `validacion_*.png` → `fig_NN_*.png` (24 figs) **no está automatizado** en el repo.
4. **22 overfull / 3 underfull** en `main.log`.
5. **7 huérfanas** en `docs/latex/figuras/` (candidatas a borrar o marcar).
6. `robot_sim/README.md` **vacío (0 B)**; `run_robot.py` (raíz) **vacío (0 B)**.
7. **`git gc`** — ~160 MiB en objetos sueltos de 165 MB totales.
8. 26 `.aux/.blg` huérfanos de capítulos viejos (ignorados, solo cosmético) + `configs/last_session.json` (raíz) trackeado (criterio idéntico al untrackeado hoy).
9. `robot_sim/capitulo_tesis.tex` trackeado pero **no incluido** en `main.tex` (¿capítulo huérfano?).
10. 21 figuras duplicadas trackeadas en `robot_sim/figuras/` frente a `docs/latex/figuras/`.

## 8. Próximos pasos sugeridos

1. **C (corto)**: ejecutar el update del cap 08 con el alcance del pendiente 1 → recompilar Tectonic → diff → commit+push con OK.
2. **B (largo)**: §3.5 costos — search previo de datos de energía/área disponibles; decidir "resultados" vs "trabajo futuro".
3. **Repo (opcional)**: `git gc`, limpieza de huérfanas, README de `robot_sim`.
4. Decidir si `INVENTARIO.md` se commitea (hoy: **untracked a propósito**).

---
*Errores durante el inventario (todos relanzados con éxito): regex `Overfull \hbox` en PowerShell → `-SimpleMatch`; `.Contains` sobre `$null` → guarda `[string]`; un comando de git redirigido a log por tiempo de respuesta.*
