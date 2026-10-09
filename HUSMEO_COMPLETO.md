# Husmeo completo del proyecto — 2026-10-08

> Barrido 100% lectura (Fases 1–5 de la consigna). Repo: `Memristor_Environment` · Rama `dev` @ `102b950` · Worktree limpio salvo `?? INVENTARIO.md` + este archivo.
> Reglas respetadas: sin edits de código, sin ejecutar experimentos, sin commits.

## 1. Capítulos y estructura

`main.tex` (87 líneas) incluye 12 archivos + 2 apéndices (L61–80). 15 `.tex` en total bajo `docs/latex`.

| Archivo | \chapter | Líneas | Labels | Refs hechos |
|---|---|---|---|---|
| main.tex | — (driver) | 87 | 0 | 0 |
| capitulos/00_preliminares.tex | \chapter*{Resumen} + Nomenclatura | 27 | 0 | 0 |
| capitulos/01_introduccion.tex | {Introducción} L5 `ch:introduccion` | 38 | 4 | 0 |
| capitulos/02_marco_teorico.tex | {Marco Teórico} L5 `ch:marco-teorico` | 127 | 19 | 0 |
| capitulos/03_fase1_nucleo.tex | {Fase 1: Núcleo…} L5 `ch:fase1` | 147 | 13 | 1 |
| capitulos/04_fase2_hibrido.tex | {Fase 2: Neuronas Spiking…} L5 `ch:fase2` | 91 | 9 | 2 |
| capitulos/05_fase3_plasticidad.tex | {Fase 3: Plasticidad Sináptica} L5 `ch:fase3` | 105 | 10 | 2 |
| capitulos/06_fase4_crossbar.tex | {Fase 4: Arquitecturas Matriciales…} L5 `ch:fase4` | 290 | 22 | 14 |
| capitulos/07_discusion.tex | {Discusión…y Limitaciones} L1 `ch:discusion_limitaciones` | 16 | 1 | 0 |
| capitulos/08_conclusiones.tex | {Conclusiones} L1 `ch:conclusiones` | 23 | 3 | 0 |
| capitulos/09_marco_practico.tex | {Marco Práctico} L1 — **sin `\label{ch:…}`** | 686 | 23 | 23 |
| apendices/A_codigo.tex | {Estructura del Código Fuente…} `ch:apendice_codigo` | 161 | 7 | 0 |
| apendices/B_datos.tex | {Conjuntos de Datos…} `ch:apendice_datos` | 122 | 7 | 0 |
| capitulo_tesis.tex | — **HUÉRFANO** (no está en main.tex) | 204 | 13 | 12 |
| fase3_plasticidad.tex | — **HUÉRFANO** (no está en main.tex) | 63 | 2 | 0 |

Totales: **133 labels crudos = 123 únicos** (10 duplicados: `fig:exp01`–`fig:exp10` definidos en 06 Y capitulo_tesis) · **54 refs crudos = 44 únicos**.

## 2. Secciones por capítulo

### 00_preliminares.tex
(sin `\section`; solo `\chapter*` Resumen y Nomenclatura)

### 01_introduccion.tex
- L8: \section{Contexto y motivación}
- L15: \section{Planteamiento del problema}
- L25: \section{Objetivos}
- L28: \subsection{Objetivo general}
- L31: \subsection{Objetivos específicos}

### 02_marco_teorico.tex
- L8: \section{Fundamentos del Memristor}
- L20: \subsection{Modelo matemático de Strukov (2008)}
- L39: \subsection{Modelo de Yakopcic y Prezioso (2011, 2014, 2015)}
- L65: \subsection{Memristores volátiles para nodos sensoriales}
- L77: \section{Modelo de Neurona Spiking LIF}
- L98: \section{Plasticidad Sináptica}
- L101: \subsection{Plasticidad dependiente del tiempo de disparo (STDP)}
- L115: \subsection{Plasticidad modulada por recompensa (R-STDP)}

### 03_fase1_nucleo.tex
- L11: \section{Etapa 1.1: Modelo Base y Curvas Características}
- L14: \subsection{Implementación Cualitativa y Fundamentos Físicos}
- L42: \subsection{Validación Cuantitativa de Curvas $I-V$}
- L104: \section{Etapa 1.2: Variabilidad Estocástica}
- L107: \subsection{Implementación Cualitativa de No-Idealidades}
- L115: \subsection{Validación Cuantitativa Estadística}

### 04_fase2_hibrido.tex
- L11: \section{Etapa 2.1: Neurona LIF y Acoplamiento Híbrido}
- L14: \subsection{Implementación Cualitativa y Teoría Física}
- L48: \subsection{Validación Cuantitativa Computacional}

### 05_fase3_plasticidad.tex
- L11: \section{Etapa 3.1: LTP/LTD y STDP}
- L14: \subsection{Implementación Cualitativa (LTP/LTD)}
- L39: \subsection{Implementación Cualitativa (STDP)}
- L62: \subsection{Validación Cuantitativa de Aprendizaje (Frente a Datos Físicos)}


### 06_fase4_crossbar.tex
- L11: \section{Etapa 4.1: Matriz Crossbar e In-Memory}
- L25: \subsection{Implementación Cualitativa (Operación Analógica $1 \times 1$)}
- L61: \subsection{Validación Cuantitativa Computacional de Red Experimental (Prezioso 2015)}
- L111: \section{Etapa 4.2: Integración Neuromórfica… NeuroBot $24 \times 12$)}
- L116: \subsection{Arquitectura del Sistema ($24 \times 12$)}
- L119: \subsubsection{Codificador de Estados (Encoder Top-K)}
- L122: \subsubsection{Crossbar Memristivo y Solución Nodal (MNA)}
- L129: \subsubsection{Decodificador y Acción (Softmax e Inercia de Motor)}
- L136: \subsection{Regla de Aprendizaje R-STDP y Traza Volátil}
- L145: \subsection{Resultados Experimentales y Discusión en Control Robótico}
- L147: \subsubsection{Convergencia y Reproducibilidad Multi-Semilla}
- L159: \subsubsection{Estudio de Ablaciones}
- L175: \subsubsection{Heatmap de la Matriz Memristiva Consolidada ($X$)}
- L187: \section{Validación del Modelo Físico del Crossbar (Suite de 10 Experimentos)}
- L192–L273: \subsection{Exp.~1 a Exp.~10} (sneak / MNA / R_wire / VMM / half-select / retención / linealidad LTP-LTD / D2D / costo solver MNA / crosstalk)
- L282: \subsection{Síntesis de Conclusiones de Fase 4}

### 07_discusion.tex
- L4: \section{Discusión de Resultados}
- L7: \section{Limitaciones del Estudio}
- L15: \section{Reproducibilidad y Código Abierto}

### 08_conclusiones.tex
- L5: \section{Conclusiones Principales}
- L16: \section{Contribuciones del Trabajo}

### 09_marco_practico.tex (686 líneas — el más largo)
- L14: \section{Esquema general del proyecto}
- L50: \section{Herramientas} · L53 Hardware · L56 Software
- L60: \section{Desarrollo}
- L71: \subsection{Etapa 1 -- Modelado del dispositivo memristivo}
- L127: \subsection{Etapa 2 -- Neurona LIF y sistema híbrido}
- L186: \subsection{Etapa 3 -- Plasticidad sináptica}
- L241: \subsection{Etapa 4 -- Crossbar e in-memory computing}
- L309: \subsection{Etapa 5 -- Integración con agente de control autónomo}
- L318/L330/L340/L350: \subsubsection{Codificación sensorial / Propagación analógica / Decodificación estocástica / Aprendizaje R-STDP}
- L385: \subsection{Etapa 6 -- Extensión a percepción visual}
- L396/L406/L416/L426: \subsubsection{Codificación retinotópica / Multiplicación analógica / Competencia blanda / R-STDP supervisado}
- L473: \subsection{Etapa 7 -- Navegación autónoma con obstáculos}
- L487/L497/L507: \subsubsection{Sensor y codificación / Recompensa / Resultados}
- L567: \section{Resultados y discusión}
- L570: \subsection{Métricas comparativas entre tareas}
- L598: \subsection{Especialización del sustrato según la tarea}
- L612: \subsection{Ablaciones A/B/C/D}
- L666: \subsection{Limitaciones}
- L683: \section{Análisis de costos} ← **§3.5: el heading YA existe; verificar si el cuerpo está vacío**

### apendices/A_codigo.tex
- L8: \section{Árbol de directorios del paquete \texttt{neurolab}}
- L59: \section{Descripción de los módulos principales}
- L84: \section{Resumen de la suite de pruebas unitarias (Pytest)}
- L111: \section{Fragmentos de Código Centrales} (L116 Strukov / L136 base_device)

### apendices/B_datos.tex
- L7: \section{Conjuntos de datos de referencia (CSVs)}
- L33: \section{Catálogo de figuras generadas} ← **mapeo oficial fig → script (23 filas)**
- L72: \section{Tabla maestra de las 27 validaciones cuantitativas} ← **V01–V27: paper, script, métrica, valor**

### capitulo_tesis.tex (HUÉRFANO — no incluido en main.tex)
- L22 Introducción y Motivación · L30 Arquitectura 24×12 (L33/L36/L43 subs) · L50 R-STDP · L59 Resultados (L61/L73/L89 subs) · L100 Validación Crossbar (L104–L185: Exp.~1–10) · L194 Conclusiones.
- Es la primera versión del material hoy vivo en `06_fase4_crossbar.tex`.

### fase3_plasticidad.tex (HUÉRFANO — no incluido en main.tex)

## 3. Cruces de labels

- **Refs SIN label (ROTO — crítico): NINGUNA** ✅ — los 44 refs únicos resuelven.
- **Labels duplicados (peligro latente):** `fig:exp01`…`fig:exp10` definidos tanto en `06_fase4_crossbar.tex` como en `capitulo_tesis.tex`. Hoy no rompe porque `capitulo_tesis.tex` no se compila; si alguna vez se incluye, el build produce warnings de label duplicado.
- **Labels SIN REF (79):** no son errores (labels de capítulo/tabla sin `\ref` son normales), pero marcan qué no se usa:

```
ch:apendice_codigo, ch:apendice_datos, ch:conclusiones, ch:discusion_limitaciones,
ch:fase1, ch:fase2, ch:fase3, ch:fase4, ch:introduccion, ch:marco-teorico,
eq:hfo2_dinamica, eq:i_inyectada, eq:lif_hibrida, eq:lif-diffeq, eq:lif-spike,
eq:memristor-fundamental, eq:resistencia_lineal, eq:rstdp-rule, eq:stdp-ajuste-jo,
eq:stdp-window, eq:strukov_dinamica, eq:strukov-dxdt, eq:strukov-rx, eq:volatile-dxdt,
eq:yakopcic-dxdt, eq:yakopcic-iv,
fig:07_r_x, fig:08_rs_x, fig:09_dt_extremos, fig:11_ruido, fig:12_estadistica,
fig:18_stdp_temporal, fig:20_vi_ltp_ltd, fig:22_prezioso_v2, fig:exp06,
fig:prezioso_s3a, fig:prezioso_s3bc, fig:val-d2d-c2c, fig:val-ltp-ltd, fig:val-prezioso,
fig:val-stdp, fig:val-stdp-spikes, fig:val-strukov, fig:val-wang2025,
sec:ablaciones_34, sec:arbol_codigo, sec:catalogo_figuras, sec:conclusiones_principales,
sec:contribuciones, sec:datos_csv, sec:descripcion_modulos, sec:etapa1_1, sec:etapa1_2,
sec:etapa2_1, sec:etapa3_1, sec:etapa4_1, sec:etapa4_2, sec:fase3_plasticidad,
sec:fragmentos_codigo, sec:intro-contexto, sec:intro-objetivos, sec:intro-problema,
sec:marco-lif, sec:marco-memristor, sec:marco-plasticidad, sec:suite_10_exp,
sec:suite_tests, sec:tabla_27_validaciones,
subsec:marco-rstdp, subsec:marco-stdp, subsec:marco-strukov, subsec:marco-volatile,
subsec:marco-yakopcic,
tab:27_validaciones, tab:catalogo_figuras, tab:csv_datasets, tab:modulos_neurolab,
tab:resumen_tests, tab:stdp_trilogia
```

- **Refs por archivo:** `09_marco_practico` 23 · `06_fase4_crossbar` 14 · `capitulo_tesis` 12 (huérfano) · `03` 1 · `04` 2 · `05` 2 · resto 0.
- **Nota:** `fig:exp06` está label-eada e incluida (L241/L243) pero **nunca se referencia con `\ref`** — la figura se imprime sin número citable.

## 4. Bibliografía

- **Total entradas en `references.bib`: 12**

| Key | Tipo | Año | Autor (1ro) | Título |
|---|---|---|---|---|
| strukov2008 | article | 2008 | Strukov, D.B. | The missing memristor found |
| prezioso2014 | article | 2014 | Prezioso, M. | Training and operation of an integrated neuromorphic network… |
| prezioso2015 | article | 2015 | Prezioso, M. | Demonstration of a 100-synapse metal-oxide memristive crossbar array |
| yakopcic2011 | article | 2011 | Yakopcic, C. | A Memristor Device Model |
| biolek2009 | article | 2009 | Biolek, D. | SPICE Model of Memristor with Nonlinear Dopant Drift |
| joglekar2009 | article | 2009 | Joglekar, Y.N. | The elusive memristor: signatures in basic electrical circuits |
| wang2025 | article | 2025 | Wang, X. | Memristor-Based Spiking Neuromorphic Systems… |
| jo2010 | article | 2010 | Jo, S.H. | Nanoscale memristor device as synapse in neuromorphic systems |
| yang2017 | article | 2017 | Yang, R. | Synaptic plasticity and dynamic reasoning in a volatile oxide memristor |
| gerstner2002 | book | 2002 | Gerstner, W. | Spiking Neuron Models |
| bi1998 | article | 1998 | Bi, G.-q. | Synaptic modifications in cultured hippocampal neurons |
| likharev2013 | article | 2003* | Likharev, K.K. | CrossNets: Hybrid nanodevice/CMOS neuromorphic networks |

\* `likharev2013` tiene el key con 2013 pero el campo `year = 2003` — inconsistencia menor a corregir.

### Papers SIN CITAR (4 de 12)
`biolek2009`, `gerstner2002`, `joglekar2009`, `likharev2013`

### Papers por capítulo

| Capítulo | Cites |
|---|---|
| 00_preliminares | — |
| 01_introduccion | — |
| 02_marco_teorico | bi1998, jo2010, prezioso2014, prezioso2015, strukov2008, wang2025, yang2017 |
| 03_fase1_nucleo | strukov2008, yakopcic2011 |
| 04_fase2_hibrido | wang2025 |
| 05_fase3_plasticidad | jo2010 |
| 06_fase4_crossbar | prezioso2015 |
| 07_discusion | — |
| 08_conclusiones | — |
| 09_marco_practico | — (**0 cites en 686 líneas**) |
| apendice B_datos | strukov2008, prezioso2014, prezioso2015, wang2025, jo2010 |
| apendice A_codigo | — |


## 5. Figuras — inventario completo

**Totales en disco: 183 archivos de imagen** — `docs/latex/figuras` 59 · `docs/latex/experiments/figures` 10 · `robot_sim/figuras` 16 · `robot_sim/figures` 5 · `neuromorphic_lab/figuras` 3 · `neuromorphic_lab/graficas` 6 · `neuromorphic_lab/validaciones/figuras` 33 · `validaciones_2x2` 15 · `validaciones_4x4` 20 · `outputs/figuras_paper` 13 · `outputs/figuras_verificacion` 3.
**Referenciadas en `.tex`: 59 rutas únicas** · **Huérfanas: 7** · **Faltantes: 0** ✅

### 5a. docs/latex/figuras (59 archivos)

| Nombre | KB | Referenciada en (archivo:Línea) | Script generador |
|---|---|---|---|
| 23a_prezioso_s3a.png | 153.2 | 06:74 | validaciones/23_prezioso_s3.py |
| 23b_prezioso_s3b.png | 116.1 | 06:85 | validaciones/23_prezioso_s3.py |
| 23c_prezioso_s3c.png | 135.3 | 06:91 | validaciones/23_prezioso_s3.py |
| agente_arquitectura.png | 88.9 | 09:369 | robot_sim/generate_agente_arquitectura.py |
| cartpole_learning.png | 265.9 | 09:379 | = curva_aprendizaje_24x12 (hash) → generate_thesis_figures_24x12.py |
| comparativa_ablaciones_24x12.pdf | 29.7 | 06:170 · cap_tesis:84 | generate_thesis_figures_24x12.py |
| comparativa_ablaciones_24x12.png | 214 | (no referenciado) | generate_thesis_figures_24x12.py |
| curva_aprendizaje_24x12.pdf | 31.3 | 06:152 · cap_tesis:66 | generate_thesis_figures_24x12.py |
| curva_aprendizaje_24x12.png | 265.9 | (no referenciado) | generate_thesis_figures_24x12.py |
| esquema_general.png | 21.6 | 09:34 | = fig2_mna_vs_ideal (hash) → generate_figures.py ⚠️ |
| fig_01_strukov.png | 889.1 | 03:64 | validaciones/01_strukov_2008.py (catálogo B) |
| fig_02_prezioso.png | 252.6 | 03:96 | validaciones/02_prezioso_2014.py |
| fig_03_lif.png | 360.2 | 04:69 | validaciones/03_lif_analitica.py |
| fig_05_d2d_c2c.png | 560.9 | 03:122 | validaciones/05_d2d_c2c.py |
| fig_06_hfo2_isi.png | 443.8 | 04:41 | validaciones/06_hfo2_isi.py |
| fig_07_r_x.png | 279.1 | 03:75 | validaciones/07_r_x_consistencia.py |
| fig_08_rs_x.png | 428.3 | 04:53 | validaciones/08_rs_de_x.py |
| fig_09_dt_extremos.png | 275.9 | 03:84 | validaciones/09_dt_extremos.py |
| fig_10_triangular.png | 574 | 03:33 | validaciones/10_triangular.py |
| fig_11_ruido.png | 279.5 | 03:133 | validaciones/11_ruido.py |
| fig_12_estadistica.png | 527.7 | 03:142 | validaciones/12_estadistica.py |
| fig_13_wang.png | 273.8 | 04:86 | validaciones/13_wang_2025.py |
| fig_14_ltp_ltd.png | 166 | 05:21 | validaciones/14_ltp_ltd.py |
| fig_14b_ltp_ltd_jo2010.png | 265.4 | 05:90 | validaciones/24_jo2010_white.py (probable) |
| fig_15_stdp.png | 206.5 | **(huérfana)** | validaciones/15_stdp.py |
| fig_15b_stdp_jo2010.png | 205.8 | 05:76 | validaciones/24_jo2010_white.py (probable) |
| fig_16_ciclo.png | 141.2 | 05:32 | validaciones/16_ltp_ltd_ciclo.py |
| fig_17_stdp_spikes.png | 254 | 05:46 | validaciones/17_stdp_spikes.py |
| fig_18_stdp_temporal.png | 263.1 | 05:55 | validaciones/18_stdp_temporal.py |
| fig_19_iv_hysteresis.png | 130 | **(huérfana)** | validaciones/19_iv_histeresis.py |
| fig_20_vi_ltp_ltd.png | 435 | 05:100 | validaciones/20_vi_ltp_ltd.py |
| fig_21a_crossbar_V.png | 146.8 | 06:34 | validaciones/21_crossbar_1x1.py |
| fig_21b_crossbar_G.png | 191.5 | 06:40 | validaciones/21_crossbar_1x1.py |
| fig_21c_crossbar_din.png | 300 | 06:46 | validaciones/21_crossbar_1x1.py |
| fig_21d_crossbar_stdp.png | 162.9 | 06:52 | validaciones/21_crossbar_1x1.py |
| fig_22_prezioso_v2.png | 125.8 | 06:102 | validaciones/22_prezioso_v2.py |
| fig_23_s3a.png | 103.4 | **(huérfana)** | copia obsoleta (el usado: 23a_prezioso_s3a) |

| fig2_mna_vs_ideal.png | 21.6 | **(huérfana)** | robot_sim/generate_figures.py:54 |
| fig3_v2_vs_v3.png | 51.3 | **(huérfana)** | robot_sim/generate_figures.py:70 |
| heatmap_matriz_X_24x12.pdf | 48.7 | 06:180 · cap_tesis:94 | generate_thesis_figures_24x12.py |
| heatmap_matriz_X_24x12.png | 168.7 | (no referenciado) | generate_thesis_figures_24x12.py |
| hfo2_lif_isi_decreciente.pdf | 77.3 | **(huérfana)** | validaciones/06_hfo2_isi.py:262 |
| ir_drops.png | 181.4 | 09:303 | sin generador .py confirmado ⚠️ |
| lif_analitica.png | 360.2 | 09:154 | = fig_03_lif (hash) → 03_lif_analitica.py |
| ltp_ltd.png | 166 | 09:223 | = fig_14_ltp_ltd (hash) → 14_ltp_ltd.py |
| mna_vs_ideal.png | 235.1 | 09:286 | sin generador .py confirmado ⚠️ |
| obstacle_arquitectura.png | 80.9 | 09:559 | generate_obstacle_arquitectura.py |
| obstacle_learning.png | 125.2 | 09:550 | generate_obstacle_figures.py:99 |
| obstacle_mapa.png | 14.1 | 09:531 | generate_obstacle_figures.py:59 |
| obstacle_trayectoria.png | 31.2 | 09:541 | generate_obstacle_figures.py:76 |
| probe_100ep_curve.png | 136.7 | **(huérfana)** | robot_sim/plot_probe_curve.py:63 |
| sistema_hibrido.png | 443.8 | 09:172 | = fig_06_hfo2_isi (hash) → 06_hfo2_isi.py |
| sneak_ratio_vs_N.png | 64.3 | 09:277 | sin generador .py confirmado ⚠️ |
| stdp.png | 206.5 | 09:233 | = fig_15_stdp (hash) → 15_stdp.py |
| validacion_memristor.png | 889.1 | 09:98 | = fig_01_strukov (hash) → 01_strukov_2008.py |
| variabilidad_d2d_c2c.png | 560.9 | 09:119 | = fig_05_d2d_c2c (hash) → 05_d2d_c2c.py |
| vision_arquitectura.png | 93.4 | 09:448 | generate_vision_arquitectura.py |
| vision_learning.png | 60 | 09:458 | run_vision.py:156 (seed0) |
| vision_seeds.png | 103.5 | 09:467 | generate_vision_seeds.py:37 |

### 5b. docs/latex/experiments/figures (10 PNG + 10 JSON)

| Nombre | KB | Referenciada en | Script generador |
|---|---|---|---|
| exp01_sneak_vs_N.png | 71.9 | 06:196 · cap_tesis:108 | robot_sim/experiments/test_crossbar_suite.py:99 |
| exp02_mna_vs_ideal.png | 180.2 | 06:205 · cap_tesis:117 | test_crossbar_suite.py:146 |
| exp03_rwire_effect.png | 92.3 | 06:214 · cap_tesis:126 | test_crossbar_suite.py |
| exp04_vmm_precision.png | 59.5 | 06:223 · cap_tesis:135 | test_crossbar_suite.py |
| exp05_v2_vs_v3.png | 195.5 | 06:232 · cap_tesis:144 | test_crossbar_suite.py |
| exp06_retention.png | 51.4 | 06:241 · cap_tesis:153 | test_crossbar_suite.py |
| exp07_ltp_ltd.png | 102.2 | 06:250 · cap_tesis:162 | test_crossbar_suite.py |
| exp08_d2d_reproducibility.png | 254.2 | 06:259 · cap_tesis:171 | test_crossbar_suite.py |
| exp09_computational_cost.png | 71.3 | 06:268 · cap_tesis:180 | test_crossbar_suite.py |
| exp10_crosstalk.png | 87 | 06:277 · cap_tesis:189 | test_crossbar_suite.py:594 |

Cada PNG tiene su `.json` hermano (data cruda). La suite escribe en `robot_sim/experiments/figures/` (FIG, L26) y hay copia en `docs/latex/experiments/figures/` — **dos ubicaciones que pueden desincronizarse**.

### 5c. Otros directorios (no referenciados por el .tex — material de soporte)

- **robot_sim/figuras (16):** curva_aprendizaje_24x12.png/.pdf, comparativa_ablaciones_24x12.png/.pdf, heatmap_matriz_X_24x12.png/.pdf (duplicados con docs/latex/figuras), fig1_rstdp_vs_frozen, fig2_mna_vs_ideal, fig3_v2_vs_v3, fig4_g_heatmap, fig5_eligibility, probe_100ep_curve, vision_dataset, vision_learning_seed0/1/2.
- **robot_sim/figures (5):** fig1_learning_curves, fig2_return_histograms, fig3_X_heatmap_runA, fig4_action_distribution, fig4_action_vs_angle (generados por plot_results.py).
- **neuromorphic_lab/figuras (3):** validacion_estadistica, validacion_ruido, validacion_triangular.
- **neuromorphic_lab/graficas (6):** lif_etapa210_FI_curve, lif_etapa28_spike, lif_etapa29_train, lif_rc_validation, lif_threshold_validation, validacion_neuromorfica_conceptual.
- **neuromorphic_lab/validaciones/figuras (33):** copia maestra de todas las validaciones 01–24 (validacion_strukov, validacion_prezioso_limpio, hfo2_lif_isi_decreciente.png/.pdf, validacion_crossbar_1x1_V/G/din/stdp, etc.) — de aquí salen las copias renombradas `fig_NN` de latex.
- **validaciones_2x2 (15) y validaciones_4x4 (20):** fig_01…fig_15/fig_20 de las suites 2×2 y 4×4 (generadas por generate_2x2_validation_figures.py / generate_4x4_validation_figures.py).
- **outputs/figuras_paper (13):** 01_strukov_iv … 12_pre_forming_paneles (dashboards de papers).
- **outputs/figuras_verificacion (3):** crossbar_demo, crossbar_ltp_ltd, crossbar_validacion_completa.


## 6. Figuras — problemas

### Huérfanas (7) — existen en `docs/latex/figuras` pero nadie las referencia
1. `fig_15_stdp.png` (206.5 KB) — el cap 05 usa `stdp.png` (idéntico por hash)
2. `fig_19_iv_hysteresis.png` (130 KB) — validación 19 existe pero no se incluyó
3. `fig_23_s3a.png` (103.4 KB) — obsoleto; el usado es `23a_prezioso_s3a.png`
4. `fig2_mna_vs_ideal.png` (21.6 KB) — copia de robot_sim/figuras
5. `fig3_v2_vs_v3.png` (51.3 KB) — copia de robot_sim/figuras
6. `hfo2_lif_isi_decreciente.pdf` (77.3 KB) — el PNG hermano se llama `sistema_hibrido.png`
7. `probe_100ep_curve.png` (136.7 KB) — figura de sondeo interno

### Faltantes (referenciadas sin archivo): **NINGUNA** ✅
(Las 59 rutas `\includegraphics` resuelven a archivo real.)

### Duplicadas entre `docs/latex/figuras` y otros directorios (15)
- vs `robot_sim/figuras` (9): comparativa_ablaciones_24x12.png/.pdf, curva_aprendizaje_24x12.png/.pdf, heatmap_matriz_X_24x12.png/.pdf, fig2_mna_vs_ideal.png, fig3_v2_vs_v3.png, probe_100ep_curve.png
- vs `neuromorphic_lab/validaciones/figuras` (6): 23a/23b/23c_prezioso_s3*.png, fig_14b_ltp_ltd_jo2010.png, fig_15b_stdp_jo2010.png, hfo2_lif_isi_decreciente.pdf

### Duplicadas POR HASH dentro de `docs/latex/figuras` (8 pares — mismo byte a byte)
| Par (nombre usado en tesis = nombre fuente) | KB |
|---|---|
| cartpole_learning.png = curva_aprendizaje_24x12.png | 265.9 |
| esquema_general.png = fig2_mna_vs_ideal.png | 21.6 |
| validacion_memristor.png = fig_01_strukov.png | 889.1 |
| lif_analitica.png = fig_03_lif.png | 360.2 |
| variabilidad_d2d_c2c.png = fig_05_d2d_c2c.png | 560.9 |
| sistema_hibrido.png = fig_06_hfo2_isi.png | 443.8 |
| ltp_ltd.png = fig_14_ltp_ltd.png | 166 |
| stdp.png = fig_15_stdp.png | 206.5 |

**Riesgo:** el cap 09 depende de 8 aliases sin contrato de generación — si se re-corre una validación, los `fig_NN` se actualizan pero los aliases (validacion_memristor, sistema_hibrido…) quedan desincronizados salvo copia manual.

## 7. Scripts ejecutables

- **Con `if __name__ == "__main__"`:** 94 scripts (72 activos + 22 en `validaciones_BACKUP_2026/`)
  - `neuromorphic_lab/validaciones/`: los 24 (01–24) + run_all_validaciones
  - `experiments/fase4_1_crossbar_1x1/`: run_all_tests + 20 test_XX
  - `neurolab/validation/`: jo2010_endurance, jo2010_ltp_ltd, prezioso2015_s3
  - `neuromorphic_lab`: run_app, test_strukov, demos (3), generate_2x2/4x4 figures, gui/app, scratch tests (4), tests/crossbar
  - `robot_sim`: eval_yamil, test_crossbar_suite, generate_obstacle_figures, generate_thesis_figures_24x12, obstacle_train, run_ablations_24x12, run_ablations_real_physics, run_pendulum, run_vision, verify_cartpole_2000
- **Con `argparse` (CLI):** 6 → run_all_validaciones.py, test_crossbar_suite.py, obstacle_train.py, run_ablations_real_physics.py, run_pendulum.py, run_vision.py
- **Con `savefig` (generan figuras):** 80 scripts (24 validaciones + 22 BACKUP + 20 tests fase4 + 12 robot_sim generate/plot + demo/GUI misc)
- **Con `json.dump`/`write_text` (generan datos):** 18 → test_crossbar_suite, obstacle_train, run_ablations_24x12, run_ablations_real_physics, run_pendulum, run_robot, run_vision, verify_cartpole_2000, probe_contract, crossbar_brain, logger, plot (ver §9)


## 8. Mapeo scripts → figuras

| Script | Figuras que produce | Destino |
|---|---|---|
| neuromorphic_lab/validaciones/01_strukov_2008.py | validacion_strukov.png → (alias latex: validacion_memristor.png, fig_01_strukov.png) | validaciones/figuras/ + datos/*.csv |
| validaciones/02_prezioso_2014.py | validacion_prezioso_limpio.png → fig_02_prezioso.png | idem |
| validaciones/03_lif_analitica.py | lif_rc_validation.png → fig_03_lif.png + lif_analitica.png | idem |
| validaciones/04_convergencia.py | validacion_convergencia.png | idem (fig_04 NO existe en latex) |
| validaciones/05_d2d_c2c.py | validacion_d2d_c2c.png → fig_05_d2d_c2c + variabilidad_d2d_c2c | idem |
| validaciones/06_hfo2_isi.py | hfo2_lif_isi_decreciente.png/.pdf → fig_06_hfo2_isi + sistema_hibrido | idem |
| validaciones/07..12 | validacion_r_x_consistencia, validacion_rs? (08), validacion_dt_extremos, validacion_triangular, validacion_ruido, validacion_estadistica → fig_07..fig_12 | idem |
| validaciones/13_wang_2025.py | validacion_wang2025.png → fig_13_wang | idem |
| validaciones/14..20 | validacion_ltp_ltd, validacion_stdp, validacion_ltp_ltd_ciclo, validacion_stdp_spikes, validacion_stdp_temporal, validacion_iv_ltp_ltd, validacion_vi_ltp_ltd → fig_14..fig_20 | idem |
| validaciones/21_crossbar_1x1.py | validacion_crossbar_1x1_V/G/din/stdp.png → fig_21a..d | idem |
| validaciones/22_prezioso_v2.py | validacion_prezioso_v2.png → fig_22 | idem |
| validaciones/23_prezioso_s3.py | 23a/23b/23c_prezioso_s3*.png | idem |
| validaciones/24_jo2010_white.py | 2 versiones blancas (probable fig_14b/15b) | idem |
| robot_sim/experiments/test_crossbar_suite.py | exp01..exp10 .png + .json | robot_sim/experiments/figures/ (copiado a docs/latex/experiments/figures/) |
| robot_sim/generate_thesis_figures_24x12.py | curva_aprendizaje_24x12.png/.pdf, comparativa_ablaciones_24x12.png/.pdf, heatmap_matriz_X_24x12.png/.pdf (usa `Yamil_1.npy`/`Base 24x12`) | robot_sim/figuras/ (copiado a docs/latex/figuras/) |
| robot_sim/generate_agente_arquitectura.py | agente_arquitectura.png | docs/latex/figuras/ |
| robot_sim/generate_vision_arquitectura.py | vision_arquitectura.png | docs/latex/figuras/ |
| robot_sim/generate_vision_seeds.py | vision_seeds.png (lee run_vision_seed*.json) | docs/latex/figuras/ |
| robot_sim/run_vision.py | vision_learning.png + run_vision_seed{0,1,2}.json | docs/latex/figuras/ + outputs/ |
| robot_sim/generate_obstacle_figures.py | obstacle_mapa/trayectoria/learning.png + obstacle_seed*_paths.json | docs/latex/figuras/ + outputs/ |
| robot_sim/generate_obstacle_arquitectura.py | obstacle_arquitectura.png | docs/latex/figuras/ |
| robot_sim/generate_figures.py | fig1_rstdp_vs_frozen, fig2_mna_vs_ideal, fig3_v2_vs_v3, fig4_g_heatmap, fig5_eligibility | robot_sim/figuras/ |
| robot_sim/plot_results.py | fig1_learning_curves, fig2_return_histograms, fig3_X_heatmap_runA, fig4_action_vs_angle (lee run_A_base.json etc.) | robot_sim/figures/ |
| robot_sim/plot_probe_curve.py | probe_100ep_curve.png | docs/latex/figuras/ |
| experiments/scripts/generate_2x2_validation_figures.py | 15 figuras 2×2 | neuromorphic_lab/validaciones_2x2/ |
| experiments/scripts/generate_4x4_validation_figures.py | 20 figuras 4×4 | neuromorphic_lab/validaciones_4x4/ |
| experiments/demos/demo_crossbar_completo.py | crossbar_demo.png, crossbar_ltp_ltd.png | outputs/figuras_verificacion/ |
| **SIN GENERADOR** | `sneak_ratio_vs_N.png`, `mna_vs_ideal.png`, `ir_drops.png` (las 3 figuras del Etapa 4 del cap 09) | ⚠️ ver §11 |


## 9. JSONs de salida

**Totales: 44 en `robot_sim/outputs/`** (todo ignorado por git ✅) · 10+10 en `experiments/figures/` (PNG+JSON de la suite) · 2 en `outputs/output_modular/` (heredados, 2026-06-15).

### robot_sim/outputs/ (44 archivos)

| JSON | KB | Última mod. | Script (writer) |
|---|---|---|---|
| run_24x12_A_base_seed0..4.json (5) | 18–22 | 2026-10-08 16:27–16:53 | run_robot.py (orquestador; leer por diag_user_check.py) |
| run_24x12_B_seed0..2.json (3) | 18.9–20.8 | 2026-10-08 00:02–17:00 | run_robot.py |
| run_24x12_C_seed0.json | 22.4 | 2026-10-08 17:00 | run_robot.py |
| run_24x12_D_seed0..2.json (3) | 22.3–22.4 | 2026-10-08 00:06–17:00 | run_robot.py |
| run_24x12_B_tempfixed/C_noaccum/D_hebbian_seed0 (3) | 9–11.3 | 2026-10-07 22:34–22:35 | run_robot.py (variantes de ablación) |
| run_A_base / run_B_temp01 / run_C_noaccum / run_D_hebbian.json (4) | 1950–5291 | 2026-10-01 23:50–23:51 | run_robot.py (corridas largas; leídas por plot_results.py) |
| run_vision_seed0..2.json (3) | ~400 | 2026-10-08 13:12–13:13 | run_vision.py:216 |
| run_pendulum_seed0..2.json (3) | 11.9–213.4 | 2026-10-08 12:43–12:54 | run_pendulum.py:200 |
| run_obstacle_seed0..2.json (3) | 90.6–100.6 | 2026-10-08 15:52 | obstacle_train.py |
| obstacle_seed0..2_paths.json (3) | 2.1–2.7 | 2026-10-08 15:52 | generate_obstacle_figures.py:30 |
| ablations_real_physics_summary.json | 1.5 | 2026-10-08 17:00 | run_ablations_real_physics.py |
| Crossbar_First.json | 460.5 | 2026-10-08 12:29 | (sin writer .py — estado de GUI/crossbar) |
| last_weights.json | 467.5 | 2026-10-08 13:10 | run_robot.py:98 |
| Yamil.json | 463.7 | 2026-10-08 14:08 | (inferido eval_yamil.py — sin match literal) |
| Yamil_eval_20ep.json | 0.9 | 2026-10-07 20:04 | eval_yamil.py (probable) |
| verify_cartpole_2000.json | 221.1 | 2026-10-08 14:09 | verify_cartpole_2000.py:86 |
| verify_cartpole_500.json | 101 | 2026-10-01 17:49 | verify_cartpole_500 (legacy/robot_sim) |
| verify_bandit.json | 334.4 | 2026-10-01 13:11 | legacy/verify_bandit.py |
| verify_contextual.json | 413.1 | 2026-10-01 13:19 | legacy/verify_contextual.py |
| verify_rstdp.json | 787.9 | 2026-09-30 16:16 | legacy/verify_rstdp.py |
| verify_agent_log.json | 2.3 | 2026-09-29 15:29 | legacy/verify_agent.py |
| crossbar_state_test.json / crossbar_state_trained.json (2) | 1.5–2.1 | 2026-09-29 | generate_figures.py:25 / crossbar_brain.py |

### experiments/figures (10 JSON)
exp01_sneak_vs_N … exp10_crosstalk.json — todos de `test_crossbar_suite.py` (`save_json`, L100/147/595…). Existe copia en `robot_sim/experiments/figures/` y en `docs/latex/experiments/figures/` (**triple ubicación**).

### outputs/raíz (2 JSON, heredados)
- `output_modular/fig2b_complete.json` (5.1 MB, 2026-06-15)
- `output_modular/fig2b_unity/metadata.json` (4.6 MB, 2026-06-15)


## 10. Validaciones (fig_NN ↔ capítulo ↔ paper ↔ script)

Cruce incluido el capítulo (por línea del `\includegraphics` vs mapa de secciones del §2) y el paper (catálogo B_datos + cites del capítulo).

| fig_NN | Capítulo | Sección | Paper validado | Script |
|---|---|---|---|---|
| fig_01_strukov | 03 (L64) | Etapa 1.1 · Validación Cuantitativa I-V | strukov2008 | validaciones/01_strukov_2008.py |
| fig_02_prezioso | 03 (L96) | Etapa 1.1 · Validación Cuantitativa I-V | prezioso2014 | validaciones/02_prezioso_2014.py |
| fig_05_d2d_c2c | 03 (L122) | Etapa 1.2 · Validación Cuantitativa Estadística | (variabilidad, sin paper) | validaciones/05_d2d_c2c.py |
| fig_07_r_x | 03 (L75) | Etapa 1.1 · Validación Cuantitativa I-V | strukov2008 | validaciones/07_r_x_consistencia.py |
| fig_09_dt_extremos | 03 (L84) | Etapa 1.1 · Validación Cuantitativa I-V | strukov2008 | validaciones/09_dt_extremos.py |
| fig_10_triangular | 03 (L33) | Etapa 1.1 · Implementación Cualitativa | strukov2008 | validaciones/10_triangular.py |
| fig_11_ruido | 03 (L133) | Etapa 1.2 · Validación Cuantitativa Estadística | (ruido, sin paper) | validaciones/11_ruido.py |
| fig_12_estadistica | 03 (L142) | Etapa 1.2 · Validación Cuantitativa Estadística | (estadística, sin paper) | validaciones/12_estadistica.py |
| fig_06_hfo2_isi | 04 (L41) | Etapa 2.1 · Implementación Cualitativa | wang2025 | validaciones/06_hfo2_isi.py |
| fig_08_rs_x | 04 (L53) | Etapa 2.1 · Validación Cuantitativa Computacional | wang2025 | validaciones/08_rs_de_x.py |
| fig_03_lif | 04 (L69) | Etapa 2.1 · Validación Cuantitativa Computacional | (analítico, sin paper) | validaciones/03_lif_analitica.py |
| fig_13_wang | 04 (L86) | Etapa 2.1 · Validación Cuantitativa Computacional | wang2025 | validaciones/13_wang_2025.py |
| fig_14_ltp_ltd | 05 (L21) | Etapa 3.1 · Implementación LTP/LTD | jo2010 | validaciones/14_ltp_ltd.py |
| fig_16_ciclo | 05 (L32) | Etapa 3.1 · Implementación LTP/LTD | jo2010 | validaciones/16_ltp_ltd_ciclo.py |
| fig_17_stdp_spikes | 05 (L46) | Etapa 3.1 · Implementación STDP | jo2010 | validaciones/17_stdp_spikes.py |
| fig_18_stdp_temporal | 05 (L55) | Etapa 3.1 · Implementación STDP | jo2010 | validaciones/18_stdp_temporal.py |
| fig_15b_stdp_jo2010 | 05 (L76) | Etapa 3.1 · Validación Cuantitativa | jo2010 | validaciones/24_jo2010_white.py |
| fig_14b_ltp_ltd_jo2010 | 05 (L90) | Etapa 3.1 · Validación Cuantitativa | jo2010 | validaciones/24_jo2010_white.py |
| fig_20_vi_ltp_ltd | 05 (L100) | Etapa 3.1 · Validación Cuantitativa | jo2010 | validaciones/20_vi_ltp_ltd.py |
| fig_21a–d_crossbar | 06 (L34–52) | Etapa 4.1 · Operación Analógica 1×1 | prezioso2015 | validaciones/21_crossbar_1x1.py |
| 23a/23b/23c_prezioso_s3* | 06 (L74/85/91) | Etapa 4.1 · Validación Prezioso 2015 | prezioso2015 | validaciones/23_prezioso_s3.py |
| fig_22_prezioso_v2 | 06 (L102) | Etapa 4.1 · Validación Prezioso 2015 | prezioso2014/2015 | validaciones/22_prezioso_v2.py |
| curva_aprendizaje_24x12 | 06 (L152) | Etapa 4.2 · Resultados (convergencia) | (propio) | generate_thesis_figures_24x12.py |
| comparativa_ablaciones_24x12 | 06 (L170) | Etapa 4.2 · Ablaciones | (propio) | generate_thesis_figures_24x12.py |
| heatmap_matriz_X_24x12 | 06 (L180) | Etapa 4.2 · Heatmap | (propio) | generate_thesis_figures_24x12.py |
| exp01_sneak_vs_N | 06 (L196) | Suite 10 Exp · Exp.~1 | strukov2008 (sneak) | test_crossbar_suite.py |
| exp02_mna_vs_ideal | 06 (L205) | Suite 10 Exp · Exp.~2 | (MNA exacto) | test_crossbar_suite.py |
| exp03_rwire_effect | 06 (L214) | Suite 10 Exp · Exp.~3 | (física wires) | test_crossbar_suite.py |
| exp04_vmm_precision | 06 (L223) | Suite 10 Exp · Exp.~4 | prezioso2015 | test_crossbar_suite.py |
| exp05_v2_vs_v3 | 06 (L232) | Suite 10 Exp · Exp.~5 | (Prezioso S5) | test_crossbar_suite.py |
| exp06_retention | 06 (L241) | Suite 10 Exp · Exp.~6 ⚠️ sin `\ref` | (no volatilidad) | test_crossbar_suite.py |
| exp07_ltp_ltd | 06 (L250) | Suite 10 Exp · Exp.~7 | jo2010 | test_crossbar_suite.py |
| exp08_d2d_reproducibility | 06 (L259) | Suite 10 Exp · Exp.~8 | (estadística) | test_crossbar_suite.py |
| exp09_computational_cost | 06 (L268) | Suite 10 Exp · Exp.~9 | (solver MNA) | test_crossbar_suite.py |
| exp10_crosstalk | 06 (L277) | Suite 10 Exp · Exp.~10 | prezioso2015 | test_crossbar_suite.py |
| validacion_memristor | 09 (L98) | Etapa 1 | strukov2008 | alias de fig_01 (hash) |
| variabilidad_d2d_c2c | 09 (L119) | Etapa 1 | (sin paper) | alias de fig_05 (hash) |
| lif_analitica | 09 (L154) | Etapa 2 | (analítico) | alias de fig_03 (hash) |
| sistema_hibrido | 09 (L172) | Etapa 2 | wang2025 | alias de fig_06 (hash) |
| ltp_ltd | 09 (L223) | Etapa 3 | jo2010 | alias de fig_14 (hash) |
| stdp | 09 (L233) | Etapa 3 | jo2010 | alias de fig_15 (hash) |
| sneak_ratio_vs_N | 09 (L277) | Etapa 4 | strukov2008 (sneak) | **sin generador** ⚠️ |
| mna_vs_ideal | 09 (L286) | Etapa 4 | (MNA exacto) | **sin generador** ⚠️ |
| ir_drops | 09 (L303) | Etapa 4 | (wires) | **sin generador** ⚠️ |

**Nota:** el Apéndice B (L72–121) trae la **tabla maestra V01–V27** con paper + script + métrica + valor de cada validación — es la fuente oficial de trazabilidad numérica.


## 11. Pendientes detectados

**Tesis (.tex):**
1. **§09 L683 "Análisis de costos"** — el heading existe; falta desarrollar el cuerpo de 3.5 (datos del crossbar/cap 06/lab).
2. **Cap 08 no referencia al 09** — 08 se escribió antes que el Marco Práctico; su alcance quedó desactualizado tras 3.3.6/3.3.7/3.4.
3. **`09_marco_practico.tex` sin `\label{ch:…}`** — no hay label de capítulo para `\ref` externo.
4. **`fig:exp06` sin `\ref`** — imagen incluida pero nunca citada numerada.
5. **0 cites en el cap 09** (686 líneas sin una sola citación).
6. **4 papers sin citar** en references.bib (biolek2009, gerstner2002, joglekar2009, likharev2013) + `likharev2013` con year=2003 (inconsistencia).
7. **Overfull/underfull boxes** en la compilación (22 overfull / 3 underfull en el último build conocido).
8. **2 `.tex` huérfanos**: `capitulo_tesis.tex` y `fase3_plasticidad.tex` (duplicados históricos de material ya integrado en 06/05).
9. **Labels duplicados latentes**: `fig:exp01`–`exp10` en 06 y capitulo_tesis.

**Figuras:**
10. **7 huérfanas** en docs/latex/figuras (§6).
11. **8 aliases por hash** — si se re-corre una validación, los aliases del cap 09 quedan desactualizados (falta automatizar `validacion_*` → alias/`fig_NN`).
12. **3 figuras sin generador** (`sneak_ratio_vs_N`, `mna_vs_ideal`, `ir_drops` — las tres del Etapa 4).
13. **15 duplicados cross-directorio** + copias exp en 2 ubicaciones (triple para exp: robot_sim/experiments/figures, docs/latex/experiments/figures) — candidatos a `git gc`/limpieza o política de canónica única.
14. **`fig_04_convergencia` no existe** — el catálogo B la lista pero no hay archivo ni referencia (menor).

**Código / repo:**
15. **`git gc`** pendiente (160 MiB loose objects en `.git`).
16. **`validaciones_BACKUP_2026/`** (22 scripts) y `legacy/`, `archive/` (~120 scripts) — evaluar mover a archive o eliminar.
17. **`Crossbar_First.json`** (460 KB) sin writer identificado — probable artefacto de GUI.
18. **READMEs / docs de robot_sim** — verificar cobertura (no evaluado en este barrido).
19. **Pendientes de Fase 6 previa**: ya resueltos (last_session.json untracked + logs ignorados, commit `fa4fe3f`).

## 12. Hallazgos inesperados

1. **`esquema_general.png` (figura de apertura del cap 09, L34) es byte-a-byte idéntica a `fig2_mna_vs_ideal.png`** — el "esquema general del proyecto" que abre el Marco Práctico está mostrando, en realidad, el gráfico MNA vs ideal. Es casi seguro un copy-paste malo de figura: **revisar visualmente antes de entregar**.
2. **8 figuras del cap 09 son aliases hash de fig_NN de validaciones** (`validacion_memristor`=`fig_01_strukov`, `sistema_hibrido`=`fig_06_hfo2_isi`, `stdp`=`fig_15_stdp`, etc.) — el capítulo 09 no tiene pipeline propio de figuras: reutiliza las de validación por copia manual. Ningún script codifica esa relación (grep de `Copy-Item`/`shutil.copy` no encuentra el renombrado a latex).
3. **El trabajo de validación vive en 3 capas con solapamiento parcial**: `validaciones/` (01–24, fuente) → `validaciones/figuras/` (33 PNGs canónicos) → `docs/latex/figuras/fig_NN*` (renombrados, 26). La capa de renombrado es **manual y no reproducible**; el Apéndice B documenta el mapeo pero no hay código que lo ejecute.
4. **`09` concentra el 52% de los refs del documento (23/44) y 0 cites** — es el capítulo más autosuficiente y menos anclado a literatura.
5. **`robot_sim/figuras` y `robot_sim/figures` (plural/singular) coexisten** con convenciones de nombres distintas (`fig1_*` REINFORCE vs `*_24x12`) — fácil de confundir al regenerar.
6. **`scratch/` tiene 18 scripts sin filtrar en el conteo raíz** y hay `validaciones_BACKUP_2026` completo (22 scripts) que aún corre con main().

---

**Errores de comando durante el barrido (regla 4):**
- Fase 1.1: parser error inicial (`/KB` → `/1KB`); 1.2/1.4 volvieron vacíos con regex escapada → relanzados con patrón simplificado ✅
- Fase 3.2/3.3: pipelines de listing volvieron vacíos → relanzados con `-ErrorAction SilentlyContinue` + formato simple ✅
- Fase 4/5: `savefig("literal")` casi no matchea (los scripts usan `out_png`/variables) → complementado con grep de líneas `savefig` crudas ✅
- `git status` tardó >300 s (persistent index); output parcial leído del log de Cline ✅
- Cruces de figuras: 3 iteraciones de normalización de prefijos (`\` vs `/`, base dir) hasta obtener el cruce correcto ✅

**Totales del inventario:** 15 `.tex` · 123 labels únicos · 44 refs únicos · 0 refs rotos · 12 papers (4 sin citar) · 183 imágenes · 59 rutas referenciadas · 7 huérfanas · 0 faltantes · 15+8 duplicados · 364 `.py` activos (+116 legacy/archive) · 94 ejecutables · 44 JSONs de salida.

- L1 \section{Fase 3: Plasticidad Sináptica Memristiva} · L14 V(t)/I(t)/R(t)/G(t) · L34 Trilogía STDP.
