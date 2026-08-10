# Salidas, datos y figuras

## Carpeta principal de resultados

La carpeta principal es:

```text
output_modular/
```

Contiene salidas generadas por simulaciones modulares y scripts por pasos.

## Tipos de archivos

### PNG

Las imagenes `.png` son figuras listas para revisar visualmente, incluir en informes o usar como base para paper.

Ejemplos:

- `fig2b_reproduced.png`
- `fig2c_reproduced.png`
- `dashboard_completo.png`
- `frequency_collapse_demonstration.png`
- `caracterizacion_estocastica.png`
- `lif_validation_dashboard.png`

### CSV

Los `.csv` contienen datos numericos. Sirven para:

- recalcular metricas
- rehacer graficas
- comparar contra otros modelos
- importar en Excel, Origin, MATLAB o Python

Ejemplos:

- `fig2b_data.csv`
- `crossbar_data.csv`
- `ltp_ltd_data.csv`
- `retention_data.csv`
- `stdp_data.csv`

### JSON

Los `.json` guardan datos estructurados. Son utiles para:

- exportar a Unity
- alimentar visualizadores
- preservar parametros junto a senales

Ejemplos:

- `fig2b_complete.json`
- `fig2b_unity/metadata.json`

### TXT

Archivos como `summary.txt` guardan resumenes legibles de una corrida.

## Paquete Unity

La carpeta:

```text
output_modular/fig2b_unity/
```

incluye:

- `metadata.json`
- `summary.txt`
- `time_series.csv`
- `iv_curve.csv`

Esto sugiere un flujo donde el simulador genera datos y Unity los consume para un gemelo digital o visualizacion interactiva.

## Figuras para paper

La carpeta:

```text
imagenes_para_el_paper/
```

contiene imagenes seleccionadas y posiblemente renombradas para narrativa academica.

Ejemplos:

- `01_strukov_iv.png`
- `02_strukov_freq.png`
- `03_lif_validation_dashboard.png`
- `06_aprendizaje_stdp.png`
- `09_fugas_sneak_path.png`
- `10_comparacion_strukov_estocastico.png`

Esta carpeta es util para armar documentos, presentaciones o articulos sin buscar entre todas las salidas generadas.

## Datos experimentales externos

El repositorio tambien contiene CSV con nombres asociados a figuras:

- `Time-Voltage.csv`
- `Time-Current.csv`
- `Time-WD.csv`
- `Vin vs Time_ Figure 3.csv`
- `Vc vs Time_ Figure 3.csv`
- `Vout vs Time_ Figure 3.csv`

Estos datos parecen provenir de digitalizacion o referencia experimental. Se usan para comparacion, calibracion o validacion.

## Como saber que archivo mirar

Si interesa el memristor Strukov:

- mirar `fig2b_reproduced.png`
- mirar `fig2c_reproduced.png`
- mirar `metodologia_strukov_valid.png`
- mirar `estado_temporal_fig2b.png`

Si interesa frecuencia:

- mirar `frequency_collapse_demonstration.png`

Si interesa variabilidad:

- mirar `caracterizacion_estocastica.png`
- mirar `ruido_fisico_c2c.png`

Si interesa LIF:

- mirar `lif_validation_dashboard.png`
- mirar `lif_fig1_integracion.png`
- mirar `lif_fig2_umbral.png`
- mirar `lif_fig3_disparo.png`
- mirar `lif_fig4_reset.png`
- mirar `lif_fig5_dashboard.png`

Si interesa plasticidad:

- mirar `paso14_ltp_ltd.png`
- mirar `paso15_curva_stdp.png`
- mirar `ltp_ltd_metrics.csv`
- mirar `stdp_metrics.csv`

Si interesa crossbar:

- mirar `paso17_red_crossbar.png`
- mirar `paso18_crossbar_escalable.png`
- mirar `paso19_caracterizacion_crossbar.png`
- mirar `crossbar_metrics.csv`

## Recomendacion para reportes

Cada figura en un informe deberia acompanarse de:

- script que la genero
- parametros principales
- fecha o version de corrida
- archivo de datos asociado
- metrica cuantitativa si aplica

Esto evita que una imagen quede como evidencia aislada sin trazabilidad.

## Nombres de archivos

El proyecto usa varios estilos de nombres:

- `pasoXX_*`: etapa academica o experimento secuencial.
- `fig2b_*`: reproduccion de figura del paper.
- `lif_*`: validacion de neurona.
- `*_metrics.csv`: metricas resumidas.
- `*_data.csv`: datos completos o series.

Mantener esta convencion ayudara a que futuras salidas sean faciles de ubicar.

## Limpieza de resultados

Hay muchos archivos generados. Antes de borrar cualquiera, verificar si:

- aparece citado en algun `.md`, informe o paper
- se usa como entrada de otra simulacion
- es evidencia de validacion
- esta incluido en `imagenes_para_el_paper`

En proyectos academicos, las salidas son parte de la trazabilidad, no solo basura generada.
