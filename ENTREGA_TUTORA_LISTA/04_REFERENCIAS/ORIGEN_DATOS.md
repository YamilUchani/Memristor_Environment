# Referencias y procedencia de los datos

## Bibliografía

`references.bib` contiene las referencias citadas por la tesis y los módulos
de validación. La tesis usa las claves BibTeX desde `01_TESIS/`.

## Datos de referencia incluidos

Los CSV y metadatos originales se conservan en
`../03_EVIDENCIAS/datos_referencia/` y junto a los validadores en
`../02_SOFTWARE/neuromorphic_lab/validaciones/datos/`.

- Strukov et al. (2008), figura 2b: `validacion_strukov.csv`; resultados y
  métricas recalculables: `validacion_strukov_metricas.csv`.
- Jo et al. (2010), ventana STDP atribuida a figura 3a:
  `jo2010_stdp_holdout.csv` y `jo2010_stdp_holdout_metrics.json`.
- Jo et al. (2010), protocolo acumulativo LTP/LTD atribuido a figura 2a:
  `jo2010_ltp_ltd_published_protocol.csv` y archivos
  `jo2010_ltp_ltd_train_validation_test.*`.
- HfO2/LIF, LIF-FI, Prezioso y otras curvas: los archivos respectivos
  incluyen el nombre de modelo o referencia. Cuando son datos sintéticos o
  una comparación solo cualitativa, no deben interpretarse como mediciones
  físicas ni como ajuste cuantitativo a una publicación.

La procedencia detallada, convención temporal, transformaciones y límites de
cada comparación están documentados en los scripts y metadatos JSON. Algunos
CSV digitalizados tienen atribución en el script, pero no incluyen metadatos
de digitalización completos dentro del propio archivo; la tesis lo declara
como limitación.
