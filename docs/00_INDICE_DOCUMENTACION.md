# Documentacion completa del entorno de simulacion de memristores

Este directorio contiene una explicacion amplia del proyecto, pensada como material de estudio, mantenimiento y defensa tecnica. La idea es que no tengas que abrir todos los archivos Python para entender que representa cada parte del sistema.

## Ruta de lectura recomendada

1. [01_VISION_GENERAL.md](01_VISION_GENERAL.md): explica que problema resuelve el proyecto, que modelos incluye y como se organiza el repositorio.
2. [02_TEORIA_MEMRISTOR_STRUKOV.md](02_TEORIA_MEMRISTOR_STRUKOV.md): desarrolla la base fisica del modelo Strukov 2008, sus ecuaciones y su interpretacion.
3. [03_ARQUITECTURA_DEL_CODIGO.md](03_ARQUITECTURA_DEL_CODIGO.md): explica la estructura interna del paquete `memristor_simulator`.
4. [04_GUIA_DE_EJECUCION.md](04_GUIA_DE_EJECUCION.md): indica como instalar dependencias, ejecutar simulaciones y leer salidas.
5. [05_VALIDACION_Y_METRICAS.md](05_VALIDACION_Y_METRICAS.md): resume las metricas, la comparacion con literatura y la validacion I-V/LIF.
6. [06_MODELO_LIF_NEUROMORFICO.md](06_MODELO_LIF_NEUROMORFICO.md): explica la neurona TSM-LIF, su circuito equivalente y sus experimentos.
7. [07_CROSSBAR_PLASTICIDAD_Y_SNEAK_PATHS.md](07_CROSSBAR_PLASTICIDAD_Y_SNEAK_PATHS.md): documenta la parte neuromorfica, crossbar, plasticidad y caminos parasitos.
8. [08_SALIDAS_DATOS_Y_FIGURAS.md](08_SALIDAS_DATOS_Y_FIGURAS.md): explica la carpeta `output_modular`, CSV, JSON, PNG y figuras para paper.
9. [09_GLOSARIO_Y_FORMULAS.md](09_GLOSARIO_Y_FORMULAS.md): recopila terminos, variables, unidades y formulas clave.
10. [10_GUIA_PARA_EXTENDER_EL_PROYECTO.md](10_GUIA_PARA_EXTENDER_EL_PROYECTO.md): recomendaciones para agregar nuevas simulaciones, modelos y validaciones.

## Que cubre esta documentacion

- Modelo fisico de memristor TiO2 basado en Strukov et al. 2008.
- Implementacion numerica con Euler explicito y utilidades RK4.
- Funcion ventana para deriva no lineal y control de limites fisicos.
- Reproduccion de curvas I-V con lazo de histeresis estrangulado.
- Analisis por frecuencia y colapso de histeresis.
- Variabilidad ciclo-a-ciclo mediante distribuciones estadisticas.
- Neurona fisica tipo Leaky Integrate-and-Fire basada en TSM.
- Crossbar, sneak paths, plasticidad LTP/LTD y STDP.
- Flujo de exportacion de resultados para graficas, CSV, JSON y Unity.

## Nota importante sobre el estado del repositorio

El repositorio tiene varios archivos generados, resultados experimentales, scripts `paso*.py`, imagenes y CSV. Esta documentacion describe el estado visible del proyecto sin borrar ni reordenar esos artefactos. Si un archivo aparece como generado, no significa que sea inutil: en este proyecto muchas salidas son evidencia para validacion, tesis o figuras de paper.
