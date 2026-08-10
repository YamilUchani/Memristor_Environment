# Guia para extender el proyecto

## Principio general

Antes de agregar codigo nuevo, decidir que tipo de cambio es:

- nueva fisica de dispositivo
- nuevo experimento
- nueva metrica
- nueva figura
- nueva exportacion
- nuevo script academico

Eso determina donde debe vivir el cambio.

## Si agregas un nuevo modelo fisico

Ubicacion recomendada:

```text
memristor_simulator/models/
```

Ejemplos:

- nuevo modelo de deriva
- memristor con otro material
- modelo termico mas complejo
- modelo de umbral distinto

Recomendaciones:

- Mantener funciones puras cuando sea posible.
- Separar parametros en un dataclass.
- Usar nombres de unidades claros.
- Incluir `reset`.
- Incluir propiedades de solo lectura para estado importante.
- Evitar que el modelo genere graficas directamente.

## Si agregas un nuevo experimento

Ubicacion recomendada:

```text
memristor_simulator/simulations/
```

Un experimento deberia:

1. Recibir parametros.
2. Ejecutar una simulacion.
3. Devolver un objeto de resultado.
4. No depender directamente de Matplotlib.

Las graficas deberian ir en `visualization`.

## Si agregas una nueva metrica

Ubicacion recomendada:

```text
memristor_simulator/validation/metrics.py
```

Una buena metrica deberia:

- recibir arrays
- devolver numeros o diccionarios
- documentar unidades
- manejar casos borde
- no guardar archivos

## Si agregas una nueva grafica

Ubicacion recomendada:

```text
memristor_simulator/visualization/
```

Una funcion de grafica deberia:

- recibir resultados ya calculados
- devolver `fig` o guardar si se pasa `save_path`
- no recalcular la fisica
- etiquetar ejes con unidades
- usar titulos explicativos

## Si agregas exportacion

Ubicacion recomendada:

```text
memristor_simulator/utils/data_export.py
```

Recomendaciones:

- Incluir parametros junto a datos.
- Usar nombres estables de columnas.
- Documentar formato si sera consumido por Unity.
- Evitar formatos ambiguos.

## Si agregas scripts `paso*.py`

Los scripts por pasos son utiles para narrativa academica. Para mantener orden:

- usar numero correlativo
- poner docstring al inicio
- guardar salidas con nombre relacionado
- escribir metricas si aplica
- evitar duplicar demasiada logica del paquete modular

Buen patron:

```python
"""
pasoXX_nombre.py
Explica que demuestra este paso, entradas, salidas y relacion con la tesis.
"""
```

## Pruebas recomendadas

El repositorio tiene pruebas en:

```text
memristor_simulator/tests/
```

Al agregar cambios importantes, conviene probar:

- limites de `x` en `[0,1]`
- `R_on < R_off`
- corriente finita
- no division por cero
- reproducibilidad con semilla
- comportamiento de umbral
- que una simulacion corta termine sin excepciones

## Cuidado con `dt`

El paso temporal es critico. Si agregas dinamicas rapidas:

- reducir `dt`
- probar estabilidad
- comparar Euler vs RK4
- revisar que no aparezcan oscilaciones numericas artificiales

## Cuidado con escalas

En memristores, pequenas variaciones de:

- `D`
- `mu_v`
- `R_on`
- `R_off`

pueden cambiar mucho la dinamica. Documentar siempre las unidades.

## Como agregar un nuevo escenario de paper

1. Crear un preset de parametros en `parameters.py`.
2. Crear un preset de simulacion en `simulation_settings.py`.
3. Crear una funcion `run_nombre_figura()` en `iv_characterization.py` o un modulo nuevo.
4. Crear figura en `visualization`.
5. Exportar CSV/JSON.
6. Agregar metrica en `validation` si corresponde.
7. Documentar en `docs/`.

## Como agregar calibracion

Una calibracion deberia:

- recibir datos experimentales
- definir funcion objetivo
- variar parametros permitidos
- guardar parametros optimizados
- guardar metrica antes/despues
- generar figura comparativa

Archivos existentes como `calibrate_lif.py` y `optimized_parameters.csv` sugieren que este flujo ya empezo.

## Reglas practicas para mantener calidad

- No mezclar simulacion y graficacion si puede evitarse.
- No repetir ecuaciones en muchos scripts.
- Centralizar parametros fisicos.
- Nombrar archivos de salida con el experimento.
- Guardar datos junto con imagenes.
- Usar semillas para variabilidad reproducible.
- Reportar unidades en ejes.
- Mantener scripts academicos simples y explicables.

## Proximas mejoras posibles

Ideas razonables para crecer:

- agregar pruebas automaticas para `StochasticAnalysis`
- agregar comparacion sistematica contra CSV experimentales
- generar un reporte HTML o Markdown automatico con metricas
- unificar scripts `paso*.py` en una CLI
- agregar configuraciones YAML/JSON para experimentos
- crear tabla maestra de figuras: script origen, parametros, salida y metrica
- mejorar control de encoding en archivos con caracteres especiales

## Objetivo de una extension bien hecha

Una extension buena no solo produce una figura nueva. Debe dejar claro:

- que fenomeno representa
- que parametros usa
- que archivo la ejecuta
- que datos genera
- que metrica la respalda
- como se reproduce

Ese nivel de trazabilidad es lo que convierte una simulacion en evidencia academica.
