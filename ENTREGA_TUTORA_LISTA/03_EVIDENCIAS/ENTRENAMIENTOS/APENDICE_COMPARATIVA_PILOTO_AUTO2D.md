# Material para el apéndice: piloto REINFORCE y R-STDP

## Diseño disponible

El piloto controlado existente contiene **12 corridas**: dos algoritmos × dos estados iniciales (`X₀=0.1`, `X₀=0.9`) × tres semillas emparejadas (0, 1 y 2). Cada corrida entrenó 4,000 episodios; total: 48,000 episodios. Se usó el mismo mapa de entrenamiento para los dos algoritmos dentro de cada semilla y un conjunto separado de 50 episodios de prueba con pesos congelados.

**Límite de versión:** el manifiesto del piloto (`PILOTO_AUTO2D_4000/source_manifest.json`) no coincide con el código actual en archivos relevantes, incluidos `run_car.py`, `experiments/training/run_auto2d_pilot.py` y `neurobot/crossbar_brain.py`. Por tanto estos 12 registros describen un snapshot anterior; son útiles como piloto histórico y para mostrar matrices, pero no deben presentarse como comparación del código corregido actual ni combinarse con corridas nuevas.

La configuración y sus diferencias específicas por algoritmo están en `PILOTO_AUTO2D_4000/protocol.json`. El crossbar es **24 × 8** en este piloto Auto2D; no debe describirse como el crossbar CartPole 24 × 12.

## Resumen por condición

Valores de `runs_summary.csv` agregados en `conditions_summary.csv`; las cifras se muestran como media ± desviación estándar entre tres semillas.

| Algoritmo | X₀ | Éxito últimos 100 episodios | Éxito en prueba (50 episodios) | Cambio medio absoluto del crossbar |
|---|---:|---:|---:|---:|
| REINFORCE | 0.1 | 0.697 ± 0.140 | 0.760 ± 0.382 | 0.2282 ± 0.0316 |
| REINFORCE | 0.9 | 0.033 ± 0.021 | 0.093 ± 0.061 | 0.2735 ± 0.0264 |
| R-STDP | 0.1 | 0.710 ± 0.417 | 0.533 ± 0.466 | 0.00658 ± 0.00283 |
| R-STDP | 0.9 | 0.420 ± 0.484 | 0.400 ± 0.514 | 0.00574 ± 0.00134 |

La dispersión entre semillas es alta, por lo que este piloto describe resultados y variabilidad; por sí solo no permite afirmar superioridad estadística. También muestra diferencias grandes en el cambio de X entre las reglas, un dato que conviene reportar junto con la recompensa, no ocultar.

## Registros y figuras disponibles

- `PILOTO_AUTO2D_4000/runs_summary.csv`: una fila por cada una de las 12 corridas.
- `PILOTO_AUTO2D_4000/conditions_summary.csv`: agregados por algoritmo y X₀.
- `PILOTO_AUTO2D_4000/episodes_all.csv`: métricas de entrenamiento por episodio.
- `PILOTO_AUTO2D_4000/runs/<algoritmo>_seed<semilla>_X0_<valor>/crossbar_initial_X.npy` y `crossbar_final_X.npy`: matriz inicial/final por corrida.
- `PILOTO_AUTO2D_4000/runs/<...>/episodes.csv` y `result.json`: registro individual de cada corrida.
- `PILOTO_AUTO2D_4000/crossbar_final_promedio.png` y `crossbar_cambio_abs_promedio.png`: comparación visual de X final y plasticidad.
- `PILOTO_AUTO2D_4000/exito_entrenamiento.png` y `exito_validacion.png`: evolución de éxito.
- `PILOTO_AUTO2D_4000/protocol.json`: protocolo y configuración.

Para el apéndice, conviene incluir las dos gráficas de aprendizaje/validación, las matrices promedio del crossbar y la tabla de 12 corridas. Conservar las 12 filas evita elegir solo semillas con buen resultado. Las matrices individuales y curvas por corrida quedan como material suplementario digital.

## Tamaño recomendado para una comparación confirmatoria

Para comparar el código final, recomiendo **10 semillas emparejadas por condición**, es decir, 40 corridas nuevas: REINFORCE y R-STDP en `X₀=0.1` y `X₀=0.9` para cada una de las mismas 10 semillas. El piloto histórico no cuenta dentro de esas 40 porque fue generado con otro snapshot. Con 5 semillas por condición (20 corridas nuevas) sería un mínimo más defendible, aunque todavía de baja potencia.

No mezclar esas corridas con los otros JSON de `robot_sim_outputs/`: allí hay tareas, tamaños de crossbar, hiperparámetros y versiones de protocolo distintos. Para que el contraste sea interpretable, fija el entorno, encoder, dimensiones, regla de evaluación y todos los hiperparámetros antes de iniciar el nuevo lote; empareja los seeds entre algoritmos y X₀.
