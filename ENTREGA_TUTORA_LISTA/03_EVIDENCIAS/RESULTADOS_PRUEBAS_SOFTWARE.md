# Estado de comprobacion del software y resultados de entrenamiento

Actualizado: 28 de octubre de 2026.

El autor informa que abrio y comprobo manualmente Robot Sim y Neuromorphic Lab despues de las correcciones. La copia de software de `02_SOFTWARE/` se contrasto con el workspace: los archivos activos de Robot Sim coinciden; Neuromorphic Lab coincide salvo el directorio historico `validaciones_BACKUP_2026/`, que se conserva fuera de la copia de entrega. El manifiesto actualizado registra el contenido entregado.

Esta revision documental no volvio a ejecutar las suites automatizadas ni un entrenamiento. Por tanto, no atribuye un conteo nuevo de pruebas al snapshot actual. Los comandos para reproducirlas estan en `05_REPRODUCIR.md`; la evidencia historica de pruebas conserva su fecha y sus hashes de origen.

## Resultados del piloto de tareas

Los registros del piloto actual estan en `ENTRENAMIENTOS/COMPARATIVA_SEEDS_0_1_X0/` y la repeticion de Car R-STDP esta en `ENTRENAMIENTOS/CAR_RSTDP_EVAL100_SEEDS_2_4_X0/`. El criterio fue lograr al menos 271 exitos en una ventana movil de 300 episodios (90.33%), con tope de 10,000 episodios; no se exigieron 300 exitos consecutivos.

| Tarea y regla | Condiciones que alcanzaron criterio | Lectura del registro |
|---|---:|---|
| CartPole, R-STDP | 4/4 | Las cuatro condiciones del piloto alcanzaron el criterio. |
| CartPole, REINFORCE | 2/4 | Dos corridas quedaron incompletas; no se cuentan como fallos al tope. |
| Vision, R-STDP | 4/4 | Las cuatro condiciones alcanzaron el criterio. |
| Vision, REINFORCE | 4/4 | Las cuatro condiciones alcanzaron el criterio. |
| Obstaculos, R-STDP | 4/4 | Las cuatro condiciones alcanzaron el criterio. |
| Obstaculos, REINFORCE | 4/4 | Las cuatro condiciones alcanzaron el criterio. |
| Car, lote original R-STDP | 1/4 | Protocolo con evaluacion greedy cada 200 episodios. |
| Car, repeticion R-STDP | 2/4 | Evaluacion cada 100 episodios: las dos condiciones de semilla 2 alcanzaron; las dos de semilla 4 llegaron a 10,000 sin alcanzar. |

### Corridas suplementarias de navegacion Auto 2D del 28 de octubre

Se conserva por separado `ENTRENAMIENTOS/CAR_RSTDP_EXTENSION_SEED2_X0_0.1/`: episodios 751–2098 de una repeticion determinista Auto 2D R-STDP (crossbar 24x8, semilla 2, X0=0.1). Los episodios 1–750 coinciden con el registro seed 2/X0=0.1 de `CAR_RSTDP_EVAL100_SEEDS_2_4_X0/` y no se duplican. Es una extension del registro, no una corrida independiente ni una reanudacion desde checkpoint. No forma parte del lote principal de 28 condiciones ni de una comparacion emparejada.

| Semilla | Episodios pedidos/completados | Exito entrenamiento: global; ultimos 100; ultimos 300 | Greedy mismo mapa (20) | Prueba en mapas apartados (50) | Lectura |
|---:|---:|---:|---:|---:|---|
| 2 | 9000/2098 en la repeticion; sufijo conservado 751–2098 | 83.08%; 95%; 97% en la repeticion completa | 20/20 (100%) | 43/50 (86%) | No es una condicion independiente: los primeros 750 episodios coinciden con el registro seed 2/X0=0.1 del lote de evaluacion cada 100. La repeticion se hizo desde el inicio, no desde un checkpoint. El sufijo se conserva para documentar la trayectoria posterior; prueba apartada queda en 86%. |

Las tasas corresponden a conjuntos diferentes: la de entrenamiento usa episodios con exploracion; la evaluacion greedy en el mismo mapa usa 20 episodios; la prueba apartada usa 50 episodios en mapas separados. Por tanto, la semilla 2 no demuestra una tasa general de 90% pese a su ventana final de entrenamiento. El registro de extension guarda el sufijo de actividad y los pesos/figuras finales; el registro de evaluacion cada 100 conserva la trayectoria completa original. Consultar `summary.csv`, `protocol.json` y `SHA256SUMS.csv` en ambas carpetas.

El lote tiene pocas semillas y no incluye una comparacion emparejada Car REINFORCE. Estos resultados describen estas corridas y no establecen superioridad estadistica ni generalizacion. El WTA y las neuronas LIF son parte activa de la dinamica simulada; las acciones se decodifican de corrientes de columnas, por separado. La STDP local del crossbar que funciona forma parte de la variante hibrida REINFORCE + R-STDP. La variante diferente que aplico el spike LIF real como evento post directamente al aprendizaje de tarea tuvo rendimiento muy bajo en pruebas exploratorias. Las cifras de esta tabla bajo la etiqueta R-STDP corresponden al agente hibrido del runner y usan credito pre/accion centrado por la politica; no corresponden a esa variante directa con post LIF.

Los JSON, matrices de pesos, metadatos, curvas y hashes de cada corrida estan en las carpetas de entrenamiento indicadas. El inventario de `ENTRENAMIENTOS/README.md` identifica las trazas detalladas por paso que se mantienen en el proyecto fuente y los lotes auxiliares parciales que no se combinaron con la tabla principal.
