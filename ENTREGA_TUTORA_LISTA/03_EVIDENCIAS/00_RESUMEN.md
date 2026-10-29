# Evidencias incluidas

Esta carpeta conserva registros originales y evidencias curadas de Neuromorphic Lab y Robot Sim. Lee cada resultado con su fecha, configuración, semillas y versión de código; una corrida antigua no demuestra el comportamiento de una versión posterior.

Los resultados REINFORCE y R-STDP exportados desde `robot_sim/outputs` se encuentran en `ENTRENAMIENTOS/robot_sim_outputs/`. Se incluyeron sus JSON, pesos y metadatos; el manifiesto SHA-256 está en `ENTRENAMIENTOS/ROBOT_SIM_ENTRENAMIENTOS_SHA256.csv`. Consulta `ENTRENAMIENTOS/README.md` para el inventario y el detalle de las trazas grandes que permanecen en el repositorio original.

El paquete incluye el piloto Auto 2D `PILOTO_AUTO2D_4000/`: 12 corridas (dos reglas, `X0=0.1/0.9`, tres semillas emparejadas y 4000 episodios por corrida). Es un piloto exploratorio con variabilidad entre semillas; no demuestra superioridad general ni validacion de hardware. No se incluye como evidencia un lote R-STDP de 40 corridas.

El apendice `Evidencia visual de los entrenamientos` de la tesis incorpora las 64 imagenes originales de crossbar y curva de exito correspondientes a las 28 condiciones del lote principal (semillas 0 y 1) y a las cuatro condiciones Car R-STDP de semillas 2 y 4. Cada par se vincula con su JSON y protocolo en `ENTRENAMIENTOS/`; las figuras documentan corridas existentes y no suman ejecuciones nuevas.

Se conserva `ENTRENAMIENTOS/CAR_RSTDP_EXTENSION_SEED2_X0_0.1/` como evidencia suplementaria: el sufijo de episodios 751–2098 de una repeticion Auto 2D R-STDP que coincide en sus primeros 750 episodios con la condicion seed 2/X0=0.1 de `CAR_RSTDP_EVAL100_SEEDS_2_4_X0/`. Se retiraron los registros duplicados de semillas 0 y 1 y las copias redundantes de Car R-STDP; el sufijo no se cuenta como corrida o semilla adicional.

El codigo fuente del paquete se contrasto con el workspace el 28 de octubre de 2026: los archivos activos compartidos de Robot Sim coinciden; Neuromorphic Lab coincide excepto por el respaldo historico `validaciones_BACKUP_2026/`, que se deja fuera de la entrega. El autor confirma haber comprobado manualmente las aplicaciones actuales despues de las correcciones. Esta actualizacion no volvio a ejecutar las suites automatizadas ni entrenamientos; los conteos historicos de `RESULTADOS_PRUEBAS_SOFTWARE.md` no se presentan como pruebas nuevas del snapshot.

- `VALIDACION_OBSERVACIONES_TUTORA.md`: trazabilidad entre observaciones, evidencia y límites.
- `datos_referencia/`: datos de referencia y salidas tabulares documentadas.
- `curvas_y_figuras/`: figuras y curvas de validación.
- `resultados_originales/`: corridas históricas y registros de sus versiones originales.
- `PILOTO_AUTO2D_4000/`: resultados, protocolo y manifiesto del piloto.
- `../02_SOFTWARE/ENTREGA_MANIFEST_SHA256.json`: hashes del software fuente incluido actualmente.

Las pruebas de software verifican operaciones implementadas; una comparación con curvas digitalizadas verifica concordancia bajo su protocolo. Ninguna constituye por sí sola una medición física del dispositivo.
