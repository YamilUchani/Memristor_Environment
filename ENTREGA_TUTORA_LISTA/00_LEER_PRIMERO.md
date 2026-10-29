# Entrega para revision de tutora

Esta carpeta reune la tesis completa, los dos proyectos de software y las evidencias disponibles. El documento principal es `01_TESIS/PLANTILLA_PDF_FINAL/main.pdf`.

## Contenido

- `01_TESIS/PLANTILLA_PDF_FINAL/`: tesis en la plantilla institucional UCB, PDF final de 109 paginas y fuentes LaTeX. El capitulo 3 ocupa 16 paginas impresas. Incluye el apendice visual con crossbar y curva de exito para 32 condiciones seleccionadas. El PDF para revisar es `main.pdf`.
- `02_SOFTWARE/`: copia de Neuromorphic Lab y Robot Sim, sus pruebas, configuraciones y lanzadores.
- `03_EVIDENCIAS/`: datos de referencia, curvas, trazabilidad, pruebas y registros de entrenamiento con JSON, pesos, metadatos e imagenes del crossbar y del resultado.
- `04_REFERENCIAS/`: bibliografia y procedencia de datos.
- `05_REPRODUCIR.md`: instalacion, lanzadores, comandos de comprobacion y compilacion.

## Estado del paquete

La tesis oficial se compilo desde `01_TESIS/PLANTILLA_PDF_FINAL/main.tex`; el PDF vigente y sus datos de compilacion estan identificados en `01_TESIS/PDF_FINAL_INFO.md`. El manifiesto SHA-256 de software esta en `02_SOFTWARE/ENTREGA_MANIFEST_SHA256.json`.

La copia de software se contrasto con el workspace y se documento el alcance de su correspondencia en `03_EVIDENCIAS/RESULTADOS_PRUEBAS_SOFTWARE.md`. Los registros de REINFORCE y R-STDP y el piloto comparativo entre tareas estan en `03_EVIDENCIAS/ENTRENAMIENTOS/`. La trazabilidad tecnica en `03_EVIDENCIAS/VALIDACION_OBSERVACIONES_TUTORA.md` aclara como funcionan LIF, spikes, WTA, decoder y aprendizaje en el simulador.

Para revisar integridad de todos los archivos entregados, consulta `ENTREGA_SHA256.csv` (el propio manifiesto queda excluido de su lista). La extension suplementaria de navegacion Auto 2D para semilla 2 del 28/10/2026 esta en `03_EVIDENCIAS/ENTRENAMIENTOS/CAR_RSTDP_EXTENSION_SEED2_X0_0.1/`; su protocolo explica que no es una condicion independiente y que conserva solo los episodios posteriores al prefijo ya documentado.
