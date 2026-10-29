# Entrega para revisión de la tutora

Esta carpeta reúne el documento de tesis, las copias de los proyectos de software y las evidencias disponibles. El documento principal es `01_TESIS/PLANTILLA_PDF_FINAL/main.pdf`.

## Contenido

- `01_TESIS/PLANTILLA_PDF_FINAL/`: tesis en la plantilla institucional de la UCB, PDF de 134 páginas y fuentes LaTeX. El capítulo 3 ocupa 16 páginas impresas. El apéndice visual incluye imágenes del crossbar y curvas de éxito para 32 condiciones seleccionadas.
- `02_SOFTWARE/`: copias de Neuromorphic Lab y Robot Sim, con pruebas, configuraciones y lanzadores.
- `03_EVIDENCIAS/`: datos de referencia, curvas, trazabilidad, resultados de pruebas y registros de entrenamiento en JSON, matrices, metadatos e imágenes.
- `04_REFERENCIAS/`: bibliografía y documentación de procedencia.
- `05_REPRODUCIR.md`: instrucciones de instalación, ejecución, verificación y compilación.

## Estado e integridad

La tesis se compiló desde `01_TESIS/PLANTILLA_PDF_FINAL/main.tex`. La identificación del PDF y sus datos de compilación se encuentran en `01_TESIS/PDF_FINAL_INFO.md`. El manifiesto SHA-256 de la copia de software está en `02_SOFTWARE/ENTREGA_MANIFEST_SHA256.json`.

La copia de software se contrastó con el espacio de trabajo y el alcance de esa comparación se documenta en `03_EVIDENCIAS/RESULTADOS_PRUEBAS_SOFTWARE.md`. Los registros de REINFORCE, R-STDP y los lotes comparativos están en `03_EVIDENCIAS/ENTRENAMIENTOS/`. La nota `03_EVIDENCIAS/VALIDACION_OBSERVACIONES_TUTORA.md` describe el funcionamiento de LIF, spikes, WTA, decoder y aprendizaje en el simulador.

`ENTREGA_SHA256.csv` permite verificar la integridad de los archivos del paquete; el propio manifiesto está excluido de su lista. La extensión suplementaria de navegación Auto 2D para semilla 2, fechada el 28/10/2026, se encuentra en `03_EVIDENCIAS/ENTRENAMIENTOS/CAR_RSTDP_EXTENSION_SEED2_X0_0.1/`. Su protocolo indica que no constituye una condición independiente y que registra episodios posteriores a un prefijo documentado.
