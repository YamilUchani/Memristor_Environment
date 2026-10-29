# Fuente y compilación de la tesis

`main.tex` es el archivo principal de la tesis en la plantilla institucional de la UCB. Incluye la portada, la notación, el resumen, los capítulos 1-4, los apéndices y la bibliografía. Los contenidos se organizan en `Cap1/`, `Cap2/`, `Cap3/`, `Cap4/`, `Resumen/`, `Apendices/` y `Simbolos/`.

El PDF compilado es `main.pdf` y tiene 134 páginas. El capítulo 3 ocupa 16 páginas impresas. Para compilar desde esta carpeta, ejecuta:

```powershell
tectonic -X compile main.tex --keep-logs --keep-intermediates
```

Desde la raíz de `ENTREGA_TUTORA_LISTA/`, también se puede ejecutar `& './01_TESIS/compile.ps1'`.

La última compilación registrada terminó con código 0 el 28 de octubre de 2026. El registro contiene advertencias tipográficas no fatales: líneas ligeramente desbordadas, cajas con espaciado irregular y sustituciones de algunos glifos en marcadores del PDF.
