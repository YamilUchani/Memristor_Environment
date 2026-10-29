# Fuente y compilacion de la tesis

`main.tex` es el controlador oficial en la plantilla UCB. Incluye portada, notacion, resumen, capitulos 1-4, apendices y bibliografia. Los contenidos estan en `Cap1/`, `Cap2/`, `Cap3/`, `Cap4/`, `Resumen/`, `Apendices/` y `Simbolos/`.

El PDF final es `main.pdf` y tiene 109 paginas. El capitulo 3 ocupa 16 paginas impresas. Para regenerarlo desde esta carpeta:

```powershell
tectonic -X compile main.tex --keep-logs --keep-intermediates
```

Desde la raiz de `ENTREGA_TUTORA_LISTA/`, tambien se puede usar `& '01_TESIS/compile.ps1'`.

La compilacion registrada termino con codigo 0 el 28 de octubre de 2026. El registro conserva avisos de composicion en algunas lineas largas y anotaciones de enlaces PDF fuera del area de pagina; las paginas inspeccionadas muestran el contenido dentro del papel.
