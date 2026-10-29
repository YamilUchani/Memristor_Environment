# Instalación, ejecución y reproducción

Los comandos de este documento están escritos para PowerShell y se ejecutan desde la raíz de `ENTREGA_TUTORA_LISTA/`. Se requiere Python 3.10 o posterior. La copia entregada se preparó y compiló en Windows con Python 3.14.

## 1. Instalar dependencias

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r 02_SOFTWARE/requirements.txt
```

## 2. Ejecutar Neuromorphic Lab

```powershell
Push-Location 02_SOFTWARE
try { python run_app.py } finally { Pop-Location }
```

Para consultar y ejecutar validaciones del laboratorio:

```powershell
Push-Location 02_SOFTWARE/neuromorphic_lab
try {
    python validaciones/run_all_validaciones.py --list
    python validaciones/01_strukov_2008.py
} finally { Pop-Location }
```

## 3. Ejecutar Robot Sim

CartPole con la ventana de configuración:

```powershell
Push-Location 02_SOFTWARE/robot_sim
try { python run_robot.py --seed 0 --env-seed 0 --encoder-seed 0 } finally { Pop-Location }
```

La ventana permite seleccionar REINFORCE o R-STDP, los perfiles de dispositivo, las semillas, el estado inicial `X0` y las opciones de variabilidad del crossbar. El runner independiente de CartPole R-STDP se ejecuta así:

```powershell
Push-Location 02_SOFTWARE/robot_sim
try { python run_rstdp.py --episodes 1000 --crossbar-seed 0 --env-seed 0 --encoder-seed 0 --initial-x 0.1 } finally { Pop-Location }
```

Para abrir los otros entornos:

```powershell
Push-Location 02_SOFTWARE/robot_sim
try {
    python run_vision.py --seed 0 --episodes 5000 --initial-X 0.1
    python run_obstacle.py --seed 0 --episodes 800 --initial-X 0.1
    python run_car.py --seed 0 --episodes 1000 --initial-X 0.1
} finally { Pop-Location }
```

Consulta las opciones de cada lanzador con `python <runner>.py --help`. Las nuevas corridas guardan sus archivos en las carpetas `outputs/` correspondientes.

## 4. Pruebas automatizadas

Desde la raíz del paquete, configura las rutas de importación y ejecuta cada suite:

```powershell
$env:PYTHONPATH = "$PWD/02_SOFTWARE/neuromorphic_lab;$PWD/02_SOFTWARE/robot_sim"
python -m pytest 02_SOFTWARE/neuromorphic_lab/tests -q
python -m pytest 02_SOFTWARE/robot_sim/tests -q
```

## 5. Reproducir el protocolo comparativo

`run_comparison_until_success.py` muestra la curva de éxito durante la ejecución. El lote archivado compara CartPole, visión y obstáculos con REINFORCE y R-STDP, semillas 0 y 1, y `X0=0.1/0.9`; el entorno Car se ejecutó con R-STDP. Para generar un lote nuevo, especifica un directorio de salida que todavía no exista:

```powershell
Push-Location 02_SOFTWARE/robot_sim
try {
    python run_comparison_until_success.py `
        --domains robot vision obstacle car `
        --algorithms reinforce rstdp `
        --skip-car-reinforce `
        --seeds 0 1 `
        --x0 0.1 0.9 `
        --window 300 `
        --target-success 0.90 `
        --max-episodes 10000 `
        --plot-every 10 `
        --output-dir ../../03_EVIDENCIAS/ENTRENAMIENTOS/COMPARATIVA_REPRODUCIDA
} finally { Pop-Location }
```

El criterio de parada del lanzador es superar el 90 % de éxito en una ventana móvil de 300 episodios; por ello, se requieren al menos 271 éxitos en esa ventana. El máximo establecido es de 10 000 episodios por condición. La duración depende del entorno y del equipo. Las corridas archivadas, sus protocolos, matrices, curvas y registros JSON se encuentran en `03_EVIDENCIAS/ENTRENAMIENTOS/`.

## 6. Compilar el PDF institucional

Instala Tectonic y asegúrate de que esté disponible en `PATH`. Desde la raíz del paquete:

```powershell
& './01_TESIS/compile.ps1'
```

El script compila `01_TESIS/PLANTILLA_PDF_FINAL/main.tex` y genera `main.pdf` en esa carpeta. El documento incluye la portada, el resumen, los capítulos 1-4, los apéndices y la bibliografía.

## Procedencia e integridad

`02_SOFTWARE/ENTREGA_MANIFEST_SHA256.json` registra los archivos fuente incluidos. Los registros de `03_EVIDENCIAS/` corresponden a las versiones de software que los produjeron; no se presentan como resultados de una versión posterior. `ENTREGA_SHA256.csv` permite verificar la integridad del paquete. La identificación del PDF se encuentra en `01_TESIS/PDF_FINAL_INFO.md`.
