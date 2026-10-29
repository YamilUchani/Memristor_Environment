# Instalar, ejecutar y reproducir

Los comandos son para PowerShell y se ejecutan desde la ra?z `ENTREGA_TUTORA_LISTA/`. Cada bloque usa rutas expl?citas para que funcione aunque se ejecute por separado. Se requiere Python 3.10 o posterior; el paquete se prepar? con Python 3.14 en Windows.

## 1. Instalar dependencias

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r 02_SOFTWARE/requirements.txt
```

## 2. Abrir Neuromorphic Lab

```powershell
Push-Location 02_SOFTWARE
try { python run_app.py } finally { Pop-Location }
```

Para listar y ejecutar validaciones del laboratorio:

```powershell
Push-Location 02_SOFTWARE/neuromorphic_lab
try {
    python validaciones/run_all_validaciones.py --list
    python validaciones/01_strukov_2008.py
} finally { Pop-Location }
```

## 3. Abrir Robot Sim

```powershell
Push-Location 02_SOFTWARE/robot_sim
try { python run_robot.py --seed 0 --env-seed 0 --encoder-seed 0 } finally { Pop-Location }
```

`run_robot.py` abre la configuraci?n de CartPole. All? se eligen REINFORCE o R-STDP, los perfiles, las semillas, `X0` y las opciones del crossbar. Para inspeccionar el runner independiente de CartPole R-STDP:

```powershell
Push-Location 02_SOFTWARE/robot_sim
try { python run_rstdp.py --episodes 1000 --crossbar-seed 0 --env-seed 0 --encoder-seed 0 --initial-x 0.1 } finally { Pop-Location }
```

Lanzadores de las otras tareas:

```powershell
Push-Location 02_SOFTWARE/robot_sim
try {
    python run_vision.py --seed 0 --episodes 5000 --initial-X 0.1
    python run_obstacle.py --seed 0 --episodes 800 --initial-X 0.1
    python run_car.py --seed 0 --episodes 1000 --initial-X 0.1
} finally { Pop-Location }
```

Los runners que aceptan `--headless` pueden ejecutarse sin ventana. Consulta las opciones disponibles con `python <runner>.py --help`. Las nuevas corridas guardan archivos en las carpetas `outputs/` de cada aplicaci?n.

## 4. Ejecutar las pruebas automatizadas

Desde la ra?z del paquete, configura las rutas de importaci?n y ejecuta las suites por separado:

```powershell
$env:PYTHONPATH = "$PWD/02_SOFTWARE/neuromorphic_lab;$PWD/02_SOFTWARE/robot_sim"
python -m pytest 02_SOFTWARE/neuromorphic_lab/tests -q
python -m pytest 02_SOFTWARE/robot_sim/tests -q
```

## 5. Reproducir el lote de comparaci?n con parada por ?xito

El lanzador `run_comparison_until_success.py` muestra la curva de ?xito durante la ejecuci?n. El protocolo principal archivado compara CartPole, visi?n y obst?culos con ambas reglas, semillas 0 y 1, y `X0=0.1/0.9`; Car se ejecut? con R-STDP. Para generar un lote nuevo usa una carpeta de salida nueva:

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
        --output-dir ..\..\03_EVIDENCIAS\ENTRENAMIENTOS\COMPARATIVA_REPRODUCIDA
} finally { Pop-Location }
```

El criterio es al menos 271 ?xitos en una ventana m?vil de 300 (90.33%); no requiere 300 ?xitos consecutivos. El m?ximo es 10,000 episodios por condici?n. Es un lote largo; el PDF y las carpetas de evidencia ya contienen las corridas archivadas con sus protocolos, matrices, curvas y registros JSON. No hace falta volver a ejecutarlas para consultar sus resultados.

## 6. Compilar el PDF institucional

Instala Tectonic y agr?galo a `PATH`. Desde la ra?z del paquete:

```powershell
& '.\01_TESIS\compile.ps1'
```

El script compila `01_TESIS/PLANTILLA_PDF_FINAL/main.tex` y genera `main.pdf` en esa carpeta. El PDF incluye portada, resumen, cap?tulos 1?4, ap?ndices y bibliograf?a.

## Procedencia y alcance

`02_SOFTWARE/ENTREGA_MANIFEST_SHA256.json` registra los archivos fuente incluidos. Las corridas de `03_EVIDENCIAS/` se conservan como registros de las versiones que las produjeron; no son resultados de una versi?n posterior. El manifiesto `ENTREGA_SHA256.csv` permite revisar la integridad del paquete. La fecha, tama?o, p?ginas y hash del PDF est?n en `01_TESIS/PDF_FINAL_INFO.md`.
