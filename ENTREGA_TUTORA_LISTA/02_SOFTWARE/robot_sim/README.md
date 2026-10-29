# Robot Sim

Simulador de agentes neuromórficos sobre tareas de control, visión y navegación. Los runners de esta carpeta usan los módulos de `neurobot/`; `lab_bridge.py` conecta con `../neuromorphic_lab/`.

## Aplicaciones

| Tarea | Runner | Interfaz |
|---|---|---|
| CartPole con selección REINFORCE/R-STDP | `run_robot.py` | Configuración y gráficas en vivo |
| CartPole R-STDP separado | `run_rstdp.py` | Configuración y gráficas en vivo |
| Visión 8x8 | `run_vision.py` | GUI o `--headless` |
| Visión 8x8 R-STDP | `run_vision_rstdp.py` | GUI o `--headless` |
| Navegación con obstáculos | `run_obstacle.py` | GUI o `--headless` |
| Navegación R-STDP | `run_obstacle_rstdp.py` | GUI o `--headless` |
| Auto 2D | `run_car.py` | GUI o `--headless` |
| Auto 2D R-STDP | `run_car_rstdp.py` | GUI o `--headless` |

## Ejemplos

Ejecuta los comandos desde esta carpeta.

```powershell
python run_robot.py --seed 0 --env-seed 0 --encoder-seed 0
python run_rstdp.py --episodes 1000 --crossbar-seed 0 --env-seed 0 --encoder-seed 0 --initial-x 0.1
python run_vision.py --headless --episodes 500 --seed 0
python run_vision_rstdp.py --headless --episodes 2000 --seed 0
python run_obstacle.py --headless --episodes 800 --seed 0
python run_obstacle_rstdp.py --headless --episodes 1500 --seed 0
python run_car.py --headless --episodes 1000 --seed 0 --initial-X 0.1
python run_car_rstdp.py --headless --episodes 3000 --seed 0 --initial-X 0.1
```

`run_robot.py` presenta la configuración para escoger REINFORCE o R-STDP. `run_rstdp.py` es un runner CartPole independiente para inspeccionar R-STDP en tiempo real. Las opciones de cada runner se consultan con `python <runner>.py --help`.

## Reproducir las pruebas

Desde la raíz del paquete `ENTREGA_TUTORA_LISTA/`, instala las dependencias combinadas y fija las rutas de importación en PowerShell:

```powershell
python -m pip install -r 02_SOFTWARE/requirements.txt
$env:PYTHONPATH = "$PWD/02_SOFTWARE/neuromorphic_lab;$PWD/02_SOFTWARE/robot_sim"
python -m pytest 02_SOFTWARE/robot_sim/tests -q
```

Para lanzadores, validaciones de Neuromorphic Lab y compilación de la tesis, sigue `../05_REPRODUCIR.md`. `run_all.py` ejecuta la verificación CartPole identificada en su propio código; no inicia todas las aplicaciones.

## Resultados y alcance

Los runners escriben nuevos registros y pesos en `outputs/`. Los registros históricos seleccionados para la revisión están en `../../03_EVIDENCIAS/`. Sus metadatos identifican la configuración de cada corrida; no deben tratarse como resultados de una versión posterior del software. Todo el comportamiento descrito es simulación computacional: no se envían pulsos a hardware físico.

## Lote de comparación con parada por éxito

El lanzador `run_comparison_until_success.py` abre una grafica en vivo del exito por episodio y su media movil. Compara ambos algoritmos en Car, Vision, Obstacle y CartPole. Por defecto el comando sugerido usa semillas 0–9, X₀=0.1/0.9, y detiene cada corrida cuando al menos 271 de los últimos 300 episodios son exitosos; el máximo es 10 000 episodios por corrida (160 corridas posibles). No reutiliza pesos guardados.

Desde la raíz del repositorio, ejecuta:

```powershell
python .\robot_sim\run_comparison_until_success.py --seeds 0 1 2 3 4 5 6 7 8 9 --x0 0.1 0.9 --window 300 --target-success 0.90 --max-episodes 10000 --plot-every 10 --output-dir .\ENTREGA_TUTORA_LISTA\03_EVIDENCIAS\ENTRENAMIENTOS\COMPARATIVA_300_90
```

Para continuar un lote interrumpido, añade `--resume` al mismo comando: omite las corridas completas y vuelve a iniciar desde cero la corrida que quedó a medias. Cada corrida completa guarda su JSON, pesos, crossbar inicial/final y gráficas; el resumen conjunto queda en `summary.csv`.



