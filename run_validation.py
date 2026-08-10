"""
run_validation.py
=================
Orquestador maestro para el Entorno de Validación Física y Neuromórfica.
Permite ejecutar en secuencia de forma profesional y ordenada todos los pasos
de validación (Pasos 1 al 29) y los scripts de análisis.

Autor: Yamil Ronald Uchani Guachalla
Taller de Grado I - Ingeniería Mecatrónica
"""

import os
import sys
import subprocess
import time

# Reconfigurar codificación de consola para evitar UnicodeEncodeError en Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# Paletas de colores ANSI para la consola
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_CYAN = "\033[96m"
C_BOLD = "\033[1m"
C_END = "\033[0m"

# Mapeo estructurado de bloques de validación del Proyecto de Grado
BLOQUES_VALIDACION = {
    "1. Caracterización del Memristor Aislado (Pasos 1-3)": [
        ("paso1_simulacion_basica.py", "Lazo de histéresis estrangulado (I-V)"),
        ("paso2_colapso_frecuencia.py", "Colapso del lazo de histéresis por frecuencia"),
        ("paso3_memristor_estocastico.py", "Variabilidad estocástica ciclo-a-ciclo (C2C)")
    ],
    "2. Acoplamiento Soma Neuronal e Integración (Pasos 4-10)": [
        ("paso4_neurona_aislada.py", "Integración y disparo de neurona LIF estándar"),
        ("paso5_tren_pulsos.py", "Respuesta dinámica del memristor ante trenes de pulsos"),
        ("paso6_memristor_neurona.py", "Transducción de corriente memristiva a la neurona"),
        ("paso7_simulacion_completa.py", "Dashboard dinámico completo memristor-neurona"),
        ("paso8_estado_hrs.py", "Inhibición de disparo en estado de alta resistencia"),
        ("paso9_estado_lrs.py", "Disparo constante en estado de baja resistencia"),
        ("paso10_comparacion.py", "Comparación cuantitativa final frente a datos del paper")
    ],
    "3. Plasticidad y Aprendizaje Sináptico (Pasos 13-16)": [
        ("paso13_plasticidad_y_memoria.py", "Retención de memoria no volátil en apagón eléctrico"),
        ("paso14_ltp_ltd_protocolo.py", "Curvas de Potenciación/Depresión a Largo Plazo (LTP/LTD)"),
        ("paso15_curva_stdp.py", "Simulación emergente de la ventana de Hebb (STDP)"),
        ("paso16_transmision_neuronal.py", "Retardo temporal de transmisión de señal")
    ],
    "4. Arquitecturas Colectivas Crossbar (Pasos 17-24)": [
        ("paso17_red_crossbar.py", "Arreglo crossbar 3x3 y leyes de Kirchhoff"),
        ("paso18_crossbar_escalable.py", "Clasificador lineal de patrones en red 8x4"),
        ("paso19_caracterizacion_crossbar.py", "Mapa de conductancias y fidelidad de lectura"),
        ("paso20_caracterizacion_forming.py", "Proceso de electroformado en muestras vírgenes"),
        ("paso21_caracterizacion_pulsos.py", "Conmutación resistiva rápida por pulsos cortos"),
        ("paso22_analisis_sneak_paths.py", "Evaluación de corrientes parásitas y fugas"),
        ("paso23_analisis_noise_margin.py", "Degradación de margen de ruido de lectura"),
        ("paso24_analisis_metales.py", "Simulación de caída IR de tensión según metalización")
    ],
    "5. Replicaciones del Estado del Arte y Pre-Forming (Pasos 25-29)": [
        ("paso25_replicacion_prezioso_s5.py", "Dispersión estadística D2D de SET/RESET (Prezioso)"),
        ("paso25b_replicacion_s5_curvas_y_mapas.py", "Curvas espaciales complementarias SET/RESET"),
        ("paso25c_replicacion_s5_strukov_ideal.py", "Simulación determinista ideal comparativa"),
        ("paso26_replicacion_prezioso_s6.py", "Evolución estocástica de conductancia ruidosa"),
        ("paso27_replicacion_figura5_sneak.py", "Simulación nodal modopolítico MNA de sneak paths"),
        ("paso28_comparacion_strukov_estocastico.py", "Barras de error determinista vs estocástico"),
        ("paso29_replicacion_pre_forming.py", "Estado de virginidad y conducción por Efecto Túnel")
    ]
}

def print_header():
    os.system('cls' if os.name == 'nt' else 'clear')
    banner = f"""
{C_CYAN}{C_BOLD}=====================================================================
  UNIVERSIDAD CATÓLICA BOLIVIANA "SAN PABLO" - SEDE LA PAZ
  FACULTAD DE INGENIERÍA - CARRERA DE INGENIERÍA MECATRÓNICA
=====================================================================
  SIMULADOR NEUROMÓRFICO BASADO EN MEMRISTORES (VALIDACIÓN)
  Proyecto de Grado - Yamil Ronald Uchani Guachalla
====================================================================={C_END}
"""
    print(banner)

def run_script(script_name, description):
    script_path = os.path.join("memristor_simulator", "steps", script_name)
    print(f"  {C_YELLOW}→ Ejecutando:{C_END} {C_BOLD}{script_name:<40}{C_END} ({description})")
    
    # Asegurar que el entorno de ejecución de Python use UTF-8 para stdout/stderr
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    
    start_time = time.time()
    try:
        # Ejecutar script silenciando salidas para mantener consola limpia, salvo errores
        res = subprocess.run(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            env=env
        )
        duration = time.time() - start_time
        if res.returncode == 0:
            print(f"    {C_GREEN}✔ COMPLETADO{C_END} en {duration:.2f}s")
            return True
        else:
            print(f"    {C_RED}✘ FALLÓ{C_END} (Código {res.returncode})")
            print(f"{C_RED}Detalle del error:{C_END}\n{res.stderr}")
            return False
    except Exception as e:
        print(f"    {C_RED}✘ ERROR EXCEPCIONAL:{C_END} {e}")
        return False

def main():
    print_header()
    
    print(f"{C_BOLD}Seleccione una opción de ejecución:{C_END}")
    print("  [1] Ejecutar toda la suite de validación (Pasos 1 al 29)")
    print("  [2] Ejecutar un bloque de validación específico")
    print("  [3] Ejecutar una simulación/paso individual")
    print("  [4] Salir")
    
    try:
        opt = input(f"\n{C_BLUE}Opción > {C_END}").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nEjecución cancelada.")
        return

    if opt == "1":
        print_header()
        print(f"{C_CYAN}{C_BOLD}=== INICIANDO VALIDACIÓN COMPLETA (PASOS 1 AL 29) ==={C_END}\n")
        failed = []
        for bloque, scripts in BLOQUES_VALIDACION.items():
            print(f"\n{C_BOLD}{bloque}{C_END}")
            print("-" * len(bloque))
            for script, desc in scripts:
                success = run_script(script, desc)
                if not success:
                    failed.append(script)
        
        print("\n" + "=" * 65)
        if not failed:
            print(f"{C_GREEN}{C_BOLD}✔ Suite de validación completada exitosamente sin errores.{C_END}")
        else:
            print(f"{C_RED}{C_BOLD}✘ Se completó la ejecución, pero fallaron los siguientes scripts:{C_END}")
            for f in failed:
                print(f"  - {f}")
        print("=" * 65)

    elif opt == "2":
        print_header()
        print(f"{C_BOLD}Seleccione el bloque a ejecutar:{C_END}")
        bloques = list(BLOQUES_VALIDACION.keys())
        for idx, b in enumerate(bloques):
            print(f"  [{idx + 1}] {b}")
        
        try:
            b_opt = int(input(f"\n{C_BLUE}Bloque > {C_END}").strip()) - 1
            if 0 <= b_opt < len(bloques):
                selected_bloque = bloques[b_opt]
                print_header()
                print(f"{C_CYAN}{C_BOLD}=== EJECUTANDO: {selected_bloque} ==={C_END}\n")
                for script, desc in BLOQUES_VALIDACION[selected_bloque]:
                    run_script(script, desc)
            else:
                print(f"{C_RED}Opción inválida.{C_END}")
        except ValueError:
            print(f"{C_RED}Entrada no válida.{C_END}")

    elif opt == "3":
        print_header()
        print(f"{C_BOLD}Seleccione el script individual a ejecutar:{C_END}")
        all_scripts = []
        for idx, (bloque, scripts) in enumerate(BLOQUES_VALIDACION.items()):
            for s, d in scripts:
                all_scripts.append((s, d))
        
        for idx, (s, d) in enumerate(all_scripts):
            print(f"  [{idx + 1:2d}] {s:<42} ({d})")
            
        try:
            s_opt = int(input(f"\n{C_BLUE}Script # > {C_END}").strip()) - 1
            if 0 <= s_opt < len(all_scripts):
                script, desc = all_scripts[s_opt]
                print_header()
                run_script(script, desc)
            else:
                print(f"{C_RED}Opción inválida.{C_END}")
        except ValueError:
            print(f"{C_RED}Entrada no válida.{C_END}")

    elif opt == "4":
        print("\nSaliendo del orquestador.")
    else:
        print(f"{C_RED}Opción no reconocida.{C_END}")

if __name__ == "__main__":
    # Asegurar soporte de colores en consolas Windows heredadas
    if sys.platform.startswith("win"):
        os.system("color")
    main()
