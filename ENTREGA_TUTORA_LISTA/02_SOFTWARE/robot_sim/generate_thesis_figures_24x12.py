"""
Generador de figuras de alta resolución para la Tesis (Arquitectura 24x12).
Genera gráficos vectoriales (.pdf y .png 300 DPI) para:
 1. Curva de aprendizaje Base Control 24x12 con intervalo de confianza (seeds 0-4)
 2. Comparativa de Ablaciones (A vs B vs C vs D)
 3. Heatmap de la Matriz X consolidada (Yamil_1.npy / Base 24x12)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT.parent / "docs" / "latex" / "figuras"
FIG_DIR.mkdir(parents=True, exist_ok=True)
THESIS_FIG_DIR = ROOT.parent / "PLANTILLA_PDF_FINAL" / "figuras"
THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Estilo académico publicable
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

def plot_learning_curve_and_seeds():
    seeds = [0, 1, 2, 3, 4]
    all_rewards = []
    
    for s in seeds:
        path = OUT_DIR / f"run_24x12_A_d2d_c2c_seed{s}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Falta el registro de la semilla {s}: {path}"
            )
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data.get("seed") != s or data.get("n_episodes") != 300:
            raise ValueError(
                f"Semilla o número de episodios incorrecto en {path.name}"
            )
        if data.get("parameters", {}).get("d2d_enabled") is not True:
            raise ValueError(f"D2D no está activado en {path.name}")
        if data.get("parameters", {}).get("c2c_enabled") is not True:
            raise ValueError(f"C2C no está activado en {path.name}")
        if data.get("freeze_update_above") != 430.0:
            raise ValueError(f"Umbral de actualización incompatible en {path.name}")
        all_rewards.append(data["rewards"])

    min_len = min(len(r) for r in all_rewards)
    all_rewards = np.array([r[:min_len] for r in all_rewards])
    
    mean_r = np.mean(all_rewards, axis=0)
    std_r = np.std(all_rewards, axis=0)
    eps = np.arange(1, min_len + 1)
    
    plt.figure(figsize=(8, 4.5))
    plt.plot(eps, mean_r, color="#1f77b4", linewidth=2.0, label=r"REINFORCE 24$\times$12 con D2D/C2C ($\mu$)")
    plt.fill_between(eps, mean_r - std_r, mean_r + std_r, color="#1f77b4", alpha=0.2, label=r"Sombra $\pm 1\sigma$ (5 semillas)")
    plt.axhline(
        430.0,
        color="#d62728",
        linestyle="--",
        alpha=0.8,
        label=r"Sin actualización si $R_{ep} \geq 430$",
    )
    
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa del Episodio ($R_{ep}$)")
    plt.title("CartPole: REINFORCE con variabilidad D2D/C2C (24×12)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    png_path = FIG_DIR / "cartpole_learning_24x12_5seeds_d2d_c2c.png"
    pdf_path = FIG_DIR / "curva_aprendizaje_24x12_5seeds_d2d_c2c.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.savefig(THESIS_FIG_DIR / png_path.name, dpi=300)
    plt.close()
    print(f"[ok] Figura generada: {png_path.name}")

def plot_ablations_comparison():
    configs = {
        "A_base_seed1": ("A: Control Base (24x12)", "#1f77b4"),
        "B_tempfixed_seed0": (r"B: Temp Fija Físicamente ($T=0.10$)", "#ff7f0e"),
        "C_seed0": (r"C: Sin Traza ($\mathcal{E}=0$)", "#2ca02c"),
        "D_hebbian_seed0": ("D: Hebbiano no modulado", "#d62728")
    }
    
    plt.figure(figsize=(9, 5))
    for key, (label, color) in configs.items():
        json_name = f"run_24x12_{key}.json"
        path = OUT_DIR / json_name
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                r = data["rewards"] if "rewards" in data else data["reward"]
                # Suavizado de media móvil de 5 episodios
                smooth_r = np.convolve(r, np.ones(5)/5, mode='valid')
                plt.plot(np.arange(5, len(r) + 1), smooth_r, color=color, linewidth=2.0, label=label)
    
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa Suavizada (Media móvil 5 ep)")
    plt.title("Estudio de Ablación en Crossbar Memristivo 24×12")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="center right")
    plt.tight_layout()
    
    png_path = FIG_DIR / "comparativa_ablaciones_24x12.png"
    pdf_path = FIG_DIR / "comparativa_ablaciones_24x12.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"[ok] Figura generada: {png_path.name}")

def plot_weights_heatmap(weights_file="Yamil_1.npy"):
    path = OUT_DIR / weights_file
    if not path.exists():
        path = OUT_DIR / "run_24x12_A_base_seed1_X.npy"
    if not path.exists():
        path = OUT_DIR / "run_24x12_A_base_seed0_X.npy"
    if not path.exists():
        print("No se encontró matriz de pesos para heatmap.")
        return

    X = np.load(path)
    plt.figure(figsize=(7, 8))
    im = plt.imshow(X, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(im)
    cbar.set_label(r"Estado interno $X_{ij} \in [0, 1]$", rotation=270, labelpad=15)
    
    plt.xlabel("Acciones de Control (12 columnas)")
    plt.ylabel("Receptores del Encoder Posición-Velocidad (24 filas)")
    plt.title("Matriz de Estado Interno Consolidada ($X$)")
    plt.xticks(np.arange(0, 12))
    plt.yticks(np.arange(0, 24))
    plt.tight_layout()
    
    png_path = FIG_DIR / "heatmap_matriz_X_24x12.png"
    pdf_path = FIG_DIR / "heatmap_matriz_X_24x12.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.savefig(THESIS_FIG_DIR / png_path.name, dpi=300)
    plt.savefig(THESIS_FIG_DIR / pdf_path.name)
    plt.close()
    print(f"[ok] Figura generada: {png_path.name}")

if __name__ == "__main__":
    plot_learning_curve_and_seeds()
    plot_ablations_comparison()
    plot_weights_heatmap()
