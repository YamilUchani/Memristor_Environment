"""Sincroniza el esquema general de la plantilla con la versión de documentación."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "PLANTILLA_PDF_FINAL" / "figuras" / "esquema_general.png"
OUTPUT = ROOT / "docs" / "latex" / "figuras" / "esquema_general.png"


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"No se encontró la figura original: {SOURCE}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE, OUTPUT)
    print(f"Esquema de la plantilla sincronizado: {OUTPUT}")


if __name__ == "__main__":
    main()
