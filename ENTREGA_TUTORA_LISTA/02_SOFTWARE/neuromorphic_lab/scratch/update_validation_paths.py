import re
from pathlib import Path

base = Path(__file__).resolve().parent.parent / "validaciones"
py_files = sorted(list(base.glob("[0-9]*.py")))

for f in py_files:
    content = f.read_text(encoding="utf-8")
    
    # Check if figuras or datos directory creation is needed or replace save paths
    content = re.sub(
        r'output_dir\s*/\s*(["\'])([^"\']+\.(?:png|pdf))\1',
        r'output_dir / "figuras" / \1\2\1',
        content
    )
    content = re.sub(
        r'output_dir\s*/\s*(["\'])([^"\']+\.csv)\1',
        r'output_dir / "datos" / \1\2\1',
        content
    )
    
    f.write_text(content, encoding="utf-8")
    print(f"Updated: {f.name}")

print("All output paths updated!")
