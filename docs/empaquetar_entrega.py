"""Genera el ZIP de entrega desde la raíz del repositorio.

Incluye código, pruebas, configuración, documentación, presentación, figuras, datos
con su manifiesto y los resultados citados. Excluye entornos, Git, cachés y dist.

    uv run python docs/empaquetar_entrega.py [--run artifacts/run_2026-09-11_seed42]
"""

import argparse
import zipfile
from pathlib import Path

INCLUDE_DIRS = ("src", "tests", "docs", "data", "notebooks")
INCLUDE_FILES = (
    "README.md",
    "PLAN.md",
    "RESULTADOS.md",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".gitignore",
    "propuesta_final_roberth_jaime.pdf",
)
EXCLUDE_PARTS = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "dist"}


def wanted(path: Path) -> bool:
    return not any(part in EXCLUDE_PARTS for part in path.parts) and not path.name.endswith(".pyc")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("artifacts/run_2026-09-11_seed42"))
    parser.add_argument("--output", type=Path, default=Path("Entrega_Chachalo_Astudillo.zip"))
    args = parser.parse_args()
    root = Path(".").resolve()
    files: list[Path] = []
    for name in INCLUDE_FILES:
        if (root / name).exists():
            files.append(root / name)
    for folder in (*INCLUDE_DIRS, args.run):
        base = root / folder
        if base.exists():
            files += [p for p in base.rglob("*") if p.is_file() and wanted(p.relative_to(root))]
    files.append(root / "artifacts" / "README.md")
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(set(files)):
            zf.write(path, path.relative_to(root).as_posix())
    size = args.output.stat().st_size / 2**20
    print(f"{args.output}: {len(set(files))} archivos, {size:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
