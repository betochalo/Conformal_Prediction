"""Punto de entrada reproducible: `uv run python -m conformal_fault_inference_with_abstention`.

Los valores por defecto son los de la ejecución reportada en RESULTADOS.md
(`artifacts/run_2026-09-11_seed42`). Sin `--output`, cada ejecución crea una carpeta
nueva con la fecha, la semilla y los modelos, y nunca sobrescribe una existente salvo
con `--overwrite`.
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

from .contracts import FAILURE_COLUMNS, LabelPolicy
from .pipeline import MODELS, RunConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conformal_fault_inference_with_abstention",
        description="Diagnóstico de fallas AI4I con predicción conforme y abstención.",
    )
    parser.add_argument("--data", type=Path, default=Path("data/raw/ai4i2020.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="carpeta de salida; por defecto artifacts/run_<fecha>_seed<semilla>_<modelos>",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="permitir escribir en una carpeta con archivos"
    )
    parser.add_argument(
        "--priority",
        nargs=4,
        default=("TWF", "PWF", "OSF", "HDF"),
        metavar="MODO",
        help="prioridad ante fallas simultáneas; el primero activo gana",
    )
    parser.add_argument(
        "--rnf-policy", choices=("exclude_rows", "exclude_only_rnf"), default="exclude_only_rnf"
    )
    parser.add_argument("--calibration-size", type=float, default=0.30)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--alphas", type=float, nargs="+", default=(0.05, 0.10, 0.20))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=("hgb", "mlp"))
    parser.add_argument("--mlp-epochs", type=int, default=200)
    parser.add_argument("--mlp-device", default="auto", help="auto, cpu o cuda")
    parser.add_argument(
        "--benchmark",
        type=int,
        default=3,
        help="repeticiones de la comparación CPU/GPU del MLP; 0 la omite",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    if sorted(args.priority) != sorted(FAILURE_COLUMNS):
        raise SystemExit(f"--priority debe ser una permutación de {FAILURE_COLUMNS}.")
    if args.output is None:
        stamp = datetime.now(UTC).strftime("%Y-%m-%d")
        args.output = Path("artifacts") / f"run_{stamp}_seed{args.seed}_{'-'.join(args.models)}"
    return RunConfig(
        data_path=args.data,
        output_dir=args.output,
        label_policy=LabelPolicy(tuple(args.priority), args.rnf_policy),
        calibration_size=args.calibration_size,
        test_size=args.test_size,
        alphas=tuple(args.alphas),
        random_state=args.seed,
        models=tuple(args.models),
        mlp_epochs=args.mlp_epochs,
        mlp_device=args.mlp_device,
        mlp_benchmark_repetitions=args.benchmark,
        overwrite=args.overwrite,
    )


def main(argv: list[str] | None = None) -> int:
    config = config_from_args(build_parser().parse_args(argv))
    results = run_pipeline(config)
    print(f"Resultados guardados en {config.output_dir}")
    print("modelo variante  alpha cobertura  tamaño  abst.")
    for model, variants in results.items():
        for variant, per_alpha in variants.items():
            for alpha, report in per_alpha.items():
                print(
                    f"{model:6s} {variant:9s} {alpha:5.2f} {report.marginal_coverage:9.3f} "
                    f"{report.mean_set_size:7.2f} {report.abstention_rate:6.3f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
