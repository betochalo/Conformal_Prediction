"""Punto de entrada importable para integrar las etapas del producto mínimo."""

from dataclasses import dataclass
from pathlib import Path

from .contracts import EvaluationReport, LabelPolicy


@dataclass(frozen=True)
class RunConfig:
    """Sin valores por defecto para decisiones aún pendientes de la auditoría."""

    data_path: Path
    output_dir: Path
    label_policy: LabelPolicy
    calibration_size: float
    test_size: float
    alphas: tuple[float, ...]
    random_state: int


def run_pipeline(config: RunConfig) -> dict[str, dict[float, EvaluationReport]]:
    """Auditar, etiquetar, partir, reportar conteos, entrenar, calibrar y evaluar.

    Guardar auditoría y conteos antes del fit. Entrenar una sola vez; reutilizar
    probabilidades y particiones para ambas variantes y todos los alpha prefijados.
    Retornar {variante: {alpha: reporte}}, con variantes 'split' y 'mondrian'.
    Exportar configuración, hash del CSV, índices, versiones, métricas y figuras
    a output_dir. No seleccionar alpha ni ajustar el modelo sobre prueba.
    """
    raise NotImplementedError("Etapa 7: integración después de completar los módulos.")
