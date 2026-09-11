# Diagnóstico de fallas con predicción conforme y abstención

Proyecto de maestría basado en AI4I 2020: convertir las salidas de un clasificador
en conjuntos de diagnósticos y decisiones automáticas, asistidas o humanas.

La [propuesta original aprobada](propuesta_final_roberth_jaime.pdf) es la referencia
conceptual. El alcance de ejecución se reduce según la observación docente:
un clasificador base y dos variantes conformes en dos semanas.
Estado actual: las ocho etapas están implementadas y ejecutadas. Los resultados de la
ejecución real están en [RESULTADOS.md](RESULTADOS.md); el plan por etapas en [PLAN.md](PLAN.md).

## Propuesta frente a entregado

La propuesta se redujo por observación docente a un producto mínimo de dos semanas.
Esta tabla deja explícito qué objetivo original se cumple, cuál se cumple de forma
adaptada y cuál queda como extensión documentada.

| Objetivo de la propuesta | Estado | Dónde |
|---|---|---|
| 1. Fundamento matemático: intercambiabilidad, no conformidad, cuantil y cobertura marginal | Cumplido | [RESULTADOS.md](RESULTADOS.md), sección "Cómo se obtiene cada umbral" |
| 2. Split conformal, Mondrian y APS en NumPy sin bibliotecas conformes | Split y Mondrian cumplidos; APS pospuesto | `conformal.py`, `scores.py`, `tests/test_conformal.py` |
| 3. Cobertura independiente del clasificador (fuerte frente a débil) | Adaptado: HistGradientBoosting frente a un MLP en PyTorch, en lugar de GaussianNB | [RESULTADOS.md](RESULTADOS.md), métricas por modelo |
| 4. Split subcubre minoritarias y Mondrian lo corrige | Cumplido y cuantificado | [RESULTADOS.md](RESULTADOS.md), cobertura por clase |
| 5. Ablación de características físicas | Pospuesto | `features.py` reservado |
| 6. Política sensible al costo y alpha óptimo | Pospuesto | `decision.py` implementa solo el enrutamiento por tamaño |
| Paquete instalable y ejecución reproducible | Cumplido | `uv build`, `python -m conformal_fault_inference_with_abstention` |
| Validación empírica de la garantía | Cumplido en una partición; particiones repetidas quedan como recomendación | [RESULTADOS.md](RESULTADOS.md), limitaciones |

## Producto mínimo aprobado

- Un único clasificador base: `HistGradientBoostingClassifier`, elegido como punto
  de partida del proyecto. Usar las variables originales y un preprocesamiento fijo.
- Implementación propia en NumPy de split conformal y Mondrian por clase, con
  puntuación `1 - p(y|x)` y el mismo modelo entrenado para ambas variantes.
- Evaluación de cobertura por clase, tamaño medio del conjunto y tasa de abstención.
  Reportar también cobertura marginal como contexto.
- Abstención cuando el conjunto no tiene exactamente una etiqueta. Desglosar los
  casos asistidos (dos etiquetas) y humanos (cero o más de dos) sin simular costos.
- Un análisis reproducible con conteos, resultados y explicación del cuantil y de
  la garantía bajo sus supuestos; pruebas de los casos matemáticos esenciales.

APS, la política sensible al costo y las clases no vistas quedan como extensiones.
También se posponen el segundo clasificador y la ablación de características físicas
para mantener el producto mínimo dentro del plazo. No son requisitos de entrega.

## Requisito previo al entrenamiento

Después de auditar RNF y resolver las etiquetas con fallas simultáneas, reportar
los conteos reales de Normal, TWF, HDF, PWF y OSF en entrenamiento, calibración y
prueba. La tabla y los criterios están descritos en [data/README.md](data/README.md).
No fijar las proporciones definitivas sin revisar estos conteos.

Para cada clase, revisar si su calibración permite el cuantil solicitado. No
duplicar ni generar ejemplos de calibración para aparentar mayor tamaño muestral.
Si una clase es demasiado escasa, documentar la limitación y revisar el diseño
antes de entrenar. El conjunto de prueba debe quedar reservado.

## Entorno

Se conserva Python 3.13, definido en `.python-version`. Con `uv` instalado:

```bash
uv sync                 # producto mínimo
uv sync --extra torch   # además, PyTorch con CUDA 12.8 para la extensión MLP
uv run ruff check .
```

El extra `torch` instala PyTorch desde el índice `cu128`; el MLP usa la GPU si
`torch.cuda.is_available()` y CPU en caso contrario. El producto mínimo no lo
requiere: sin el extra, `models=("hgb",)` funciona y las pruebas del MLP se omiten.

Ejecutar las pruebas con `uv run pytest`. Para generar el paquete
instalable: `uv build`. Versionar `uv.lock` para reproducir las dependencias.

## Organización

```text
src/conformal_fault_inference_with_abstention/
  data.py          Carga, auditoría y etiquetas
  contracts.py     Tipos y formatos compartidos
  splitting.py     Particiones, conteos y viabilidad de calibración
  model.py         Construcción del único clasificador
  features.py      Variables físicas (extensión)
  scores.py        Puntuaciones de no conformidad
  conformal.py     Split conformal y Mondrian
  decision.py      Abstención según tamaño del conjunto
  evaluation.py    Métricas
  torch_model.py   Extensión: MLP en PyTorch con traza de entrenamiento
  pipeline.py      Configuración y función de ejecución integrada
data/
  raw/             Datos originales
  processed/       Datos derivados
tests/             Pruebas matemáticas y de comportamiento
artifacts/         Resultados generados, excluidos de Git
docs/figures/      Figuras de la ejecución reportada en RESULTADOS.md
```

`src/` contiene el núcleo importable del proyecto: carga de datos, predicción
conforme y evaluación. Instalado con `uv sync`, se usa así:

```python
from conformal_fault_inference_with_abstention import conformal, data, evaluation
```

El modelo se entrena fuera de la capa conforme, que recibe probabilidades y el orden
explícito de las clases; así ambas variantes comparten el mismo clasificador.
`pipeline.run_pipeline` es el punto de entrada importable y
`python -m conformal_fault_inference_with_abstention` el ejecutable. `tests/`
verifica esa lógica y `artifacts/` guarda métricas y figuras de cada ejecución.

## Conceptos del curso aplicados

| Concepto | Dónde se usa |
|---|---|
| Tensores, capas lineales y activaciones | `torch_model.py`: MLP con `nn.Linear` y ReLU sobre lotes de 256 filas |
| Autodiferenciación y ciclo de entrenamiento | `loss.backward()`, `optimizer.zero_grad()`, Adam; traza de pérdida, norma del gradiente y cambio de parámetros por época |
| CPU frente a GPU y precisión float32 | `benchmark_devices`: mediana de repeticiones, comparación de parámetros con tolerancia, costo de transferencias |
| Cuantiles, estadísticos de orden y probabilidad | `conformal_quantile`: rango `ceil((n + 1)(1 − α))`, argumento de intercambiabilidad |
| Comparación de modelos y métricas por clase | HGB frente a MLP con la misma capa conforme; cobertura, tamaño, abstención y error entre automáticas |
| Pipelines de scikit-learn y fuga de información | Preprocesamiento ajustado solo con entrenamiento; calibración y prueba separadas |
| Reproducibilidad | `uv.lock`, semillas, hash SHA-256 del CSV, manifiesto con versiones y GPU, 112 pruebas |

## Ejemplo manual de split conformal

Este ejemplo usa probabilidades inventadas para entender el cálculo; no son
resultados del dataset. Las puntuaciones verdaderas son 0.125, 0.25 y 0.5.
Con alpha=0.5, el rango corregido es 2 y el umbral es 0.25.

```python
from conformal_fault_inference_with_abstention.conformal import SplitConformal
from conformal_fault_inference_with_abstention.decision import route_predictions
from conformal_fault_inference_with_abstention.evaluation import evaluate_sets

predictor = SplitConformal(alpha=0.5).calibrate(
    [[0.875, 0.125], [0.75, 0.25], [0.5, 0.5]],
    ["Normal", "Normal", "TWF"],
    classes=["Normal", "TWF"],
)
sets = predictor.predict_set([[0.75, 0.25], [0.5, 0.5]], classes=["Normal", "TWF"])
print(sets.mask)  # [[True, False], [False, False]]
print(route_predictions(sets))  # ['automatic', 'human']
report = evaluate_sets(["Normal", "TWF"], sets)
print(report.marginal_coverage)  # 0.5
```

Para el modelo real, usar `build_model(random_state=42)`, ajustar con entrenamiento
y pasar `predict_proba` de calibración y `model.classes_` a cada variante. Mondrian
expone la misma interfaz; agrupa por etiqueta verdadera y calcula un umbral por
clase. Los alpha de AI4I están prefijados en `data/README.md`.

## Ejecución real

La forma más corta de reproducir la ejecución reportada, una vez descargados los
datos según [data/README.md](data/README.md):

```bash
uv sync --extra torch
uv run python -m conformal_fault_inference_with_abstention
```

Sin GPU ni PyTorch: `uv sync` y `uv run python -m conformal_fault_inference_with_abstention
--models hgb --benchmark 0`. Con `--help` se listan las opciones; los valores por
defecto son los de `artifacts/run_2026-09-11_seed42`. El mismo pipeline está
disponible como función importable:

`run_pipeline` integra todas las etapas y exporta a `output_dir` la auditoría, las
exclusiones, los índices de partición, los conteos, la viabilidad por alpha, las
probabilidades de calibración y prueba, los umbrales, las métricas globales y por
clase, las figuras y un manifiesto con hash de datos, versiones y GPU usada.

```python
from pathlib import Path

from conformal_fault_inference_with_abstention.contracts import LabelPolicy
from conformal_fault_inference_with_abstention.pipeline import RunConfig, run_pipeline

results = run_pipeline(
    RunConfig(
        data_path=Path("data/raw/ai4i2020.csv"),
        output_dir=Path("artifacts/run_2026-09-11_seed42"),
        label_policy=LabelPolicy(("TWF", "PWF", "OSF", "HDF"), "exclude_only_rnf"),
        calibration_size=0.30,
        test_size=0.20,
        alphas=(0.05, 0.10, 0.20),
        random_state=42,
        models=("hgb", "mlp"),  # ("hgb",) para el producto mínimo sin PyTorch
        mlp_epochs=200,
        mlp_benchmark_repetitions=3,  # 0 omite la comparación CPU/GPU
    )
)
print(results["hgb"]["mondrian"][0.10].by_class)
```

Devuelve `{modelo: {variante: {alpha: EvaluationReport}}}`. Ambos modelos se entrenan
una sola vez y comparten particiones y alphas; el MLP registra además la traza de
entrenamiento por época (pérdida, norma del gradiente y norma del cambio de
parámetros) y una tabla de aciertos por clase antes y después de entrenar. Con
`mlp_benchmark_repetitions > 0` entrena además el MLP en CPU y en CUDA varias
veces y exporta mediana de tiempos, pérdidas, comparación de parámetros con
tolerancia y costo de transferencias, como en el ejercicio de la semana 3.

La calibración debe permanecer separada del ajuste del modelo. La garantía requiere
intercambiabilidad; no garantiza exactitud individual ni detección de clases
desconocidas. Las conclusiones experimentales se medirán, no se asumirán.

## Entrega

El ZIP de entrega debe contener el código, las pruebas, `pyproject.toml`, `uv.lock`,
la documentación (`README.md`, `PLAN.md`, `RESULTADOS.md`, `data/README.md`), la
presentación y las figuras de `docs/`, los datos de `data/raw/` con su manifiesto de
procedencia y los resultados citados en `artifacts/run_2026-09-11_seed42/`. Excluir
`.venv`, `.git`, `dist` y cachés. El script `docs/empaquetar_entrega.py` genera ese
ZIP desde la raíz del repositorio:

```bash
uv run python docs/empaquetar_entrega.py
```

Para comprobar el paquete construido en un entorno limpio:

```bash
uv build
uv venv .venv-prueba && uv pip install --python .venv-prueba dist/*.whl
.venv-prueba/Scripts/python -m conformal_fault_inference_with_abstention --help
```
