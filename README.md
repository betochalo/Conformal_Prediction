# Diagnóstico de fallas con predicción conforme y abstención

Proyecto de maestría basado en AI4I 2020: convertir las salidas de un clasificador
en conjuntos de diagnósticos y decisiones automáticas, asistidas o humanas.


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
| 5. Ablación de características físicas | Pospuesto | Sin módulo; extensión futura |
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

Cumplido: la auditoría, la política de etiquetas, los conteos reales por bloque y la
viabilidad de calibración por alpha están en [data/README.md](data/README.md).
Cambiar la política o la partición exige regenerar ese reporte antes de entrenar.

## Entorno

Se conserva Python 3.13, definido en `.python-version`. Con `uv` instalado:

```bash
uv sync              
uv sync --extra torch 
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
  scores.py        Puntuaciones de no conformidad
  conformal.py     Split conformal y Mondrian
  decision.py      Abstención según tamaño del conjunto
  evaluation.py    Métricas
  torch_model.py   Extensión: MLP en PyTorch con traza de entrenamiento
  pipeline.py      Configuración y función de ejecución integrada
data/
  raw/             Datos originales, excluidos de Git
tests/             Pruebas matemáticas y de comportamiento
artifacts/         Resultados versionados, una carpeta por ejecución
docs/              Presentación y scripts auxiliares
notebooks/         Notebook de análisis de la ejecución guardada
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

## Ejecución real

La forma más corta de reproducir la ejecución reportada, una vez descargados los
datos según [data/README.md](data/README.md):

```bash
uv sync --extra torch
uv run python -m conformal_fault_inference_with_abstention
```

Sin GPU ni PyTorch: `uv sync` y `uv run python -m conformal_fault_inference_with_abstention
--models hgb --benchmark 0`. Con `--help` se listan las opciones; los valores por
defecto son los de `artifacts/run_2026-09-11_seed42`, que está versionada. El mismo pipeline está
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

## Notebook de análisis

[notebooks/analisis_resultados.ipynb](notebooks/analisis_resultados.ipynb) lee la
ejecución guardada en `artifacts/run_2026-09-11_seed42`, muestra las tablas y figuras,
recorre a mano el cálculo del cuantil conforme con probabilidades inventadas, reconstruye las métricas desde las
probabilidades exportadas y genera la figura de error entre decisiones automáticas en
`artifacts/<ejecución>/figures/`. Está guardado con sus salidas; para volver a
ejecutarlo hace falta la ejecución en `artifacts/`:

```bash
uv run --with jupyterlab jupyter lab notebooks/analisis_resultados.ipynb
```
