# Diagnóstico de fallas con predicción conforme y abstención

Proyecto de maestría basado en AI4I 2020: convertir las salidas de un clasificador
en conjuntos de diagnósticos y decisiones automáticas, asistidas o humanas.

La [propuesta original aprobada](propuesta_final_roberth_jaime.pdf) es la referencia
conceptual. El alcance de ejecución se reduce según la observación docente:
un clasificador base y dos variantes conformes en dos semanas.
Estado actual: estructura del proyecto; algoritmos y experimentos pendientes.

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
uv sync
uv run ruff check .
```

Cuando existan pruebas, ejecutarlas con `uv run pytest`. Para generar el paquete
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
  pipeline.py      Configuración y función de ejecución integrada
data/
  raw/             Datos originales
  processed/       Datos derivados
tests/             Pruebas matemáticas y de comportamiento
artifacts/         Resultados generados, excluidos de Git
```

`src/` contiene el núcleo importable del proyecto: carga de datos, predicción
conforme y evaluación. La lógica se llamará desde el código que utilice el paquete.
Por ejemplo, una vez instalado con `uv sync`:

```python
from conformal_fault_inference_with_abstention import conformal, data, evaluation
```

Los módulos tienen firmas, tipos y contratos; las funciones pendientes lanzan
`NotImplementedError`. El modelo se entrenará fuera de la capa conforme, que recibe
probabilidades y el orden explícito de las clases. Así ambas variantes comparten
el mismo clasificador. `pipeline.run_pipeline` será el punto de entrada importable
cuando se complete la integración; todavía no ejecuta el proyecto. `tests/` se
reserva para verificar esa lógica y `artifacts/` para guardar métricas y figuras.

La calibración debe permanecer separada del ajuste del modelo. La garantía requiere
intercambiabilidad; no garantiza exactitud individual ni detección de clases
desconocidas. Las conclusiones experimentales se medirán, no se asumirán.
