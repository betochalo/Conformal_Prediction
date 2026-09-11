# Pruebas

Agregar pruebas junto con la implementación de cada componente:

Seguir las etapas y criterios de [PLAN.md](../PLAN.md). Archivos implementados:
`test_data.py`, `test_splitting.py`, `test_model.py`, `test_scores.py`,
`test_conformal.py`, `test_decision.py`, `test_evaluation.py`, `test_pipeline.py`
(integración con un CSV sintético de 900 filas), `test_cli.py` (punto de entrada y
protección contra sobrescritura) y `test_torch_model.py` (se omite sin el extra
torch; la prueba de CUDA se omite sin GPU).
Usar fixtures pequeñas y sintéticas para las pruebas unitarias, sin descargas.
Las funciones aún pendientes no deben tener pruebas ficticias que den éxito.

- Cuantil de rango corregido, empates y caso de rango mayor que la muestra.
- Conjuntos calculados a mano para split y Mondrian.
- Orden de clases del estimador y clases sin ejemplos de calibración.
- Particiones disjuntas y ausencia de ajuste con datos de calibración o prueba.
- Cobertura, tamaño y abstención calculados a mano, incluyendo conjuntos vacíos.

Las pruebas de APS, características físicas y costos se agregarán únicamente si
se desarrollan esas extensiones. Una simulación de cobertura, si se incorpora,
debe tener tolerancia estadística explícita; no exigir que cada muestra finita
alcance exactamente la cobertura nominal.

`test_data.py` cubre la Etapa 1 con filas sintéticas: carga validada, auditoría,
prioridad, ambas políticas de RNF y políticas inválidas. `test_splitting.py` cubre
la Etapa 2: disjunción, conservación de filas, estratificación, reproducibilidad,
fracciones inválidas, conteos con ceros y el rango del cuantil por alpha.

Las etapas 3–6 añaden pruebas de entrenamiento sin ajustar durante la predicción,
Type desconocido, puntuaciones y umbrales manuales, empates, clases sin calibración,
recalibración, validación de estado y orden de clases, todas las rutas de decisión
y cobertura no estimable. Total actual: 112 pruebas; no requieren acceso a la red.
