# Plan de implementación

El alcance es un HistGradientBoosting, split conformal y Mondrian;
las métricas principales son cobertura por clase, tamaño medio y abstención.

## Estado inicial

- [x] Entorno uv, paquete importable y dependencias.
- [x] AI4I descargado con metadatos y hash.
- [x] Contratos de datos y firmas de las funciones.
- [ ] Auditoría, política de etiquetas y conteos reales por partición.
- [ ] Modelo, algoritmos conformes, métricas y ejecución integrada.

Las funciones pendientes lanzan `NotImplementedError`: las firmas son acuerdos
de integración, no algoritmos ya implementados. Los tipos no validan los datos
automáticamente; cada implementación debe comprobar su contrato.

## Reparto propuesto

Persona A y Persona B son roles intercambiables; asignen sus nombres al comenzar.

| Etapa | Responsable | Revisión | Archivos principales |
|---|---|---|---|
| 1. Datos y etiquetas | A | B | `data.py`, `tests/test_data.py` |
| 2. Particiones y viabilidad | A | B | `splitting.py`, `tests/test_splitting.py` |
| 3. Modelo base | A | B | `model.py`, `tests/test_model.py` |
| 4. Split conformal | B | A | `scores.py`, `conformal.py`, pruebas respectivas |
| 5. Mondrian | B | A | `conformal.py`, `tests/test_conformal.py` |
| 6. Decisiones y métricas | A | B | `decision.py`, `evaluation.py`, pruebas respectivas |
| 7. Integración y resultados | A integra; B revisa resultados | Ambos | `pipeline.py`, `tests/test_pipeline.py` |
| 8. Explicación y entrega | Ambos | Ambos | README, reporte final y ejemplos |

A puede trabajar en datos mientras B desarrolla puntuaciones y cuantiles con
ejemplos inventados. La evaluación real espera a que ambos trabajos estén listos.
No editar simultáneamente el mismo archivo; acordar cambios a `contracts.py` antes
de modificar consumidores. Cada cambio debe incluir una explicación y sus pruebas.

## Etapa 1 — Auditar antes de etiquetar

Implementar `load_raw`, `audit_failures` y luego `build_labels`.

- Confirmar esquema, faltantes, IDs y flags binarios.
- Contar cada modo, RNF solo, modos simultáneos e inconsistencias con Machine failure.
- Elegir y justificar `LabelPolicy`: tratamiento de RNF y prioridad de modos físicos.
  Las opciones propuestas deben revisarse con la auditoría, no elegirse por inercia.
- Registrar exclusiones y distribución final. Conservar índices originales para
  rastrear filas; X solo contiene los seis predictores.

Cierre: reporte en `data/README.md` y pruebas con filas pequeñas que cubran Normal,
RNF, solapamientos y fallos sin modo. No confundir conteos de indicadores con
conteos de las clases finales.

## Etapa 2 — Revisar cuánta calibración queda

Implementar `split_data`, `class_counts` y `calibration_feasibility`.

- Comparar proporciones candidatas mediante conteos, antes de entrenar.
- Fijar semilla y fracciones respecto al total; guardar índices disjuntos.
- Completar la tabla Normal/TWF/HDF/PWF/OSF × entrenamiento/calibración/prueba.
- Para cada alpha previsto, reportar n por clase y rango del cuantil. Un rango
  mayor que n implica umbral infinito; no corregirlo recortando el rango.
- Justificar el esquema de muestreo y discutir la dependencia temporal potencial
  del dataset. Estratificar no demuestra por sí solo intercambiabilidad.

Cierre: tabla real y decisión documentada sobre particiones y alpha. No entrenar
sin revisar este reporte. Probar disjunción, conservación de filas, reproducibilidad
y errores ante tamaños inviables. No sobremuestrear calibración ni prueba.

## Etapa 3 — Un único clasificador

Implementar `build_model` como Pipeline con codificación de Type y
HistGradientBoostingClassifier. Ajustarlo solo en entrenamiento.

Cierre: `predict_proba` devuelve una matriz finita con filas que suman uno;
`classes_` identifica exactamente el orden de columnas. Probar el manejo de Type
desconocido y que el preprocesamiento no se ajusta con calibración o prueba.
No comparar modelos ni hacer búsqueda extensa de hiperparámetros.

## Etapa 4 — Split conformal desde cero

Implementar `inverse_probability`, `true_label_scores`, `conformal_quantile` y
`SplitConformal`.

- Calcular a mano un ejemplo antes de programar.
- Usar rango `ceil((n+1)*(1-alpha))`, sin interpolación.
- Con rango mayor que n, devolver infinito; conservar empates con `<=`.
- Rechazar probabilidades inválidas, etiquetas desconocidas y uso sin calibrar.

Cierre: pruebas de resultados numéricos conocidos, alpha inválido, empates,
muestra vacía, orden de clases y conjuntos vacíos/completos. La cobertura de una
única muestra finita no es un criterio exacto de éxito de una prueba estadística.

## Etapa 5 — Mondrian por clase

Implementar `MondrianConformal` reutilizando puntuaciones y cuantil. Agrupar la
calibración por etiqueta verdadera, no por predicción. Mantener umbrales alineados
con `classes_`; una clase sin calibración recibe infinito.

Cierre: ejemplo manual con umbrales diferentes y prueba de clase ausente. Comprobar
que el umbral de cada candidata se usa al predecir. Documentar por qué la cobertura
marginal puede ocultar problemas de las clases minoritarias.

## Etapa 6 — Medir conjuntos y abstención

Implementar `route_predictions` y `evaluate_sets`.

- Cobertura: fracción de conjuntos que incluyen la etiqueta verdadera.
- Tamaño: número de etiquetas incluidas por fila.
- Automática: tamaño 1; asistida: tamaño 2; humana: tamaño 0 o mayor que 2.
- Abstención: fracción con tamaño distinto de 1.
- Cobertura por clase: incluir denominador; sin casos, reportar NaN, no cero.

Cierre: pruebas con conjuntos construidos a mano de tamaños 0, 1, 2 y mayores;
verificar conteos y métricas. No interpretar cobertura global como exactitud
condicionada a decisiones automáticas.

## Etapa 7 — Integrar desde el paquete

Implementar `run_pipeline(RunConfig(...))` como función importable, sin carpetas
de notebooks ni de experimentos. La secuencia será:

```text
CSV → auditoría → etiquetas → particiones → reporte previo
    → modelo entrenado → probabilidades de calibración y prueba
    → split / Mondrian → conjuntos → métricas y figuras
```

Entrenar una sola vez y compartir modelo, particiones y alpha entre variantes.
Guardar una ejecución en `artifacts/<nombre_de_ejecucion>/` con configuración,
hash de datos, versiones, índices, auditoría, conteos, umbrales, métricas y figuras.
No elegir el mejor alpha con prueba ni prometer que Mondrian siempre mejora tamaño.

Cierre: prueba de integración pequeña con datos sintéticos y una ejecución real
reproducible. Incorporar el ejemplo de llamada al README cuando la función funcione.

## Etapa 8 — Entregar y explicar

- Comparar ambas variantes por alpha con cobertura por clase, tamaño y abstención.
- Explicar puntuaciones, cuantil, intercambiabilidad y cobertura marginal/por clase.
- Discutir escasez de calibración, desbalance, dependencia y carácter sintético.
- Registrar qué hipótesis se apoyan en resultados y cuáles no.
- Comprobar instalación y ejecución desde un entorno limpio; generar paquete.

APS, costos, clases desconocidas, segundo modelo y variables físicas siguen fuera
de la entrega mínima. `features.py` queda reservado y no requiere implementación.

## Comprobaciones durante el trabajo

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Crear pruebas al implementar cada etapa; no marcar pruebas vacías como aprobadas.
Hasta tener la primera prueba, pytest devuelve código 5. Antes de cerrar la entrega,
ejecutar también `uv build` y verificar el ejemplo de ejecución documentado.
