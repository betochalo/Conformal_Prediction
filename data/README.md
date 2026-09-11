# Datos

- `raw/`: archivo original AI4I, sin modificaciones.
- `processed/`: etiquetas, características y particiones derivadas.

Fuente: https://archive.ics.uci.edu/dataset/601/ai4i

Datos descargados mediante `fetch_ucirepo(id=601)` de `ucimlrepo` el
2026-09-11 (UTC). Licencia del dataset: CC BY 4.0, según UCI.

- `raw/ai4i2020.csv`: CSV de UCI conservado sin modificar sus bytes; 10 000 filas,
  14 columnas y ningún valor faltante.
- `raw/metadata.json`: metadatos devueltos por UCI.
- `raw/variables.csv`: roles, tipos y unidades de las variables.
- `raw/manifest.json`: URL, fecha UTC, columnas y SHA-256 de la descarga.

URL del CSV: https://archive.ics.uci.edu/static/public/601/data.csv

SHA-256: `ffc3c28f35b1a1636eacd2bf46aca0ff5e795b4a8eb934309c5819079e9b5a18`.

La versión de la API usa nombres sin unidades, como `Air temperature` y `Torque`;
las unidades están en `variables.csv`. `X` tiene seis predictores; `y` contiene
seis indicadores binarios, todavía no la etiqueta multiclase del proyecto.

Para cargar los datos locales sin volver a descargarlos:

```python
import pandas as pd

df = pd.read_csv("data/raw/ai4i2020.csv")
variables = pd.read_csv("data/raw/variables.csv")
X = df[variables.loc[variables["role"].eq("Feature"), "name"]]
y = df[variables.loc[variables["role"].eq("Target"), "name"]]
```

No usar identificadores ni indicadores de falla como
predictores. Auditar RNF y fallas simultáneas antes de definir el objetivo multiclase.
Los archivos de datos se excluyen del control de versiones.

## Reporte obligatorio antes de entrenar

Completar con datos reales después de aplicar las reglas de etiquetas. Los guiones
significan pendiente, no cero. Guardar el reporte final en documentación versionada.

| Clase | Entrenamiento | Calibración | Prueba | Total |
|---|---:|---:|---:|---:|
| Normal | — | — | — | — |
| TWF | — | — | — | — |
| HDF | — | — | — | — |
| PWF | — | — | — | — |
| OSF | — | — | — | — |

Documentar también registros originales, exclusiones, casos con RNF, fallas
simultáneas, regla de prioridad, semilla y procedimiento de partición. Los conteos
de indicadores originales no sustituyen los de las etiquetas finales.

Para Mondrian, con `n_k` ejemplos de calibración de la clase k, el rango requerido
es `ceil((n_k + 1) * (1 - alpha))`. Si supera `n_k`, usar el umbral conservador
infinito: esa clase se incluye siempre. Con alpha = 0.05, hacen falta al menos
19 ejemplos por clase para que el rango no exceda la muestra; esto es un mínimo
aritmético, no una garantía de estimación precisa ni de conjuntos pequeños.

Reportar esta revisión para cada alpha previsto. Si no hay ejemplos de una clase
en prueba, su cobertura es no estimable, no cero. Si faltan en entrenamiento,
revisar la partición antes de ajustar el clasificador.
