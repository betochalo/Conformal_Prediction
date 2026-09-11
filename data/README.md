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

## Auditoría de indicadores (Etapa 1)

Resultado de `audit_failures(load_raw("data/raw/ai4i2020.csv"))` sobre las
10 000 filas originales. `load_raw` valida columnas, ausencia de faltantes,
unicidad de UID y Product ID y flags binarios; devuelve el DataFrame indexado por UID.

| Métrica | Conteo |
|---|---:|
| Filas | 10 000 |
| Machine failure = 1 | 339 |
| Indicador TWF | 46 |
| Indicador HDF | 115 |
| Indicador PWF | 95 |
| Indicador OSF | 98 |
| Indicador RNF | 19 |
| Filas con al menos un modo físico | 330 |
| Filas con dos o más modos físicos | 23 |
| TWF+OSF / HDF+PWF / HDF+OSF / PWF+OSF / TWF+PWF+OSF | 2 / 3 / 6 / 11 / 1 |
| RNF sin modo físico | 18 |
| RNF con modo físico | 1 |
| Machine failure = 1 sin ningún modo ni RNF | 9 |
| Modo físico con Machine failure = 0 | 0 |
| RNF con Machine failure = 0 | 18 |
| Normal estricto (todo apagado) | 9 643 |

Hallazgos que condicionan las etiquetas:

- Los 18 registros con RNF puro tienen `Machine failure = 0`: en este CSV el
  indicador aleatorio no activa la falla global. Confirma que RNF no tiene firma
  física y que tratarlo como Normal contaminaría esa clase; se excluye.
- Hay 9 registros con `Machine failure = 1` y ningún modo. No pueden recibir una
  etiqueta de modo; se excluyen y se reportan.
- Los 23 solapamientos físicos requieren la regla de prioridad. La suma de
  indicadores (354) no es el número de fallas etiquetadas (330 filas con modo).
- No hay modos activos con `Machine failure = 0`, así que la inconsistencia
  contemplada en `build_labels` no ocurre en estos datos.

## Política de etiquetas propuesta

```python
LabelPolicy(priority=("TWF", "PWF", "OSF", "HDF"), rnf_policy="exclude_only_rnf")
```

- Prioridad por escasez: ante solapamiento gana el modo con menos registros en
  el dataset (TWF 46 < PWF 95 < OSF 98 < HDF 115). Así los solapamientos no
  restan ejemplos a las clases que ya tienen menos calibración disponible, lo cual
  importa para Mondrian. La alternativa opuesta (HDF, OSF, PWF, TWF) deja TWF con 43.
- `exclude_only_rnf`: se excluyen los 18 RNF sin modo físico; el único registro
  con RNF y modo físico conserva su etiqueta física. `exclude_rows` lo excluiría
  también y solo cambia TWF de 46 a 45.

Distribución final con la política propuesta (9 973 filas conservadas, 27 excluidas:
9 fallas sin modo y 18 RNF puros):

| Clase | Total |
|---|---:|
| Normal | 9 643 |
| TWF | 46 |
| HDF | 106 |
| PWF | 94 |
| OSF | 84 |

Esta decisión debe confirmarse por el equipo antes de la Etapa 2; cambiarla solo
requiere otro `LabelPolicy`, sin tocar el código.

## Reporte obligatorio antes de entrenar (Etapa 2)

Conteos reales con la política propuesta, `split_data(calibration_size=0.30,
test_size=0.20, random_state=42)`. Las fracciones son respecto al total: primero
se aparta prueba y luego calibración del remanente, ambas estratificadas por clase.
Cada fila aparece en exactamente un bloque; los índices UID se conservan.

| Clase | Entrenamiento | Calibración | Prueba | Total |
|---|---:|---:|---:|---:|
| Normal | 4 821 | 2 893 | 1 929 | 9 643 |
| TWF | 23 | 14 | 9 | 46 |
| HDF | 53 | 32 | 21 | 106 |
| PWF | 47 | 28 | 19 | 94 |
| OSF | 42 | 25 | 17 | 84 |
| Total | 4 986 | 2 992 | 1 995 | 9 973 |

Registros originales: 10 000. Exclusiones: 9 fallas sin modo y 18 RNF sin modo
físico (27 en total). El registro con RNF y modo físico se conserva con su modo.
Regla de prioridad: TWF > PWF > OSF > HDF (ver sección anterior).

### Comparación de fracciones candidatas (semilla 42)

| Calibración / Prueba | TWF entren. / calib. / prueba | Umbral infinito con alpha 0.05 | Con 0.10 | Con 0.20 |
|---|---|---|---|---|
| 0.20 / 0.20 | 28 / 9 / 9 | TWF, OSF | ninguna | ninguna |
| **0.30 / 0.20** | 23 / 14 / 9 | TWF | ninguna | ninguna |
| 0.40 / 0.20 | 19 / 18 / 9 | TWF | ninguna | ninguna |
| 0.40 / 0.15 | 21 / 18 / 7 | TWF | ninguna | ninguna |
| 0.45 / 0.15 | 18 / 21 / 7 | ninguna | ninguna | ninguna |

Las demás clases tienen umbral finito en todas las combinaciones para alpha ≤ 0.10.
TWF es la restricción: con 46 registros, alcanzar los 19 de calibración que exige
alpha = 0.05 obliga a poner el 45 % de los datos en calibración, dejando 18 para
entrenar y 7 para medir la cobertura de esa clase. Ese reparto hace que la cobertura
por clase de TWF sea prácticamente no medible en prueba.

### Decisión propuesta

- Partición 0.30 / 0.20 con semilla 42.
- Alphas prefijados: 0.05, 0.10 y 0.20, todos reportados; ninguno se elige con prueba.
- Con alpha = 0.05, Mondrian asigna umbral infinito a TWF: esa clase entra siempre
  en el conjunto y su cobertura es 1 por construcción. No es un error, es el
  comportamiento conservador de la garantía cuando la calibración es escasa, y se
  reportará como tal. Con alpha = 0.10 todas las clases tienen umbral finito.
- No se duplican ni generan ejemplos de calibración. El bloque de prueba queda
  reservado hasta la evaluación final.

Esta decisión debe confirmarse por el equipo antes de entrenar.

### Viabilidad por alpha (calibración 0.30 / prueba 0.20)

`calibration_feasibility(splits, alpha)` calcula `rank = ceil((n_k + 1)(1 - alpha))`.

| Clase | n_calib | rank 0.05 | finito | rank 0.10 | finito | rank 0.20 | finito |
|---|---:|---:|---|---:|---|---:|---|
| Normal | 2 893 | 2 750 | sí | 2 605 | sí | 2 316 | sí |
| TWF | 14 | 15 | **no** | 14 | sí | 12 | sí |
| HDF | 32 | 32 | sí | 30 | sí | 27 | sí |
| PWF | 28 | 28 | sí | 27 | sí | 24 | sí |
| OSF | 25 | 25 | sí | 24 | sí | 21 | sí |

Un rango igual a n usa la puntuación máxima de calibración como umbral, por lo que
las clases con pocos ejemplos tendrán umbrales cercanos a 1 y conjuntos amplios.
Esto es un mínimo aritmético, no una garantía de estimación precisa ni de conjuntos
pequeños.

### Esquema de muestreo y dependencia temporal

Se usa muestreo aleatorio estratificado por clase porque la garantía conforme
requiere intercambiabilidad entre calibración y prueba, y una partición aleatoria
es la forma más simple de sostenerla. Estratificar solo controla las proporciones;
no demuestra intercambiabilidad por sí mismo.

El CSV tiene estructura secuencial: la temperatura del aire tiene autocorrelación
de retardo 1 de 0.999 (paseo aleatorio en el proceso generador) y el desgaste de
herramienta crece fila a fila con 119 reinicios. Torque y velocidad no muestran
autocorrelación (0.005 y 0.008). Los indicadores de falla se derivan de umbrales
sobre los valores de la misma fila, no de la historia, así que las parejas (X, y)
son intercambiables bajo mezcla aleatoria aunque las filas consecutivas se parezcan.
La consecuencia práctica es que una partición temporal (entrenar con las primeras
filas, probar con las últimas) tendría un desplazamiento de temperatura ambiente y
no sería equivalente a la aleatoria. El proyecto usa la aleatoria y documenta que
la garantía se afirma para ese esquema, no para despliegue en el tiempo.

Para Mondrian, con `n_k` ejemplos de calibración de la clase k, el rango requerido
es `ceil((n_k + 1) * (1 - alpha))`. Si supera `n_k`, usar el umbral conservador
infinito: esa clase se incluye siempre. Con alpha = 0.05, hacen falta al menos
19 ejemplos por clase para que el rango no exceda la muestra; esto es un mínimo
aritmético, no una garantía de estimación precisa ni de conjuntos pequeños.

Reportar esta revisión para cada alpha previsto. Si no hay ejemplos de una clase
en prueba, su cobertura es no estimable, no cero. Si faltan en entrenamiento,
revisar la partición antes de ajustar el clasificador.
