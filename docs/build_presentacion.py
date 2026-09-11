"""Genera docs/Presentacion_Proyecto_Final.pptx. Ejecutar desde la raíz con:

    uv run --with python-pptx python docs/build_presentacion.py
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(".")
FIG = ROOT / "artifacts" / "run_2026-09-11_seed42" / "figures"
OUT = ROOT / "docs" / "Presentacion_Proyecto_Final.pptx"

INK = RGBColor(0x0B, 0x0B, 0x0B)
MUTED = RGBColor(0x52, 0x51, 0x4E)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
BAND = RGBColor(0x1C, 0x5C, 0xAB)
SURFACE = RGBColor(0xFC, 0xFC, 0xFB)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = prs.slide_width, prs.slide_height


def band(slide, title, minutes):
    shape = slide.shapes.add_shape(1, 0, 0, W, Inches(1.05))
    shape.fill.solid()
    shape.fill.fore_color.rgb = BAND
    shape.line.fill.background()
    tf = shape.text_frame
    tf.margin_left = Inches(0.5)
    tf.margin_right = Inches(2.4)
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.alignment = PP_ALIGN.LEFT
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    tag = slide.shapes.add_textbox(W - Inches(2.2), Inches(0.3), Inches(1.9), Inches(0.5))
    q = tag.text_frame.paragraphs[0]
    q.text = f"{minutes} min"
    q.alignment = PP_ALIGN.RIGHT
    q.font.size = Pt(14)
    q.font.color.rgb = RGBColor(0xDD, 0xE6, 0xF5)


def bullets(slide, items, left=0.6, top=1.35, width=12.1, height=5.6, size=20):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        level = 0
        text = item
        if item.startswith("  "):
            level = 1
            text = item.strip()
        p.text = ("• " if level == 0 else "– ") + text
        p.level = level
        p.font.size = Pt(size if level == 0 else size - 3)
        p.font.color.rgb = INK if level == 0 else MUTED
        p.space_after = Pt(8)
    return box


def table(slide, rows, left, top, width, col_widths=None, size=13, bold_first_col=False):
    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, Inches(left), Inches(top), Inches(width), Inches(0.35 * n_rows))
    tbl = shape.table
    if col_widths:
        for j, cw in enumerate(col_widths):
            tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(value)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(size)
                p.font.bold = i == 0 or (bold_first_col and j == 0)
                p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else INK
                if j > 0:
                    p.alignment = PP_ALIGN.RIGHT
            cell.fill.solid()
            cell.fill.fore_color.rgb = BAND if i == 0 else (SURFACE if i % 2 else RGBColor(0xF1, 0xF3, 0xF7))
    return shape


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def picture(slide, name, left, top, width):
    slide.shapes.add_picture(str(FIG / name), Inches(left), Inches(top), width=Inches(width))


def caption(slide, text, left, top, width, size=13):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.6))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = MUTED


# 1 Portada ------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(1, 0, 0, W, H)
bg.fill.solid()
bg.fill.fore_color.rgb = BAND
bg.line.fill.background()
box = s.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(4.5))
tf = box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "IA industrial consciente del riesgo"
p.font.size = Pt(44)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
p = tf.add_paragraph()
p.text = "Predicción conforme y abstención selectiva para diagnóstico de fallas"
p.font.size = Pt(26)
p.font.color.rgb = RGBColor(0xDD, 0xE6, 0xF5)
p = tf.add_paragraph()
p.text = " "
p = tf.add_paragraph()
p.text = "Roberth Chachalo · Jaime Astudillo"
p.font.size = Pt(22)
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
p = tf.add_paragraph()
p.text = "MSDS 6014 · Matemáticas y Programación IA · Proyecto final · septiembre 2026"
p.font.size = Pt(16)
p.font.color.rgb = RGBColor(0xDD, 0xE6, 0xF5)
notes(s, "0:00-0:30. Presentar el título y la pregunta central: no si se puede clasificar fallas, "
         "sino cuándo un sistema debe decidir solo y cuándo abstenerse. Ambos integrantes se presentan.")

# 2 Problema -----------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
band(s, "El problema: costos asimétricos y probabilidades sin garantía", 1.5)
bullets(s, [
    "Detener una línea sin motivo es caro; no detectar una falla es mucho más caro; confundir el modo envía al técnico equivocado.",
    "Un clasificador siempre responde la clase más probable. Su probabilidad es una salida del modelo, no una garantía.",
    "Con 96.7 % de ciclos normales, esa probabilidad es optimista justo en las fallas, que son lo que importa.",
    "Pregunta del proyecto: ¿cuándo debe el sistema actuar solo y cuándo pedir intervención humana?",
    "Respuesta propuesta: conjuntos de predicción con garantía estadística y una política de tres vías por su tamaño.",
])
notes(s, "0:30-2:00. Contar el escenario de planta. Enfatizar que la probabilidad del modelo no es calibrada y que el "
         "desbalance hace que el error se concentre en las clases minoritarias.")

# 3 Tres vías ----------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
band(s, "La política de tres vías", 1.0)
table(s, [
    ["Tamaño del conjunto C(x)", "Vía", "Interpretación operativa"],
    ["|C(x)| = 1", "Automática", "Evidencia suficiente; el sistema actúa solo"],
    ["|C(x)| = 2", "Asistida", "Diagnóstico parcial útil: 'es HDF u OSF, no otra cosa'"],
    ["|C(x)| > 2", "Humana", "Sin evidencia discriminante; escalamiento"],
    ["|C(x)| = 0", "Humana", "Observación atípica para todas las clases"],
], 0.8, 1.5, 11.7, [3.2, 2.2, 6.3], size=16)
bullets(s, [
    "El conjunto de dos etiquetas no es un fracaso: descarta el resto del espacio de diagnóstico con garantía.",
    "La abstención es la fracción de ciclos con tamaño distinto de 1: la carga que el sistema traslada a personas.",
], top=3.9, size=18)
notes(s, "2:00-3:00. Explicar la tabla fila por fila. Insistir en que el caso de dos etiquetas es información accionable.")

# 4 Datos --------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
band(s, "Datos: AI4I 2020 y lo que la auditoría encontró", 1.5)
table(s, [
    ["Hallazgo", "Conteo", "Decisión"],
    ["Registros", "10 000", "6 predictores; IDs solo para trazabilidad"],
    ["Machine failure = 1", "339", "330 con modo físico"],
    ["Fallas sin ningún modo", "9", "Excluidas: no admiten etiqueta"],
    ["RNF puro (todos con Machine failure = 0)", "18", "Excluidas: ruido sin firma física"],
    ["Modos físicos simultáneos", "23", "Prioridad por escasez: TWF > PWF > OSF > HDF"],
    ["Temperatura del aire, autocorrelación lag 1", "0.999", "Partición aleatoria; garantía no temporal"],
], 0.6, 1.4, 12.1, [4.6, 1.5, 6.0], size=14)
table(s, [
    ["Clase", "Entren.", "Calib.", "Prueba", "Total"],
    ["Normal", "4 821", "2 893", "1 929", "9 643"],
    ["TWF", "23", "14", "9", "46"],
    ["HDF", "53", "32", "21", "106"],
    ["PWF", "47", "28", "19", "94"],
    ["OSF", "42", "25", "17", "84"],
], 0.6, 4.35, 6.0, [1.6, 1.1, 1.1, 1.1, 1.1], size=13, bold_first_col=True)
caption(s, "Partición estratificada 0.50 / 0.30 / 0.20, semilla 42. TWF con 14 ejemplos de calibración es la clase "
           "que limita el nivel de confianza alcanzable: con α = 0.05 hacen falta al menos 19.", 6.9, 4.4, 5.8, size=14)
notes(s, "3:00-4:30. Es el criterio de calidad de datos de la rúbrica. Mencionar que RNF no activa Machine failure, "
         "que la prioridad se eligió para no restar ejemplos a las clases escasas, y por qué no se usa partición temporal.")

# 5 Método -------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
band(s, "Método: split conformal y Mondrian desde cero en NumPy", 1.5)
bullets(s, [
    "Puntuación de no conformidad: s(x, y) = 1 − p(y | x), con el modelo ya entrenado.",
    "Umbral: estadístico de orden de rango ⌈(n + 1)(1 − α)⌉ de las puntuaciones de calibración, sin interpolación.",
    "Conjunto: C(x) = { y : s(x, y) ≤ q̂ }. Si el rango supera n, el umbral es infinito y la clase entra siempre.",
    "Garantía: si calibración y nuevo caso son intercambiables, P(Y ∈ C(X)) ≥ 1 − α. No depende del modelo.",
    "Split: un umbral global. Mondrian: un umbral por clase, calibrado solo con los ejemplos de esa clase.",
    "  Ejemplo real: split, hgb, α = 0.05 → n = 2 992, rango 2 844, umbral 0.0019.",
    "  Mondrian, TWF, α = 0.05 → n = 14, rango 15 > 14 → umbral infinito.",
    "Calibración separada del entrenamiento y del bloque de prueba. Nada se selecciona con prueba.",
])
notes(s, "4:30-6:30. Explicar el cuantil con el ejemplo numérico. Subrayar que la garantía es marginal y que "
         "Mondrian aplica el mismo argumento dentro de cada clase, a cambio de calibrar con pocas muestras.")

# 6 Modelos y arquitectura ---------------------------------------------------
s = prs.slides.add_slide(BLANK)
band(s, "Dos clasificadores, una capa conforme, un paquete reproducible", 1.0)
bullets(s, [
    "HistGradientBoosting (base aprobado) y un MLP 64-64 en PyTorch entrenado en RTX 5070 Ti (extensión).",
    "Ambos exponen fit, predict_proba y classes_; la capa conforme no distingue cuál la alimenta.",
    "Preprocesamiento ajustado solo con entrenamiento: one-hot de Type y escalado numérico para el MLP.",
    "Paquete instalable con uv, 112 pruebas (cuantil, empates, clases ausentes, particiones, métricas, integración).",
    "Cada ejecución guarda hash de datos, índices, umbrales, probabilidades, métricas, figuras y versiones.",
    "Reproducir: uv sync --extra torch && uv run python -m conformal_fault_inference_with_abstention",
], size=19)
notes(s, "6:30-8:00. Criterio de código reproducible. Mencionar el manifiesto y que las pruebas incluyen "
         "casos calculados a mano.")

# 7 Resultado 1: cobertura marginal -----------------------------------------
s = prs.slides.add_slide(BLANK)
band(s, "Resultado 1: coberturas cerca del nominal con ambos modelos", 1.0)
table(s, [
    ["Modelo", "Variante", "α", "Cobertura marginal", "Tamaño medio", "Abstención"],
    ["hgb", "split", "0.10", "0.909", "0.91", "0.088"],
    ["hgb", "mondrian", "0.10", "0.917", "1.32", "0.305"],
    ["mlp", "split", "0.10", "0.902", "0.90", "0.096"],
    ["mlp", "mondrian", "0.10", "0.903", "1.07", "0.085"],
    ["hgb", "split", "0.05", "0.954", "0.96", "0.041"],
    ["hgb", "mondrian", "0.05", "0.959", "2.64", "1.000"],
], 0.6, 1.4, 12.1, [1.4, 1.8, 1.0, 3.0, 2.4, 2.5], size=15)
bullets(s, [
    "En esta partición, las 12 combinaciones quedan a menos de 0.02 del nominal (n = 1 995; desviación típica ≈ 0.007).",
    "Consistente con la garantía, que es una afirmación en probabilidad bajo intercambiabilidad; una partición la ilustra, no la demuestra.",
    "Árbol y MLP difieren en tamaño y abstención, no en cobertura: la incertidumbre se traduce en tamaño.",
], top=4.3, size=18)
notes(s, "8:00-9:00. Leer dos filas. La cobertura es la restricción a cumplir, no el objetivo a maximizar.")

# 8 Resultado 2: por clase ---------------------------------------------------
s = prs.slides.add_slide(BLANK)
band(s, "Resultado 2: split ignora las fallas; Mondrian las cubre", 2.0)
picture(s, "coverage_by_class_hgb.png", 0.5, 1.25, 12.3)
caption(s, "Con α = 0.10 y hgb: cobertura marginal 0.909, pero TWF y PWF con 0.000 y HDF con 0.143. Normal absorbe la "
           "garantía. Mondrian lleva las cuatro fallas a 0.95–1.00. Con α = 0.05, TWF entra siempre por umbral infinito.",
        0.6, 5.0, 12.0, size=15)
notes(s, "9:00-11:00. Esta es la diapositiva central. Señalar las barras azules en cero y las naranjas sobre la línea. "
         "Explicar que la barra de TWF en 0.20 es 7 de 9 casos.")

# 9 Resultado 3: precio ------------------------------------------------------
s = prs.slides.add_slide(BLANK)
band(s, "Resultado 3: el precio de la garantía es el tamaño del conjunto", 1.0)
picture(s, "global_metrics_hgb.png", 0.5, 1.25, 12.3)
caption(s, "Mondrian con α = 0.05 abstiene el 100 %: TWF acompaña a Normal en casi todos los conjuntos. Con α = 0.10 "
           "resuelve el 69.5 % de forma automática y deriva el 28.6 % a la vía asistida. Split solo produce vía humana "
           "por conjuntos vacíos, no porque distinga fallas.", 0.6, 5.0, 12.0, size=14)
table(s, [
    ["Error entre decisiones automáticas, α = 0.10", "Automáticas", "Erróneas", "Tasa", "Tipo de error dominante"],
    ["hgb · split", "1 819", "5", "0.3 %", "5 fallas enviadas como Normal"],
    ["hgb · mondrian", "1 387", "82", "5.9 %", "81 normales enviados como falla"],
    ["mlp · mondrian", "1 825", "150", "8.2 %", "149 normales enviados como falla"],
], 0.6, 5.85, 12.1, [4.0, 1.5, 1.3, 1.1, 4.2], size=12)
notes(s, "11:00-12:00. Conectar con la política de tres vías: la abstención es carga de trabajo humana medible. "
         "La cobertura no garantiza exactitud entre las automáticas: Mondrian da falsas alarmas; split omite fallas.")

# 10 MLP y GPU ---------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
band(s, "Entrenamiento del MLP: traza y CPU frente a GPU", 1.5)
picture(s, "training_trace_mlp.png", 0.4, 1.2, 8.3)
table(s, [
    ["Dispositivo", "Mediana de 3 (s)", "Pérdida final"],
    ["cpu", "5.0", "0.0209"],
    ["cuda", "6.8", "0.0209"],
], 8.9, 1.3, 4.1, [1.5, 1.5, 1.1], size=13)
bullets(s, [
    "5 061 parámetros y lotes de 256: el costo de lanzar kernels supera al cálculo. La GPU no gana; se mide, no se asume.",
    "CPU y CUDA no coinciden a 1e-4 tras 200 épocas (diferencia 0.027 en parámetros), pero la clase coincide en 99.98 %.",
    "Exactitud de entrenamiento 0.995 frente a 0.967 de responder siempre Normal; TWF baja de 0.22 a 0.13.",
], left=0.6, top=3.9, width=12.1, size=16)
notes(s, "12:00-13:00. Traza y benchmark: ciclo de entrenamiento, traza de pérdida y gradiente, float32 y "
         "comparación de dispositivos con mediana de repeticiones, como en el ejercicio de la semana 3.")

# 10b Conceptos del curso -----------------------------------------------------
s = prs.slides.add_slide(BLANK)
band(s, "Conceptos del curso aplicados", 1.0)
table(s, [
    ["Concepto", "Dónde se usa en el proyecto"],
    ["Tensores, capas lineales, activaciones", "MLP con nn.Linear y ReLU sobre lotes de 256 filas en GPU"],
    ["Autodiferenciación y ciclo de entrenamiento", "backward(), zero_grad(), Adam; traza de pérdida, norma del gradiente y cambio de parámetros"],
    ["CPU frente a GPU, precisión float32", "Mediana de repeticiones, parámetros comparados con tolerancia, costo de transferencias"],
    ["Cuantiles, estadísticos de orden, probabilidad", "Rango ⌈(n + 1)(1 − α)⌉ y argumento de intercambiabilidad del cuantil conforme"],
    ["Comparación de modelos y métricas por clase", "HGB frente a MLP con la misma capa conforme; cobertura, tamaño, abstención, error automático"],
    ["Pipelines y fuga de información", "Preprocesamiento ajustado solo con entrenamiento; calibración y prueba separadas"],
    ["Reproducibilidad", "uv.lock, semillas, hash SHA-256 del CSV, manifiesto con versiones y GPU, 112 pruebas"],
], 0.6, 1.4, 12.1, [4.0, 8.1], size=14)
caption(s, "Fuera del alcance reducido: Docker, SQL, SVD y autovalores, embeddings aprendidos. Tema adicional investigado: "
           "predicción conforme (Vovk et al. 2005; Lei et al. 2018; Angelopoulos y Bates 2021).", 0.6, 4.6, 12.0, size=14)
notes(s, "13:00-14:00. Recorrer la tabla de arriba abajo en 30 segundos por bloque. Citar las referencias del informe.")

# 11 Propuesta vs entregado --------------------------------------------------
s = prs.slides.add_slide(BLANK)
band(s, "Propuesta frente a entregado", 0.5)
table(s, [
    ["Objetivo de la propuesta", "Estado"],
    ["Fundamento matemático y cobertura marginal", "Cumplido"],
    ["Split conformal y Mondrian en NumPy; APS", "Cumplido; APS pospuesto"],
    ["Cobertura independiente del clasificador", "Adaptado: MLP en lugar de GaussianNB"],
    ["Split subcubre minoritarias, Mondrian corrige", "Cumplido y cuantificado"],
    ["Ablación física y política de costos", "Pospuestas por observación docente"],
    ["Paquete instalable y ejecución reproducible", "Cumplido"],
], 0.8, 1.4, 11.7, [7.2, 4.5], size=15)
notes(s, "14:00-14:30. Dejar explícito el recorte de alcance aprobado y qué queda como extensión.")

# 12 Conclusiones ------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
band(s, "Conclusiones y recomendaciones", 1.0)
bullets(s, [
    "Coberturas consistentes con la garantía y sin depender del clasificador; la calidad del modelo se ve en el tamaño.",
    "La cobertura marginal es insuficiente para diagnóstico de fallas: puede cumplirse ignorando todas las fallas.",
    "Mondrian con α = 0.10: garantía por clase, 70–90 % de decisiones automáticas, con 6–8 % de falsas alarmas entre ellas.",
    "La abstención es el precio de la garantía y depende de los datos escasos: TWF fija el α alcanzable.",
    "Recomendaciones: repetir particiones con varias semillas; reunir más TWF; pasar a APS para evitar conjuntos vacíos;",
    "  y definir la política de costos para elegir α por costo esperado, no solo por cobertura.",
    "Límites: una partición, datos sintéticos, dependencia temporal no evaluada, sin selección de hiperparámetros.",
])
notes(s, "14:30-15:00. Cerrar con la pregunta inicial respondida y las recomendaciones. Las preguntas van después del límite.")

OUT.parent.mkdir(exist_ok=True)
prs.save(OUT)
print("guardado", OUT, "diapositivas:", len(prs.slides))
