"""Genera docs/Presentacion_Proyecto_Final.pptx. Ejecutar desde la raíz con:

    uv run --with python-pptx python docs/build_presentacion.py

Estructura para 10 minutos repartidos en dos bloques de 5:
  Bloque 1 (Jaime): problema, política de tres vías, datos y auditoría, método.
  Bloque 2 (Roberth): implementación, resultados, conclusiones.
El apéndice queda como respaldo para preguntas y no se expone.
Cada diapositiva lleva notas de orador con el intervalo de tiempo previsto.
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(".")
FIG = ROOT / "artifacts" / "run_2026-09-11_seed42" / "figures"
OUT = ROOT / "docs" / "Presentacion_Proyecto_Final.pptx"

# Paleta (validada con el validador de dataviz): azul y naranja para series,
# navy para fondos oscuros, tintas neutras para texto.
NAVY = RGBColor(0x0F, 0x2A, 0x4A)
NAVY_2 = RGBColor(0x1C, 0x5C, 0xAB)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
BLUE_SOFT = RGBColor(0xDC, 0xE8, 0xF8)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
ORANGE_SOFT = RGBColor(0xFB, 0xE3, 0xD8)
AQUA = RGBColor(0x1B, 0xAF, 0x7A)
AQUA_SOFT = RGBColor(0xD7, 0xF1, 0xE6)
INK = RGBColor(0x0B, 0x0B, 0x0B)
MUTED = RGBColor(0x52, 0x51, 0x4E)
LIGHT = RGBColor(0x89, 0x87, 0x81)
GRID = RGBColor(0xE1, 0xE0, 0xD9)
SURFACE = RGBColor(0xF7, 0xF7, 0xF5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ROW_ALT = RGBColor(0xEE, 0xF1, 0xF6)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = prs.slide_width, prs.slide_height
SLIDE_NO = {"n": 0}
SCHEDULE: list[tuple[str, float]] = []


# ----------------------------------------------------------------------------- helpers
def rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.shadow.inherit = False
    return s


def text(slide, s, x, y, w, h, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, font="Calibri", italic=False, line_spacing=1.1):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    lines = s if isinstance(s, list) else [s]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.line_spacing = line_spacing
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.italic = italic
        p.font.name = font
        p.font.color.rgb = color
    return box


def shape_text(shape, s, size=14, bold=False, color=INK, align=PP_ALIGN.CENTER,
               anchor=MSO_ANCHOR.MIDDLE):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.08)
    lines = s if isinstance(s, list) else [s]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.font.size = Pt(size if i == 0 else max(size - 3, 9))
        p.font.bold = bold if i == 0 else False
        p.font.name = "Calibri"
        p.font.color.rgb = color


def card(slide, x, y, w, h, title, body, accent=BLUE, fill=SURFACE, title_size=15, body_size=12):
    rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    rect(slide, x, y, 0.09, h, accent, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
    text(slide, title, x + 0.2, y + 0.1, w - 0.3, 0.45, size=title_size, bold=True, color=INK)
    text(slide, body, x + 0.2, y + 0.55, w - 0.3, h - 0.6, size=body_size, color=MUTED)


def stat(slide, x, y, w, value, label, color=BLUE, value_size=34):
    text(slide, value, x, y, w, 0.7, size=value_size, bold=True, color=color)
    text(slide, label, x, y + 0.68, w, 0.5, size=12, color=MUTED)


def arrow(slide, x1, y1, x2, y2, color=LIGHT, width=1.5):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd", {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return c


def bullets(slide, items, x=0.7, y=1.55, w=11.9, h=5.2, size=17, color=INK, gap=7):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        level = 1 if item.startswith("  ") else 0
        p.text = ("•  " if level == 0 else "–  ") + item.strip()
        p.level = level
        p.font.size = Pt(size if level == 0 else size - 3)
        p.font.name = "Calibri"
        p.font.color.rgb = color if level == 0 else MUTED
        p.space_after = Pt(gap)
    return box


def table(slide, rows, x, y, w, col_w=None, size=12, header=NAVY_2, bold_col0=False,
          row_h=0.34, align_first_left=True, highlight_rows=()):
    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y), Inches(w), Inches(row_h * n_rows))
    tbl = shape.table
    if col_w:
        for j, cw in enumerate(col_w):
            tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, value in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(value)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(size)
                p.font.name = "Calibri"
                p.font.bold = i == 0 or (bold_col0 and j == 0)
                p.font.color.rgb = WHITE if i == 0 else INK
                p.alignment = PP_ALIGN.LEFT if (j == 0 and align_first_left) else PP_ALIGN.CENTER
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = header
            elif i in highlight_rows:
                cell.fill.fore_color.rgb = ORANGE_SOFT
            else:
                cell.fill.fore_color.rgb = WHITE if i % 2 else ROW_ALT
    return shape


def picture(slide, name, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(FIG / name), Inches(x), Inches(y), **kw)


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


def slide_base(kicker, title, minutes, dark=False):
    """Diapositiva estándar: kicker de sección, título, tiempo, pie con numeración."""
    s = prs.slides.add_slide(BLANK)
    SLIDE_NO["n"] += 1
    if minutes is not None:
        SCHEDULE.append((title, minutes))
    bg = NAVY if dark else WHITE
    rect(s, 0, 0, 13.333, 7.5, bg)
    rect(s, 0.7, 1.28, 0.6, 0.05, ORANGE)
    text(s, kicker.upper(), 0.7, 0.42, 8, 0.3, size=11, bold=True, color=ORANGE)
    text(s, title, 0.7, 0.62, 11.0, 0.7, size=24, bold=True, color=WHITE if dark else NAVY)
    if minutes is not None:
        text(s, f"{minutes:g} min", 11.9, 0.45, 0.8, 0.3, size=10, color=LIGHT, align=PP_ALIGN.RIGHT)
    rect(s, 0, 7.15, 13.333, 0.35, NAVY if not dark else NAVY_2)
    text(s, "Predicción conforme y abstención selectiva · AI4I 2020", 0.7, 7.19, 8, 0.28, size=9, color=RGBColor(0xC9, 0xD6, 0xE8))
    text(s, f"{SLIDE_NO['n']}", 12.2, 7.19, 0.5, 0.28, size=9, color=RGBColor(0xC9, 0xD6, 0xE8), align=PP_ALIGN.RIGHT)
    return s


def speaker_band(slide, name, block, span):
    """Banda discreta que indica quién habla en esta diapositiva."""
    rect(slide, 11.28, 0.78, 1.3, 0.3, BLUE_SOFT if block == 1 else ORANGE_SOFT,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.3)
    text(slide, name, 11.28, 0.81, 1.3, 0.26, size=9, bold=True,
         color=NAVY_2 if block == 1 else ORANGE, align=PP_ALIGN.CENTER)


def divider(title, subtitle, minutes=None):
    s = prs.slides.add_slide(BLANK)
    SLIDE_NO["n"] += 1
    rect(s, 0, 0, 13.333, 7.5, NAVY)
    rect(s, 0.9, 3.05, 0.9, 0.06, ORANGE)
    text(s, title, 0.9, 3.2, 11, 0.9, size=40, bold=True, color=WHITE)
    text(s, subtitle, 0.9, 4.1, 11, 0.8, size=18, color=RGBColor(0xC9, 0xD6, 0xE8))
    text(s, f"{SLIDE_NO['n']}", 12.2, 7.05, 0.5, 0.3, size=9, color=RGBColor(0x8F, 0xA6, 0xC4), align=PP_ALIGN.RIGHT)
    if minutes is not None:
        SCHEDULE.append((title, minutes))
    return s


# =============================================================================== portada
s = prs.slides.add_slide(BLANK)
SLIDE_NO["n"] += 1
SCHEDULE.append(("Portada", 0.3))
rect(s, 0, 0, 13.333, 7.5, NAVY)
rect(s, 0, 0, 0.35, 7.5, ORANGE)
text(s, "MSDS 6014 · MATEMÁTICAS Y PROGRAMACIÓN IA · PROYECTO FINAL", 1.0, 0.8, 11, 0.4, size=12, bold=True, color=RGBColor(0xC9, 0xD6, 0xE8))
text(s, "IA industrial consciente del riesgo", 1.0, 1.7, 11.5, 1.1, size=46, bold=True, color=WHITE)
text(s, "Predicción conforme y abstención selectiva para diagnóstico de fallas bajo incertidumbre", 1.0, 2.85, 11.3, 1.0, size=22, color=RGBColor(0xDD, 0xE6, 0xF5))
rect(s, 1.0, 4.15, 0.9, 0.06, ORANGE)
text(s, "¿Cuándo debe un sistema de IA decidir solo, y cuándo abstenerse y pedir intervención humana?", 1.0, 4.35, 11.3, 0.8, size=18, italic=True, color=WHITE)
text(s, "Jaime Astudillo  ·  Roberth Chachalo", 1.0, 5.6, 8, 0.5, size=20, bold=True, color=WHITE)
text(s, "Bloque 1 (5 min): Jaime · Bloque 2 (5 min): Roberth", 1.0, 6.05, 8, 0.35, size=13, color=ORANGE)
text(s, "Septiembre de 2026 · Repositorio: github.com/betochalo/Conformal_Prediction", 1.0, 6.45, 10, 0.4, size=13, color=RGBColor(0xC9, 0xD6, 0xE8))
notes(s, "JAIME · 0:00–0:20. Presentarse ambos en una frase. Leer la pregunta central en voz alta: el proyecto no pregunta si se puede clasificar fallas, sino cuándo el sistema debe decidir solo. Anunciar el reparto: Jaime problema, datos y método; Roberth implementación, resultados y conclusiones.")

# =============================================================================== agenda
s = slide_base("Agenda", "Diez minutos en dos bloques", 0.2)
blocks = [
    ("Bloque 1 · Jaime · 5 min", BLUE, [
        ("Problema", "Costos asimétricos y probabilidades sin garantía"),
        ("Idea central", "La política de tres vías por tamaño del conjunto"),
        ("Datos", "AI4I 2020 y lo que encontró la auditoría"),
        ("Método", "El cuantil conforme y las dos variantes"),
    ]),
    ("Bloque 2 · Roberth · 5 min", ORANGE, [
        ("Implementación", "Paquete reproducible, dos clasificadores, GPU"),
        ("Resultados", "Cobertura por clase, tamaño, abstención, errores"),
        ("En planta", "Cómo se repartirían 1 995 ciclos"),
        ("Conclusiones", "Qué se sostiene y qué recomendamos"),
    ]),
]
for bi, (btitle, bcolor, items) in enumerate(blocks):
    y = 1.65 + bi * 2.15
    rect(s, 0.7, y, 11.9, 1.95, SURFACE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.04)
    rect(s, 0.7, y, 0.09, 1.95, bcolor, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
    text(s, btitle, 0.95, y + 0.12, 5, 0.4, size=16, bold=True, color=NAVY)
    for i2, (name, desc) in enumerate(items):
        x = 0.95 + i2 * 2.92
        circle = rect(s, x, y + 0.65, 0.42, 0.42, bcolor, shape=MSO_SHAPE.OVAL)
        shape_text(circle, str(i2 + 1), size=13, bold=True, color=WHITE)
        text(s, name, x + 0.55, y + 0.66, 2.3, 0.35, size=14, bold=True, color=NAVY)
        text(s, desc, x + 0.55, y + 1.0, 2.3, 0.9, size=10, color=MUTED)
rect(s, 0.7, 6.15, 11.9, 0.85, BLUE_SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
text(s, ["Quince diapositivas de apéndice quedan como respaldo para preguntas: métricas completas, cobertura por clase, umbrales, objetivos de la propuesta,",
         "particiones, split frente a Mondrian, los dos modelos, benchmark CPU/GPU, hipótesis, conceptos del curso, recomendaciones, mapa de la rúbrica y referencias."],
     0.95, 6.24, 11.5, 0.7, size=11, color=INK)
notes(s, "JAIME · 0:20–0:35. Anunciar el reparto y los seis bloques. No leer la agenda entera: señalar los dos bloques y seguir.")

# =============================================================================== problema
s = slide_base("1 · Problema", "Equivocarse cuesta distinto según la dirección del error", 0.75)
speaker_band(s, "Jaime", 1, None)
cards = [
    ("Detener sin motivo", "Parar la línea por una falsa alarma cuesta producción y confianza en el sistema.", ORANGE),
    ("No detectar una falla", "Una falla inminente que pasa como operación normal es el error más caro: daño y parada no planificada.", NAVY_2),
    ("Confundir el modo", "Diagnosticar desgaste cuando es sobreesfuerzo envía al técnico equivocado con la herramienta equivocada.", BLUE),
]
for i, (t, b, c) in enumerate(cards):
    card(s, 0.7 + i * 4.0, 1.6, 3.85, 1.75, t, b, accent=c)
rect(s, 0.7, 3.65, 11.9, 0.04, GRID)
stat(s, 0.8, 3.9, 2.6, "96.7 %", "de los ciclos son normales: el desbalance es el régimen del problema", ORANGE)
text(s, ["Un clasificador convencional siempre responde la clase más probable y no distingue cuándo esa respuesta es confiable.",
         "La probabilidad que reporta es una salida del propio modelo, no una garantía. Con desbalance severo es optimista justo en las fallas.",
         "Lo que hace falta: convertir la incertidumbre en una decisión operativa que sepa cuándo abstenerse."],
     3.7, 3.85, 8.9, 2.6, size=15, color=INK, line_spacing=1.25)
notes(s, "JAIME · 0:30–1:15. Tres errores con costos distintos. Insistir en que la probabilidad del softmax no está calibrada y que el desbalance la vuelve optimista donde menos conviene. Cerrar con la necesidad de un mecanismo principiado para abstenerse.")

# =============================================================================== tres vías (diagrama)
s = slide_base("1 · Idea central", "La política de tres vías: el tamaño del conjunto decide quién actúa", 1.0)
speaker_band(s, "Jaime", 1, None)
src = rect(s, 0.7, 2.3, 2.6, 1.5, NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
shape_text(src, ["Conjunto de predicción C(x)", "clases que el modelo no puede descartar con confianza 1 − α"], size=14, bold=True, color=WHITE)
routes = [
    ("|C(x)| = 1", "Automática", "El sistema actúa solo. Ejemplo: {HDF}", BLUE, BLUE_SOFT),
    ("|C(x)| = 2", "Asistida", "Diagnóstico parcial útil: «es HDF u OSF, no otra cosa». Ejemplo: {HDF, OSF}", ORANGE, ORANGE_SOFT),
    ("|C(x)| = 0 o > 2", "Humana", "Sin evidencia discriminante o caso atípico. Ejemplos: {} · {Normal, TWF, HDF}", MUTED, SURFACE),
]
for i, (cond, name, desc, accent, fill) in enumerate(routes):
    y = 1.55 + i * 1.65
    arrow(s, 3.3, 3.05, 4.6, y + 0.6, color=accent, width=2)
    box = rect(s, 4.6, y, 2.0, 1.2, accent, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1)
    shape_text(box, cond, size=15, bold=True, color=WHITE)
    rect(s, 6.7, y, 5.9, 1.2, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    text(s, name, 6.9, y + 0.1, 5.5, 0.4, size=16, bold=True, color=INK)
    text(s, desc, 6.9, y + 0.5, 5.5, 0.7, size=12, color=MUTED)
text(s, "Abstención = fracción de ciclos con tamaño distinto de 1: la carga de trabajo que el sistema traslada a personas. El conjunto de dos etiquetas no es un fracaso: descarta el resto del espacio de diagnóstico con garantía.",
     0.7, 6.45, 11.9, 0.65, size=12, italic=True, color=MUTED)
notes(s, "JAIME · 1:15–2:15. Recorrer las tres vías. Enfatizar la vía asistida como información accionable y que la abstención es medible: es lo que después se reporta como métrica.")

# =============================================================================== auditoría
s = slide_base("2 · Datos", "AI4I 2020: 10 000 ciclos y lo que encontró la auditoría", 1.25)
speaker_band(s, "Jaime", 1, None)
table(s, [
    ["Hallazgo de la auditoría", "Conteo", "Decisión y justificación"],
    ["Fallas con Machine failure = 1 y ningún modo activo", "9", "Excluidas: no admiten etiqueta de modo"],
    ["RNF puro (falla aleatoria); todos con Machine failure = 0", "18", "Excluidas: sin firma física, ruido irreducible que contaminaría Normal"],
    ["RNF junto con un modo físico", "1", "Conservado con su modo físico (exclude_only_rnf)"],
    ["Modos físicos simultáneos (PWF+OSF 11, HDF+OSF 6, HDF+PWF 3, TWF+OSF 2, triple 1)", "23", "Prioridad por escasez TWF > PWF > OSF > HDF: no restar ejemplos a las clases escasas"],
    ["Modo activo con Machine failure = 0", "0", "La inconsistencia prevista no ocurre"],
    ["Autocorrelación lag-1 de la temperatura del aire", "0.999", "Estructura secuencial: partición aleatoria; la garantía no se afirma para despliegue temporal"],
], 0.7, 1.55, 11.9, [5.2, 0.9, 5.8], size=11, row_h=0.46)
card(s, 0.7, 4.95, 5.8, 1.55, "Representa fielmente el problema, con límites declarados",
     "Sintético con mecanismos físicos reales; desbalance 96.7 / 3.3 igual al de planta; etiquetas derivadas de umbrales sobre la fila actual, lo que sostiene la intercambiabilidad entre bloques aleatorios.", accent=AQUA)
card(s, 6.8, 4.95, 5.8, 1.55, "Distribución final: 9 973 filas",
     "Normal 9 643 · TWF 46 · HDF 106 · PWF 94 · OSF 84. Los conteos de indicadores (354) no son los de las clases finales (330 con modo): la suma engaña si no se resuelven los solapamientos.", accent=ORANGE)
notes(s, "JAIME · 2:15–3:30. Criterio 2b. Abrir con una frase sobre el dataset (10 000 ciclos de una fresadora, 6 predictores, 3.4 % de fallas, sintético con mecanismos físicos reales, hash registrado). Luego SOLO tres filas de la tabla: RNF, fallas sin modo y solapamientos. El resto está en el apéndice. Tres hallazgos clave: RNF no activa Machine failure, nueve fallas sin modo, y 23 solapamientos resueltos por escasez. La autocorrelación justifica por qué la garantía es para partición aleatoria y no para el tiempo.")

# =============================================================================== método
s = slide_base("3 · Método", "El cuantil conforme: un estadístico de orden con corrección finita", 1.5)
speaker_band(s, "Jaime", 1, None)
text(s, "1. Puntuación de no conformidad", 0.7, 1.5, 6, 0.35, size=14, bold=True, color=NAVY_2)
text(s, "s(x, y) = 1 − p̂(y | x), con el modelo ya entrenado. Para cada ejemplo de calibración se toma la puntuación de su etiqueta verdadera.", 0.7, 1.85, 5.6, 0.9, size=12, color=INK)
text(s, "2. Umbral", 0.7, 2.75, 6, 0.35, size=14, bold=True, color=NAVY_2)
text(s, "q̂ = puntuación de rango ⌈(n + 1)(1 − α)⌉ entre las n de calibración, sin interpolación. Si el rango supera n: q̂ = ∞.", 0.7, 3.1, 5.6, 0.9, size=12, color=INK)
text(s, "3. Conjunto de predicción", 0.7, 4.0, 6, 0.35, size=14, bold=True, color=NAVY_2)
text(s, "C(x) = { y : s(x, y) ≤ q̂ }. Los empates se incluyen con ≤.", 0.7, 4.35, 5.6, 0.6, size=12, color=INK)
rect(s, 0.7, 5.15, 5.6, 1.45, BLUE_SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
text(s, "Garantía (Vovk et al., 2005)", 0.9, 5.22, 5.2, 0.35, size=13, bold=True, color=NAVY_2)
text(s, "Si (X₁,Y₁), …, (Xₙ,Yₙ), (Xₙ₊₁,Yₙ₊₁) son intercambiables, P(Yₙ₊₁ ∈ C(Xₙ₊₁)) ≥ 1 − α. No depende de que el modelo sea bueno: un modelo malo produce conjuntos grandes, no conjuntos incorrectos.", 0.9, 5.55, 5.2, 1.0, size=11, color=INK)

# diagrama: recta numérica con puntuaciones de calibración
x0, x1, yl = 7.0, 12.4, 3.3
text(s, "Ejemplo a mano: n = 5, α = 0.4 → rango ⌈6 × 0.6⌉ = 4", 6.8, 1.5, 5.9, 0.4, size=13, bold=True, color=NAVY_2)
rect(s, x0, yl - 0.02, (0.3 / 0.6) * (x1 - x0), 0.6, ORANGE_SOFT)  # zona ≤ q̂ (0 a 0.30)
arrow(s, x0, yl + 0.3, x1 + 0.2, yl + 0.3, color=MUTED, width=1.2)
for tick in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
    tx = x0 + (tick / 0.6) * (x1 - x0)
    rect(s, tx - 0.005, yl + 0.22, 0.01, 0.16, MUTED)
    text(s, f"{tick:.1f}", tx - 0.3, yl + 0.42, 0.6, 0.3, size=9, color=MUTED, align=PP_ALIGN.CENTER)
scores = [(0.10, 1), (0.125, 2), (0.25, 3), (0.30, 4), (0.50, 5)]
for value, rank in scores:
    sx = x0 + (value / 0.6) * (x1 - x0)
    dot = rect(s, sx - 0.13, yl + 0.17, 0.26, 0.26, ORANGE if rank == 4 else BLUE, shape=MSO_SHAPE.OVAL)
    shape_text(dot, str(rank), size=9, bold=True, color=WHITE)
text(s, "q̂ = 0.30 (rango 4)", x0 + (0.30 / 0.6) * (x1 - x0) - 0.9, yl - 0.55, 1.8, 0.35, size=11, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
text(s, "puntuación s de la etiqueta verdadera en calibración", x0, yl + 0.75, 5.4, 0.3, size=10, color=MUTED, align=PP_ALIGN.CENTER)
text(s, ["Caso nuevo con p̂ = (0.80, 0.20): s(Normal) = 0.20 ≤ 0.30 ✓, s(TWF) = 0.80 ✗ → C = {Normal}, vía automática.",
         "Caso nuevo con p̂ = (0.55, 0.45): s = 0.45 y 0.55, ambas > 0.30 → C = ∅, vía humana.",
         "Con las mismas cifras reales del proyecto: split hgb, α = 0.05, n = 2 992, rango 2 844, q̂ = 0.0019."],
     6.8, 4.6, 5.9, 2.0, size=11, color=INK, line_spacing=1.2)
notes(s, "JAIME · 3:30–5:00. Recorrer los tres pasos con la recta numérica: cinco puntuaciones ordenadas, el rango 4 marca el umbral 0.30, la zona sombreada es lo que entra al conjunto. Luego los dos casos nuevos. Cerrar con el número real: q̂ = 0.0019 porque el modelo es muy confiado en Normal.")

# =============================================================================== relevo
s = prs.slides.add_slide(BLANK)
SLIDE_NO["n"] += 1
SCHEDULE.append(("Relevo", 0.15))
rect(s, 0, 0, 13.333, 7.5, NAVY)
rect(s, 0.9, 2.55, 0.9, 0.06, ORANGE)
text(s, "Hasta aquí: el método", 0.9, 2.7, 11, 0.7, size=32, bold=True, color=WHITE)
text(s, "Conjuntos de predicción con garantía y una política de tres vías por su tamaño.", 0.9, 3.5, 11, 0.5, size=17, color=RGBColor(0xC9, 0xD6, 0xE8))
rect(s, 0.9, 4.35, 5.6, 1.5, NAVY_2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
text(s, "Ahora: ¿funciona?", 1.15, 4.5, 5.1, 0.45, size=20, bold=True, color=WHITE)
text(s, "Implementación, resultados y conclusiones.", 1.15, 4.98, 5.1, 0.7, size=14, color=RGBColor(0xC9, 0xD6, 0xE8))
text(s, "Roberth Chachalo", 7.0, 4.6, 5, 0.5, size=22, bold=True, color=ORANGE)
text(s, "Bloque 2 · 5 minutos", 7.0, 5.1, 5, 0.4, size=14, color=RGBColor(0xC9, 0xD6, 0xE8))
text(s, f"{SLIDE_NO['n']}", 12.2, 7.05, 0.5, 0.3, size=9, color=RGBColor(0x8F, 0xA6, 0xC4), align=PP_ALIGN.RIGHT)
notes(s, "RELEVO · 5:00–5:10. Jaime cierra: «hasta aquí el método; Roberth cuenta si funciona». Cambio de orador. No leer la diapositiva.")

# =============================================================================== implementación
s = slide_base("4 · Implementación", "Pipeline reproducible: dos clasificadores, una capa conforme", 0.85)
speaker_band(s, "Roberth", 2, None)
stages = [
    ("CSV + hash", "data.py", NAVY_2), ("Auditoría y etiquetas", "data.py", NAVY_2), ("Particiones y viabilidad", "splitting.py", NAVY_2),
    ("HGB · MLP (GPU)", "model.py · torch_model.py", BLUE), ("Puntuaciones y cuantil", "scores.py · conformal.py", ORANGE),
    ("Split · Mondrian", "conformal.py", ORANGE), ("Tres vías", "decision.py", BLUE), ("Métricas y figuras", "evaluation.py · pipeline.py", NAVY_2),
]
for i, (name, mod, color) in enumerate(stages):
    x = 0.7 + i * 1.5
    b = rect(s, x, 1.7, 1.35, 1.0, color, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1)
    shape_text(b, name, size=11, bold=True, color=WHITE)
    text(s, mod, x - 0.05, 2.75, 1.45, 0.5, size=9, color=MUTED, align=PP_ALIGN.CENTER)
    if i < 7:
        arrow(s, x + 1.37, 2.2, x + 1.48, 2.2, color=LIGHT, width=1.5)
text(s, "Un solo entrenamiento por modelo; probabilidades, particiones y alphas compartidos por ambas variantes. Calibración separada del ajuste y del bloque de prueba. Nada se selecciona con prueba.",
     0.7, 3.3, 11.9, 0.6, size=12, italic=True, color=MUTED)
stat(s, 0.7, 4.1, 2.3, "112", "pruebas: cuantil, empates, clases ausentes, particiones, métricas, CLI, integración", BLUE, value_size=30)
stat(s, 3.2, 4.1, 2.3, "1", "comando reproduce todo:\npython -m conformal_fault_inference_with_abstention", BLUE, value_size=30)
stat(s, 5.7, 4.1, 2.3, "SHA-256", "del CSV, semillas, versiones y GPU en el manifiesto de cada ejecución", NAVY_2, value_size=22)
stat(s, 8.2, 4.1, 2.3, "0", "bibliotecas conformes: NumPy propio, verificado contra cálculos a mano", ORANGE, value_size=30)
stat(s, 10.6, 4.1, 2.1, "1.9 MB", "de resultados versionados: CSV, umbrales, probabilidades, figuras", NAVY_2, value_size=22)
text(s, ["Entorno: uv con uv.lock, Python 3.13, ruff. Extra opcional torch (índice CUDA 12.8) para el MLP; sin él, el producto mínimo funciona igual.",
         "Salidas por ejecución en artifacts/, sin sobrescritura accidental; notebook de análisis que reconstruye las métricas desde las probabilidades guardadas."],
     0.7, 5.75, 11.9, 1.2, size=11, color=INK)
notes(s, "ROBERTH · 5:10–6:00. Criterio 2d. Recorrer el pipeline de izquierda a derecha en 20 segundos. Mencionar los dos clasificadores (HistGradientBoosting como base y un MLP en PyTorch en GPU como extensión) y que ambos alimentan la misma capa conforme. Luego las cinco cifras. Detalles de arquitectura y benchmark CPU/GPU están en el apéndice. Enfatizar contratos por módulo, una sola línea para reproducir y el manifiesto con hash y versiones. Las pruebas incluyen casos calculados a mano.")

# =============================================================================== resultados
s = slide_base("5 · Resultado 1", "Coberturas marginales cerca del nominal en ambos modelos", 0.75)
speaker_band(s, "Roberth", 2, None)
picture(s, "global_metrics_hgb.png", 0.7, 1.5, w=7.9)
table(s, [
    ["Modelo · variante", "α", "Cob.", "Tam.", "Abst."],
    ["hgb · split", "0.10", "0.909", "0.91", "0.088"],
    ["hgb · mondrian", "0.10", "0.917", "1.32", "0.305"],
    ["mlp · split", "0.10", "0.902", "0.90", "0.096"],
    ["mlp · mondrian", "0.10", "0.903", "1.07", "0.085"],
    ["hgb · mondrian", "0.05", "0.959", "2.64", "1.000"],
], 8.8, 1.55, 3.8, [1.5, 0.5, 0.6, 0.6, 0.6], size=10, row_h=0.32)
text(s, "Cob. = cobertura marginal · Tam. = tamaño medio del conjunto · Abst. = tasa de abstención", 8.8, 3.55, 3.8, 0.6, size=9, color=MUTED)
text(s, ["En esta partición, las 12 combinaciones quedan a menos de 0.02 del nominal (n = 1 995; desviación típica de una proporción ≈ 0.007).",
         "Consistente con la garantía, que es una afirmación en probabilidad bajo intercambiabilidad; una partición la ilustra, no la demuestra.",
         "Árbol y MLP difieren en tamaño y abstención, no en cobertura: la incertidumbre se traduce en tamaño."],
     0.7, 4.45, 11.9, 2.3, size=13, color=INK, line_spacing=1.2)
notes(s, "ROBERTH · 6:00–6:45. Leer la curva de cobertura pegada a la línea 1 − α. Moderar la afirmación: consistente con la garantía, no demostración. Tamaño y abstención sí dependen del modelo.")

s = slide_base("5 · Resultado 2", "Split ignora las fallas; Mondrian las cubre", 1.25)
speaker_band(s, "Roberth", 2, None)
picture(s, "coverage_by_class_hgb.png", 0.7, 1.5, w=11.9)
card(s, 0.7, 5.05, 3.85, 1.7, "Split, α = 0.10 (hgb)", "Cobertura marginal 0.909, pero TWF 0.000, PWF 0.000, HDF 0.143. Normal, con 0.938, absorbe toda la garantía.", accent=BLUE)
card(s, 4.72, 5.05, 3.85, 1.7, "Mondrian, α = 0.10 (hgb)", "Las cuatro fallas entre 0.952 y 1.000; Normal 0.915. Con α = 0.05, TWF entra siempre por umbral infinito.", accent=ORANGE)
card(s, 8.75, 5.05, 3.85, 1.7, "Lectura con cautela", "TWF tiene 9 casos en prueba: cada uno mueve su cobertura 0.11. La barra 0.78 con α = 0.20 son 7 de 9; no permite concluir subcobertura.", accent=MUTED)
notes(s, "ROBERTH · 6:45–8:00. Diapositiva central; si el tiempo aprieta, esta es la que NO se recorta. Señalar las barras azules en cero y las naranjas sobre la línea. Es el objetivo 4 de la propuesta, cuantificado. Mencionar el soporte de TWF.")

s = slide_base("5 · Resultado 3", "El precio de la garantía: tamaño, abstención y errores automáticos", 0.75)
speaker_band(s, "Roberth", 2, None)
picture(s, "automatic_error_rate.png", 0.7, 1.5, w=7.6)
text(s, ["La cobertura no garantiza la exactitud condicionada a decidir automáticamente: un conjunto de tamaño 1 puede contener la clase equivocada (clasificación selectiva, Geifman y El-Yaniv 2017).",
         "Mondrian automatiza más pero con falsas alarmas: el umbral de Normal es casi cero (rango 2 605 de 2 893 puntuaciones concentradas en 0); Normal sale del conjunto en cuanto p̂(Normal) baja de 0.99999 y queda un singleton de falla.",
         "Split se equivoca poco, pero en la dirección peor: sus errores son fallas enviadas como Normal, y sus abstenciones son conjuntos vacíos."],
     8.5, 1.5, 4.1, 3.5, size=10, color=INK, line_spacing=1.15)
table(s, [
    ["α = 0.10", "Automáticas", "Erróneas", "Tasa de error", "Error dominante"],
    ["hgb · split", "1 819", "5", "0.3 %", "5 fallas enviadas como Normal"],
    ["hgb · mondrian", "1 387", "82", "5.9 %", "81 normales enviados como falla"],
    ["mlp · split", "1 803", "3", "0.2 %", "3 fallas enviadas como Normal"],
    ["mlp · mondrian", "1 825", "150", "8.2 %", "149 normales enviados como falla"],
], 0.7, 4.95, 11.9, [2.4, 2.0, 1.8, 1.7, 4.0], size=11, row_h=0.34)
text(s, "Con α = 0.05 Mondrian abstiene el 100 % (hgb): TWF acompaña a Normal en casi todos los conjuntos. Con α = 0.10 el tamaño medio baja a 1.32 y la mayoría de los ciclos se resuelve sola.",
     0.7, 6.7, 11.9, 0.45, size=10, italic=True, color=MUTED)
notes(s, "ROBERTH · 8:00–8:45. El hallazgo que cambia la lectura: automatizar 69.5 % o 91.5 % debe ir con su error entre automáticas. Explicar el mecanismo del umbral de Normal. Cuál conviene depende de costos que el proyecto no fija.")

s = slide_base("5 · En planta", "Cómo se repartirían 1 995 ciclos de prueba con Mondrian y α = 0.10", 0.4)
speaker_band(s, "Roberth", 2, None)
for j, (model, auto, asis, hum, err, cov) in enumerate((("HistGradientBoosting", 1387, 571, 37, 82, "0.917"), ("MLP en GPU", 1825, 136, 34, 150, "0.903"))):
    y = 1.6 + j * 2.7
    text(s, model, 0.7, y, 3, 0.4, size=15, bold=True, color=NAVY)
    text(s, f"cobertura marginal {cov} · las cuatro fallas ≥ 0.95", 0.7, y + 0.38, 3.2, 0.6, size=11, color=MUTED)
    total = 1995
    xs = 4.0
    width_total = 8.6
    small = []
    for value, label, color in ((auto, "Automática", BLUE), (asis, "Asistida", ORANGE), (hum, "Humana", MUTED)):
        w = width_total * value / total
        b = rect(s, xs, y, max(w, 0.05), 1.0, color)
        if w > 1.2:
            shape_text(b, [f"{label} {value}", f"{value / total:.1%}"], size=13, bold=True, color=WHITE)
        else:
            small.append(f"{label} {value} ({value / total:.1%})")
        xs += w
    text(s, "   ·   ".join(small), 4.0, y + 1.05, 8.6, 0.35, size=11, color=MUTED, align=PP_ALIGN.RIGHT)
    text(s, f"Entre las {auto} automáticas, {err} son incorrectas ({err / auto:.1%}), casi todas ciclos normales enviados como falla.", 4.0, y + 1.45, 8.6, 0.5, size=11, color=INK)
rect(s, 0.7, 6.15, 11.9, 0.75, BLUE_SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
text(s, "La abstención es carga de trabajo humana medible. Elegir entre 30 % de abstención con 5.9 % de falsas alarmas y 8.5 % con 8.2 % exige la matriz de costos de la propuesta original, que queda como extensión.",
     0.9, 6.22, 11.5, 0.65, size=12, color=INK)
notes(s, "ROBERTH · 8:45–9:10. Traducir las métricas a operación: cuántos ciclos van a cada vía y con qué error. Cerrar con que la decisión final necesita costos.")

# =============================================================================== conclusiones
s = slide_base("6 · Conclusiones", "Lo que la ejecución sostiene y qué recomendamos", 0.6)
speaker_band(s, "Roberth", 2, None)
concl = [
    ("Coberturas consistentes con la garantía y sin depender del clasificador", "12 de 12 combinaciones a menos de 0.02 del nominal. La calidad del modelo se ve en el tamaño del conjunto, no en la cobertura."),
    ("La cobertura marginal es insuficiente para diagnóstico de fallas", "Split alcanza 0.909 cubriendo cero casos de TWF y PWF. Un sistema que reporte solo esa cifra parece correcto mientras ignora lo que importa."),
    ("Mondrian responde la pregunta del proyecto, con un costo medible", "Con α = 0.10: garantía por clase, 70–90 % de decisiones automáticas, y entre ellas 6–8 % de falsas alarmas. Split se equivoca menos, pero omitiendo fallas."),
    ("La abstención es el precio de la garantía y depende de los datos escasos", "Con 14 ejemplos de TWF, exigir α = 0.05 obliga a incluirla siempre y la abstención sube a 1.0. Es aritmética del rango, no un fallo del método."),
    ("La GPU no aceleró este entrenamiento", "Con 5 061 parámetros la CPU fue 1.4 veces más rápida. Afirmar la ventaja sin medir habría sido un error."),
]
RECOM = "Recomendamos operar con Mondrian y α = 0.10; repetir la partición con varias semillas; sustituir 1 − p por APS para evitar conjuntos vacíos; definir la matriz de costos para elegir α por costo esperado; y priorizar la recolección de TWF."
for i, (t, b) in enumerate(concl):
    y = 1.5 + i * 0.99
    num = rect(s, 0.7, y + 0.1, 0.5, 0.5, ORANGE if i in (1, 2) else BLUE, shape=MSO_SHAPE.OVAL)
    shape_text(num, str(i + 1), size=14, bold=True, color=WHITE)
    text(s, t, 1.4, y, 11.2, 0.4, size=14, bold=True, color=NAVY)
    text(s, b, 1.4, y + 0.4, 11.2, 0.6, size=11, color=MUTED)
rect(s, 0.7, 6.5, 11.9, 0.55, BLUE_SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1)
text(s, RECOM, 0.95, 6.58, 11.5, 0.45, size=11, color=INK)
notes(s, "ROBERTH · 9:10–9:45. Cinco conclusiones, cada una con su número. Volver a la pregunta de la portada: el sistema sabe cuándo abstenerse, y sabemos cuánto cuesta.")

s = prs.slides.add_slide(BLANK)
SLIDE_NO["n"] += 1
SCHEDULE.append(("Cierre", 0.25))
rect(s, 0, 0, 13.333, 7.5, NAVY)
rect(s, 0, 0, 0.35, 7.5, ORANGE)
text(s, "¿Cuándo debe un sistema de IA decidir solo?", 1.0, 1.9, 11.5, 0.9, size=36, bold=True, color=WHITE)
text(s, "Cuando el conjunto de predicción tiene una sola etiqueta bajo una garantía por clase. Y ahora sabemos qué fracción de ciclos es esa, y con qué error.", 1.0, 2.9, 11.3, 1.2, size=20, color=RGBColor(0xDD, 0xE6, 0xF5))
text(s, "Gracias. Preguntas.", 1.0, 4.6, 8, 0.6, size=24, bold=True, color=WHITE)
text(s, ["Repositorio: github.com/betochalo/Conformal_Prediction", "Reproducir: uv sync --extra torch && uv run python -m conformal_fault_inference_with_abstention", "Informe: RESULTADOS.md · Notebook: notebooks/analisis_resultados.ipynb"], 1.0, 5.4, 11.3, 1.3, size=13, color=RGBColor(0xC9, 0xD6, 0xE8))
notes(s, "ROBERTH · 9:45–10:00. Responder la pregunta de la portada en una frase y abrir preguntas. El apéndice tiene tablas completas, umbrales, benchmark y el mapa de la rúbrica.")

# =============================================================================== apéndice
divider("Apéndice", "Material de respaldo para preguntas; no forma parte de los 10 minutos", None)
notes(prs.slides[-1], "No forma parte del tiempo de exposición.")

# =============================================================================== objetivos y alcance
s = slide_base("Apéndice F", "Objetivo general, objetivos específicos y alcance aprobado", None)
rect(s, 0.7, 1.55, 11.9, 1.05, BLUE_SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
text(s, "Objetivo general", 0.9, 1.62, 3, 0.35, size=12, bold=True, color=NAVY_2)
text(s, "Diseñar, implementar desde cero y evaluar un sistema de diagnóstico de fallas que convierta la incertidumbre cuantificada por predicción conforme en una política de decisión de tres vías: automática, asistida y humana.",
     0.9, 1.92, 11.5, 0.7, size=13, color=INK)
table(s, [
    ["Objetivo específico de la propuesta", "Estado en la entrega"],
    ["1. Fundamento matemático: intercambiabilidad, no conformidad, cuantil y cobertura marginal", "Cumplido"],
    ["2. Split conformal, Mondrian y APS en NumPy, sin bibliotecas conformes", "Split y Mondrian cumplidos · APS pospuesto"],
    ["3. Cobertura independiente del clasificador (modelo fuerte frente a débil)", "Adaptado: MLP en PyTorch en lugar de GaussianNB; el contraste de calidad es menor"],
    ["4. Split subcubre las minoritarias y Mondrian lo corrige, cuantificado", "Cumplido"],
    ["5. Ablación de características físicas", "Pospuesto por observación docente"],
    ["6. Política sensible al costo y alpha óptimo", "Pospuesto por observación docente"],
], 0.7, 2.85, 11.9, [8.0, 3.9], size=12, row_h=0.4)
text(s, "El docente redujo el alcance a un producto mínimo de dos semanas: un clasificador base, dos variantes conformes, métricas por clase y análisis reproducible. El MLP en GPU es una extensión añadida sobre ese mínimo.",
     0.7, 5.8, 11.9, 0.8, size=12, italic=True, color=MUTED)
notes(s, "Respaldo. Criterio 2a: qué se prometió y qué se entregó, con el recorte aprobado explícito. Si preguntan por GaussianNB: la propuesta lo quería precisamente por estar mal calibrado, para demostrar que la garantía no depende del modelo; el MLP no cumple ese papel porque resulta tan bueno como el árbol, así que el contraste de calidad que se observa es menor que el previsto. Lo que sí se comprobó es que dos modelos distintos dan la misma cobertura y difieren en tamaño y abstención.")

# =============================================================================== datos
s = slide_base("Apéndice G", "AI4I 2020: predictores y clases", None)
stat(s, 0.7, 1.55, 2.5, "10 000", "ciclos de operación", NAVY_2)
stat(s, 3.2, 1.55, 2.5, "6", "predictores físicos", NAVY_2)
stat(s, 5.7, 1.55, 2.5, "339", "fallas registradas (3.4 %)", ORANGE)
stat(s, 8.2, 1.55, 2.5, "4", "modos físicos de falla", NAVY_2)
stat(s, 10.4, 1.55, 2.3, "0", "valores faltantes", AQUA)
table(s, [
    ["Predictor", "Tipo", "Unidad"],
    ["Type", "categórica L / M / H (50 / 30 / 20 %)", "—"],
    ["Air temperature", "continua", "K"],
    ["Process temperature", "continua", "K"],
    ["Rotational speed", "entera", "rpm"],
    ["Torque", "continua", "Nm"],
    ["Tool wear", "entera", "min"],
], 0.7, 3.0, 6.0, [2.2, 2.9, 0.9], size=12, row_h=0.36)
table(s, [
    ["Clase", "Modo de falla", "Mecanismo físico"],
    ["Normal", "Sin falla", "—"],
    ["TWF", "Desgaste de herramienta", "Desgaste acumulado sobre umbral"],
    ["HDF", "Disipación de calor", "Diferencial térmico insuficiente"],
    ["PWF", "Potencia", "Torque × velocidad fuera de rango"],
    ["OSF", "Sobreesfuerzo", "Desgaste × torque sobre umbral"],
], 7.0, 3.0, 5.6, [0.9, 2.0, 2.7], size=12, row_h=0.36)
text(s, "Fuente: UCI Machine Learning Repository, id 601, CC BY 4.0. Descarga con ucimlrepo, bytes conservados, SHA-256 registrado (ffc3c28f…9b5a18). Los identificadores UID y Product ID se conservan para trazabilidad, nunca como predictores.",
     0.7, 5.75, 11.9, 0.8, size=11, color=MUTED)
notes(s, "2:45–3:15. Dataset sintético pero generado con mecanismos físicos reales. Cinco clases construidas a partir de cuatro indicadores binarios. Mencionar hash y procedencia: parte del criterio de reproducibilidad.")

# =============================================================================== particiones y viabilidad
s = slide_base("Apéndice H", "Particiones y viabilidad de calibración por alpha", None)
table(s, [
    ["Clase", "Entren.", "Calib.", "Prueba", "Total"],
    ["Normal", "4 821", "2 893", "1 929", "9 643"],
    ["TWF", "23", "14", "9", "46"],
    ["HDF", "53", "32", "21", "106"],
    ["PWF", "47", "28", "19", "94"],
    ["OSF", "42", "25", "17", "84"],
], 0.7, 1.55, 5.4, [1.2, 1.05, 1.05, 1.05, 1.05], size=12, bold_col0=True, highlight_rows=(2,))
text(s, "Estratificada 0.50 / 0.30 / 0.20, semilla 42, fracciones respecto al total. Cada fila en un solo bloque; índices UID guardados.", 0.7, 3.8, 5.4, 0.7, size=11, color=MUTED)
table(s, [
    ["Clase (n calib.)", "α = 0.05: rango", "finito", "α = 0.10: rango", "finito", "α = 0.20: rango", "finito"],
    ["Normal (2 893)", "2 750", "sí", "2 605", "sí", "2 316", "sí"],
    ["TWF (14)", "15", "no → ∞", "14", "sí", "12", "sí"],
    ["HDF (32)", "32", "sí", "30", "sí", "27", "sí"],
    ["PWF (28)", "28", "sí", "27", "sí", "24", "sí"],
    ["OSF (25)", "25", "sí", "24", "sí", "21", "sí"],
], 6.4, 1.55, 6.2, [1.5, 1.0, 0.65, 1.0, 0.65, 1.0, 0.65], size=11, bold_col0=True, highlight_rows=(2,))
text(s, "rango = ⌈(n + 1)(1 − α)⌉. Si supera n, el umbral es infinito y la clase entra siempre en el conjunto.", 6.4, 3.8, 6.2, 0.5, size=11, color=MUTED)
card(s, 0.7, 4.7, 11.9, 1.8, "Por qué 0.30 de calibración y no más",
     "Con α = 0.05, TWF necesita 19 ejemplos de calibración. Alcanzarlos exige poner el 45 % de los datos en calibración, dejando 7 casos de TWF en prueba: su cobertura sería prácticamente no medible. Se eligió 0.30 / 0.20 y se reportan tres alphas (0.05, 0.10, 0.20). Con 0.05, Mondrian asigna umbral infinito a TWF: no es un error, es el comportamiento conservador de la garantía con calibración escasa. No se duplicaron ni generaron ejemplos.", accent=NAVY_2, body_size=12)
notes(s, "4:15–4:45. TWF es la restricción de todo el proyecto. La decisión de fracciones se tomó con esta tabla antes de entrenar, como exige el plan.")

s = slide_base("Apéndice I", "Split conformal frente a Mondrian y el argumento del rango", None)
# panel split
rect(s, 0.7, 1.55, 5.8, 3.1, SURFACE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
text(s, "Split conformal", 0.9, 1.62, 5.4, 0.4, size=15, bold=True, color=BLUE)
text(s, "Todas las puntuaciones de calibración juntas → un solo q̂. Garantía marginal: promedio sobre todas las clases.", 0.9, 2.0, 5.4, 0.7, size=12, color=INK)
pool = rect(s, 0.9, 2.75, 5.4, 0.7, BLUE_SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.15)
shape_text(pool, "2 992 puntuaciones: 2 893 Normal + 99 fallas → q̂ = 0.0019 (α = 0.05)", size=12, color=INK)
text(s, "Riesgo: con 96.7 % de Normal, la cobertura del 95 % puede alcanzarse ignorando por completo las clases de falla.", 0.9, 3.55, 5.4, 0.9, size=12, italic=True, color=ORANGE)
# panel mondrian
rect(s, 6.8, 1.55, 5.8, 3.1, SURFACE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
text(s, "Mondrian (condicionado por clase)", 7.0, 1.62, 5.4, 0.4, size=15, bold=True, color=ORANGE)
text(s, "Calibración agrupada por etiqueta verdadera → un q̂ₖ por clase. El mismo argumento dentro de cada clase garantiza cobertura por clase.", 7.0, 2.0, 5.4, 0.7, size=12, color=INK)
per = [("Normal", "n 2 893", "1.1e-4"), ("TWF", "n 14", "∞"), ("HDF", "n 32", "0.982"), ("PWF", "n 28", "1.000"), ("OSF", "n 25", "1.000")]
for i, (c, n, q) in enumerate(per):
    b = rect(s, 7.0 + i * 1.09, 2.75, 1.0, 0.95, ORANGE_SOFT if c != "TWF" else ORANGE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
    shape_text(b, [c, n, f"q̂ = {q}"], size=11, bold=True, color=WHITE if c == "TWF" else INK)
text(s, "Precio: cada clase se calibra con pocas muestras; TWF con 14 ejemplos recibe umbral infinito con α = 0.05.", 7.0, 3.8, 5.4, 0.7, size=12, italic=True, color=MUTED)
# argumento del rango
rect(s, 0.7, 4.9, 11.9, 1.7, BLUE_SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
text(s, "Por qué ⌈(n + 1)(1 − α)⌉ y no el cuantil empírico de n", 0.9, 4.97, 11.5, 0.4, size=13, bold=True, color=NAVY_2)
text(s, ["Bajo intercambiabilidad, el rango de la puntuación nueva entre las n + 1 es uniforme en {1, …, n + 1}. Así P(s_new ≤ s_(k)) ≥ k / (n + 1) para cualquier k.",
         "Con k = ⌈(n + 1)(1 − α)⌉ se obtiene k / (n + 1) ≥ 1 − α. Sin la corrección +1 el rango es menor y la cobertura puede quedar por debajo de 1 − α. Sin empates, la cobertura tampoco supera 1 − α + 1/(n + 1) (Lei et al., 2018)."],
     0.9, 5.35, 11.5, 1.2, size=11, color=INK, line_spacing=1.15)
notes(s, "5:45–6:45. Izquierda: por qué la garantía marginal no basta. Derecha: Mondrian con los umbrales reales; TWF infinito. Abajo, el argumento de simetría que justifica el rango: es el fundamento matemático del criterio 3b.")

s = slide_base("Apéndice J", "Los dos clasificadores y la traza de entrenamiento del MLP", None)
table(s, [
    ["", "HistGradientBoosting (base aprobado)", "MLP en PyTorch (extensión)"],
    ["Preprocesamiento", "One-hot de Type; numéricas sin escalar", "One-hot de Type; StandardScaler ajustado solo con entrenamiento"],
    ["Arquitectura", "100 iteraciones, sin early stopping", "8 → 64 → 64 → 5, ReLU; 5 061 parámetros"],
    ["Entrenamiento", "scikit-learn, CPU", "Adam 1e-3, 200 épocas, lotes de 256, RTX 5070 Ti"],
    ["Selección", "Ninguna: configuración fija", "Ninguna: épocas fijas, sin usar calibración ni prueba"],
    ["Exactitud en prueba (referencia)", "0.979 global · TWF 0.000", "0.982 global · TWF 0.111"],
], 0.7, 1.55, 11.9, [2.4, 4.6, 4.9], size=11, bold_col0=True, row_h=0.4)
picture(s, "training_trace_mlp.png", 0.7, 4.1, w=8.2)
text(s, ["Traza por época al estilo del hackatón 3: pérdida antes del paso, norma del gradiente y norma del cambio de parámetros.",
         "La exactitud global de entrenamiento (0.995) apenas supera responder siempre Normal (0.967) y esconde que TWF baja de 0.22 a 0.13: la misma lección del 75 % del hackatón."],
     9.1, 4.15, 3.5, 2.7, size=11, color=INK)
notes(s, "7:30–8:00. Ambos exponen fit, predict_proba y classes_. La exactitud es solo referencia: el predictor conforme no se evalúa por exactitud.")

s = slide_base("Apéndice K", "CPU frente a GPU con el protocolo del ejercicio de la semana 3", None)
table(s, [
    ["Dispositivo", "Épocas", "Repeticiones", "Mediana", "Mín", "Máx", "Pérdida inicial", "Pérdida final"],
    ["cpu", "200", "3", "5.0 s", "4.99 s", "5.06 s", "1.1035", "0.0209"],
    ["cuda", "200", "3", "6.8 s", "6.76 s", "6.99 s", "1.1035", "0.0209"],
], 0.7, 1.55, 11.9, [1.4, 1.0, 1.4, 1.4, 1.4, 1.4, 1.95, 1.95], size=12)
card(s, 0.7, 2.9, 3.85, 1.9, "La GPU no gana aquí",
     "5 061 parámetros y lotes de 256: cada paso lanza kernels diminutos y el costo de lanzarlos supera al cálculo. La ventaja de CUDA aparece con matrices grandes. Se mide, no se asume.", accent=ORANGE)
card(s, 4.72, 2.9, 3.85, 1.9, "Resultados no idénticos bit a bit",
     "Pérdidas iguales a cuatro decimales, pero tras 200 épocas de Adam los parámetros difieren hasta 0.027 (fuera de la tolerancia 1e-4) y las probabilidades hasta 0.041. Misma clase predicha en 99.98 % de las filas.", accent=NAVY_2)
card(s, 8.75, 2.9, 3.85, 1.9, "Transferencias",
     "Inferencia sobre 1 995 filas: 0.16 ms con datos residentes en GPU frente a 0.30 ms incluyendo la copia CPU → GPU → CPU. Para la capa conforme, en NumPy, es irrelevante.", accent=BLUE)
text(s, ["Entorno registrado en el manifiesto: Intel Core, 24 hilos; RTX 5070 Ti Laptop, 11.94 GiB; torch 2.11.0+cu128; float32; precisión de matmul «highest». La primera ejecución en GPU paga la inicialización del contexto CUDA (9.2 s); el benchmark corre después.",
         "Conceptos del curso: tensores en dispositivo, autodiferenciación con backward(), ciclo zero_grad → forward → backward → step, medianas de repeticiones y comparación con tolerancia."],
     0.7, 5.0, 11.9, 1.8, size=11, color=INK)
notes(s, "8:00–8:30. Resultado honesto: la CPU fue 1.4 veces más rápida. Explicar por qué y conectar con el ejercicio de la semana 3 y sus tolerancias.")

s = slide_base("Apéndice L", "Hipótesis: qué se apoya en resultados y qué no", None)
table(s, [
    ["Hipótesis de la propuesta", "Estado tras la ejecución"],
    ["La cobertura marginal alcanza 1 − α sin depender del modelo", "Consistente en esta partición (12 de 12 dentro de ±0.02); la garantía teórica no se demuestra con una partición"],
    ["Split conformal subcubre las clases minoritarias", "Apoyada: TWF con cobertura 0 en los seis escenarios de split"],
    ["Mondrian corrige la subcobertura por clase", "Apoyada para α ≤ 0.10; con 0.20 el soporte de TWF (9) no permite concluir"],
    ["La incertidumbre se traduce en tamaño, no en error de cobertura", "Apoyada: hgb y MLP difieren en tamaño y abstención; el error entre automáticas sí difiere (5.9 % frente a 8.2 %)"],
    ["Mondrian produce conjuntos más grandes", "Sí en las seis comparaciones; no implica más abstención (con α = 0.20 abstiene menos porque split da conjuntos vacíos)"],
    ["Las características físicas reducen el tamaño", "No evaluada: fuera del producto mínimo"],
    ["Existe un α óptimo por costo", "No evaluada: política de costos como extensión"],
], 0.7, 1.55, 11.9, [4.6, 7.3], size=11, row_h=0.55)
notes(s, "12:15–12:45. Criterio 3c y 3d: metodología honesta. Decir explícitamente qué no se evaluó.")

s = slide_base("Apéndice M", "Conceptos del curso aplicados y tema adicional investigado", None)
table(s, [
    ["Concepto del curso", "Dónde se aplica en el proyecto"],
    ["Tensores, capas lineales, activaciones", "MLP con nn.Linear y ReLU sobre lotes de 256 filas, en GPU"],
    ["Autodiferenciación y ciclo de entrenamiento", "backward(), zero_grad(), Adam; traza de pérdida, norma del gradiente y cambio de parámetros por época"],
    ["CPU frente a GPU, precisión float32", "Mediana de repeticiones, parámetros comparados con tolerancia, costo de transferencias"],
    ["Probabilidad, cuantiles y estadísticos de orden", "Rango ⌈(n + 1)(1 − α)⌉ y argumento de intercambiabilidad del cuantil conforme"],
    ["Comparación de modelos y métricas por clase", "HGB frente a MLP con la misma capa conforme; cobertura, tamaño, abstención, error automático"],
    ["Pipelines y fuga de información", "Preprocesamiento ajustado solo con entrenamiento; calibración y prueba separadas"],
    ["Reproducibilidad", "uv.lock, semillas, hash SHA-256, manifiesto con versiones y GPU, 112 pruebas, notebook"],
], 0.7, 1.55, 11.9, [3.9, 8.0], size=11, row_h=0.4)
card(s, 0.7, 5.0, 11.9, 1.75, "Tema adicional investigado: predicción conforme",
     "Vovk, Gammerman y Shafer (2005) para el marco y la validez; Papadopoulos et al. (2002) para el esquema split; Vovk (2012) para Mondrian; Lei et al. (2018) para las cotas de cobertura; Angelopoulos y Bates (2021) para la distribución de la cobertura con calibración fija; Romano et al. (2020) para APS; Geifman y El-Yaniv (2017) para clasificación selectiva. Fuera del alcance reducido: Docker, SQL, SVD y autovalores, embeddings aprendidos.", accent=ORANGE, body_size=11)
notes(s, "12:45–13:15. Criterios 3a y 3b. Recorrer la tabla en bloques de dos filas; nombrar las referencias del informe.")

s = slide_base("Apéndice N", "Recomendaciones y límites declarados", None)
card(s, 0.7, 1.55, 5.8, 5.2, "Recomendaciones", [
    "1. Operar con Mondrian y α = 0.10: el nivel más exigente con umbral finito para todas las clases. α = 0.05 solo con ≥ 19 ejemplos de calibración por clase.",
    "2. Repetir la partición con varias semillas y reportar la distribución de la cobertura por clase, no un punto.",
    "3. Sustituir 1 − p por APS: evita cuantiles saturados en 0 y conjuntos vacíos que hoy van a la vía humana.",
    "4. Definir la matriz de costos para elegir α por costo esperado y no solo por cobertura.",
    "5. Al recolectar datos, priorizar TWF: limita la calibración Mondrian y la medición de su cobertura.",
], accent=AQUA, body_size=12)
card(s, 6.8, 1.55, 5.8, 5.2, "Límites", [
    "Una sola partición: la garantía es en probabilidad sobre particiones; aquí hay una realización con semilla 42.",
    "TWF con 14 casos de calibración y 9 de prueba: cifras indicativas, no estimaciones precisas.",
    "Puntuación 1 − p saturada por modelos muy confiados: umbrales cercanos a 0 y conjuntos vacíos.",
    "Datos sintéticos y secuenciales: la garantía se afirma para partición aleatoria, no para despliegue en el tiempo.",
    "Sin selección de hiperparámetros ni costos asignados a la abstención.",
    "CPU y CUDA convergen a parámetros ligeramente distintos; las métricas del MLP corresponden a CUDA.",
], accent=ORANGE, body_size=12)
notes(s, "14:00–14:30. Recomendaciones concretas y límites declarados. Criterio 3d.")


s = slide_base("Apéndice A", "Métricas globales completas en prueba (n = 1 995)", None)
table(s, [
    ["Modelo", "Variante", "α", "Cobertura", "Tamaño medio", "Abstención", "Asistida", "Humana", "Automáticas", "Error automáticas"],
    ["hgb", "split", "0.05", "0.954", "0.96", "0.041", "0.000", "0.041", "1 914", "0.5 %"],
    ["hgb", "mondrian", "0.05", "0.959", "2.64", "1.000", "0.373", "0.627", "0", "—"],
    ["hgb", "split", "0.10", "0.909", "0.91", "0.088", "0.000", "0.088", "1 819", "0.3 %"],
    ["hgb", "mondrian", "0.10", "0.917", "1.32", "0.305", "0.286", "0.019", "1 387", "5.9 %"],
    ["hgb", "split", "0.20", "0.801", "0.80", "0.199", "0.000", "0.199", "1 598", "0.1 %"],
    ["hgb", "mondrian", "0.20", "0.799", "0.89", "0.146", "0.018", "0.129", "1 703", "7.8 %"],
    ["mlp", "split", "0.05", "0.954", "0.96", "0.041", "0.000", "0.041", "1 913", "0.5 %"],
    ["mlp", "mondrian", "0.05", "0.950", "2.07", "0.974", "0.881", "0.093", "52", "96.2 %"],
    ["mlp", "split", "0.10", "0.902", "0.90", "0.096", "0.000", "0.096", "1 803", "0.2 %"],
    ["mlp", "mondrian", "0.10", "0.903", "1.07", "0.085", "0.068", "0.017", "1 825", "8.2 %"],
    ["mlp", "split", "0.20", "0.808", "0.81", "0.191", "0.000", "0.191", "1 613", "0.1 %"],
    ["mlp", "mondrian", "0.20", "0.794", "0.85", "0.148", "0.002", "0.147", "1 699", "6.9 %"],
], 0.7, 1.55, 11.9, [0.9, 1.2, 0.7, 1.1, 1.3, 1.2, 1.0, 1.0, 1.3, 1.5], size=11, row_h=0.36)
notes(s, "Respaldo: tabla completa de metrics.csv.")

s = slide_base("Apéndice B", "Cobertura por clase en prueba: las seis configuraciones de cada modelo", None)
table(s, [
    ["Modelo", "Variante", "α", "Normal (1 929)", "TWF (9)", "HDF (21)", "PWF (19)", "OSF (17)"],
    ["hgb", "split", "0.05", "0.977", "0.000", "0.714", "0.000", "0.235"],
    ["hgb", "mondrian", "0.05", "0.958", "1.000", "0.952", "1.000", "1.000"],
    ["hgb", "split", "0.10", "0.938", "0.000", "0.143", "0.000", "0.059"],
    ["hgb", "mondrian", "0.10", "0.915", "1.000", "0.952", "1.000", "1.000"],
    ["hgb", "split", "0.20", "0.827", "0.000", "0.000", "0.000", "0.059"],
    ["hgb", "mondrian", "0.20", "0.795", "0.778", "0.952", "0.842", "1.000"],
    ["mlp", "split", "0.05", "0.969", "0.000", "0.571", "0.789", "0.412"],
    ["mlp", "mondrian", "0.05", "0.949", "1.000", "1.000", "1.000", "1.000"],
    ["mlp", "split", "0.10", "0.922", "0.000", "0.333", "0.632", "0.176"],
    ["mlp", "mondrian", "0.10", "0.901", "1.000", "0.952", "0.947", "1.000"],
    ["mlp", "split", "0.20", "0.831", "0.000", "0.048", "0.421", "0.000"],
    ["mlp", "mondrian", "0.20", "0.794", "0.778", "0.762", "0.947", "0.706"],
], 0.7, 1.55, 11.9, [1.0, 1.4, 0.8, 1.9, 1.7, 1.7, 1.7, 1.7], size=11, row_h=0.36)
notes(s, "Respaldo: metrics_by_class.csv. La figura del MLP está en el informe.")

s = slide_base("Apéndice C", "Umbrales calibrados por clase", None)
table(s, [
    ["Modelo", "Variante", "α", "Normal", "TWF", "HDF", "PWF", "OSF"],
    ["hgb", "split", "0.05", "1.9e-3", "1.9e-3", "1.9e-3", "1.9e-3", "1.9e-3"],
    ["hgb", "split", "0.10", "2.1e-5", "2.1e-5", "2.1e-5", "2.1e-5", "2.1e-5"],
    ["hgb", "mondrian", "0.05", "1.1e-4", "∞", "0.982", "1.000", "1.000"],
    ["hgb", "mondrian", "0.10", "6.7e-6", "1.000", "0.576", "1.000", "0.9999"],
    ["hgb", "mondrian", "0.20", "6.3e-7", "1.000", "0.163", "0.996", "0.9985"],
    ["mlp", "split", "0.05", "0.106", "0.106", "0.106", "0.106", "0.106"],
    ["mlp", "mondrian", "0.05", "0.052", "∞", "0.920", "1.000", "0.9985"],
    ["mlp", "mondrian", "0.10", "0.011", "0.997", "0.815", "1.000", "0.990"],
    ["mlp", "mondrian", "0.20", "7.4e-4", "0.980", "0.643", "0.663", "0.346"],
], 0.7, 1.55, 11.9, [1.0, 1.4, 0.8, 1.74, 1.74, 1.74, 1.74, 1.74], size=11, row_h=0.36)
text(s, ["Ninguno es exactamente cero. Split hgb con α = 0.10: n = 2 992, rango 2 694 → 2.1 × 10⁻⁵; el conjunto solo admite clases con p̂ ≥ 0.99998, y donde ninguna lo alcanza queda vacío.",
         "Mondrian: el umbral de Normal es casi cero porque 2 893 puntuaciones de Normal se concentran en 0; los de las fallas son altos porque el modelo duda en ellas. De ahí las falsas alarmas entre las decisiones automáticas."],
     0.7, 5.35, 11.9, 1.5, size=11, color=INK)
notes(s, "Respaldo: thresholds.csv y la sección de umbrales del informe.")

s = slide_base("Apéndice D", "Propuesta frente a entregado y mapa de la rúbrica", None)
table(s, [
    ["Criterio de la rúbrica", "Dónde se cubre"],
    ["1a–1d Presentación: herramienta TIC, coherencia, fluidez, tiempo", "Esta presentación: 10 min en dos bloques de 5 (Jaime y Roberth), con notas y tiempo por diapositiva"],
    ["2a Funcionamiento y objetivos de la propuesta", "Apéndice F (objetivos y alcance); README «Propuesta frente a entregado»"],
    ["2b Calidad y representatividad de los datos", "Diapositiva de datos y auditoría; apéndices G y H; data/README.md con la auditoría real"],
    ["2c Resultados relevantes e interpretables", "Resultados 1–3 y en planta; apéndices A–C y L; RESULTADOS.md; notebook con verificación"],
    ["2d Código organizado y reproducible", "Pipeline, 112 pruebas, un comando, manifiesto con hash y versiones, artifacts versionados"],
    ["3a Conceptos del curso", "Apéndice M; pipeline y método en la exposición; MLP, autodiferenciación, CPU/GPU, cuantiles"],
    ["3b Investigación adicional", "Método (cuantil corregido); apéndices I y E: argumento del rango y nueve referencias"],
    ["3c Metodología acertada", "Calibración separada, alphas prefijados, nada seleccionado con prueba, hipótesis contrastadas"],
    ["3d Conclusiones y recomendaciones", "Diapositiva de conclusiones con recomendaciones; apéndice N con los límites"],
], 0.7, 1.55, 11.9, [4.6, 7.3], size=11, row_h=0.46)
notes(s, "Respaldo: para el docente, dónde está cada criterio.")

s = slide_base("Apéndice E", "Referencias", None)
bullets(s, [
    "Vovk, V., Gammerman, A. y Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.",
    "Papadopoulos, H., Proedrou, K., Vovk, V. y Gammerman, A. (2002). Inductive confidence machines for regression. ECML 2002.",
    "Vovk, V. (2012). Conditional validity of inductive conformal predictors. ACML 2012.",
    "Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. y Wasserman, L. (2018). Distribution-free predictive inference for regression. JASA, 113(523).",
    "Angelopoulos, A. N. y Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511.",
    "Romano, Y., Sesia, M. y Candès, E. (2020). Classification with valid and adaptive coverage. NeurIPS 2020.",
    "Geifman, Y. y El-Yaniv, R. (2017). Selective classification for deep neural networks. NeurIPS 2017.",
    "Matzka, S. (2020). Explainable artificial intelligence for predictive maintenance applications. AI4I 2020.",
    "Dua, D. y Graff, C. AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository, id 601.",
], size=13, gap=6)
notes(s, "Respaldo: referencias completas, también en RESULTADOS.md.")

OUT.parent.mkdir(exist_ok=True)
prs.save(OUT)
total = sum(m for _, m in SCHEDULE)
principal = SLIDE_NO["n"] - sum(1 for _ in range(0))
print(f"guardado {OUT} · diapositivas: {len(prs.slides)} · tiempo principal: {total:g} min")
for title, m in SCHEDULE:
    print(f"  {m:4.2f}  {title}")
