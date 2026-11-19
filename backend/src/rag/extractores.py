"""Extracción de texto de documentos de recetas, conservando la estructura (tablas, títulos, columnas).

Formatos: PDF, Word (.docx), Excel (.xlsx), CSV, texto (.txt) y Markdown (.md).

Convenciones de la salida (las usa el troceado del indexador):
- "## Título" marca un título o sección (DÍA 1, LÁCTEOS...). Cada fragmento lo repite si la sección se parte.
- Cada fila de tabla es una línea "Cabecera: valor · Cabecera: valor", así cada dato dice qué es
  aunque la fila acabe en otro fragmento.

En las tablas, una celda None significa "combinada con la de arriba" y se rellena con ese valor
(p. ej. DESAYUNO, que abarca tres filas, se repite en las tres).
"""
import csv
import io
import re
import statistics
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime

FORMATOS = (".pdf", ".docx", ".xlsx", ".csv", ".txt", ".md")

# Cabeceras que identifican un recetario en hoja de cálculo (comparadas sin tildes y en minúsculas)
COLUMNAS_NOMBRE = {"nombre", "receta", "plato", "titulo", "nombre de la receta", "nombre del plato"}
COLUMNAS_CONTENIDO = {"ingredientes", "pasos", "preparacion", "elaboracion", "instrucciones",
                      "procedimiento", "modo de preparacion"}


@dataclass
class Extraccion:
    texto: str = ""                                           # se trocea normalmente
    recetas: list[str] = field(default_factory=list)          # una receta completa por elemento
    avisos: list[str] = field(default_factory=list)           # problemas no fatales (p. ej. páginas sin OCR)

    @property
    def vacia(self) -> bool:
        return not self.texto.strip() and not any(r.strip() for r in self.recetas)


# ============================================================== utilidades de texto
def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.lower().strip())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def _arreglar_puntuacion(texto: str) -> str:
    """'Avena , cebada,centeno ( bulgur )' -> 'Avena, cebada, centeno (bulgur)'."""
    texto = re.sub(r"\s+([,.;:)\]])", r"\1", texto)
    texto = re.sub(r"([(\[¿¡])\s+", r"\1", texto)
    texto = re.sub(r",(?=[^\W\d_])", ", ", texto)        # coma pegada a una palabra (no en 1,5)
    return re.sub(r"[ \t]{2,}", " ", texto).strip()


def _unir_lineas(lineas: list[str]) -> str:
    """Une líneas de una celda o párrafo; una palabra partida con guion al final de línea se reúne."""
    resultado = ""
    for linea in (l.strip() for l in lineas):
        if not linea:
            continue
        if resultado.endswith("-") and len(resultado) > 1 and resultado[-2].isalpha():
            resultado = resultado[:-1] + linea
        else:
            resultado = f"{resultado} {linea}" if resultado else linea
    return _arreglar_puntuacion(resultado)


def _es_mayusculas(texto: str) -> bool:
    letras = [c for c in texto if c.isalpha()]
    return len(letras) >= 3 and all(c.isupper() for c in letras)


def _limpiar_celda(valor) -> str | None:
    if valor is None:
        return None                                          # combinada con la de arriba
    if isinstance(valor, float) and valor.is_integer():
        valor = int(valor)
    elif isinstance(valor, (datetime, date)):
        valor = valor.isoformat()
    return _arreglar_puntuacion(re.sub(r"\s+", " ", str(valor)))


# ============================================================== tablas → texto
def _rellenar_combinadas(filas: list[list]) -> list[list[str]]:
    """Sustituye cada None por el valor de la celda de arriba (celdas combinadas verticalmente)."""
    resultado, anterior = [], []
    for fila in filas:
        nueva = [(anterior[j] if j < len(anterior) else "") if c is None else c for j, c in enumerate(fila)]
        resultado.append(nueva)
        anterior = nueva
    return resultado


def _limpiar_filas(filas: list[list]) -> list[list]:
    """Celdas limpias (None se conserva), sin filas vacías y sin columnas vacías."""
    limpias = [[_limpiar_celda(c) for c in fila] for fila in filas]
    limpias = [f for f in limpias if any(f)]
    if not limpias:
        return []
    ancho = max(len(f) for f in limpias)
    limpias = [f + [""] * (ancho - len(f)) for f in limpias]
    columnas_con_datos = [j for j in range(ancho) if any(f[j] for f in limpias)]
    return [[f[j] for j in columnas_con_datos] for f in limpias]


def _valores(fila: list) -> list[str]:
    return [c for c in fila if c]


def _parece_cabecera(fila: list, resto: list) -> bool:
    """Es cabecera si tiene al menos dos textos cortos y no numéricos, y hay filas debajo."""
    valores = _valores(fila)
    if not resto or len(valores) < 2 or any(c is None for c in fila):
        return False
    return all(not re.fullmatch(r"[\d.,\s/%-]+", c) and len(c) <= 40 for c in valores)


def _etiqueta(cabecera: str) -> str:
    return cabecera.capitalize() if cabecera.isupper() else cabecera


def tabla_a_texto(filas: list[list]) -> str:
    """Convierte una tabla en líneas legibles: títulos como '## ...' y una línea por fila."""
    filas = _limpiar_filas(filas)
    if not filas:
        return ""
    lineas = []

    # Filas de título al principio (un solo valor que ocupa toda la fila): "DÍA 1", "VERDURAS"...
    while len(filas) > 1 and len(filas[0]) > 1 and len(_valores(filas[0])) == 1 and None not in filas[0]:
        lineas.append(f"## {_valores(filas[0])[0]}")
        filas = filas[1:]

    cabecera = None
    if _parece_cabecera(filas[0], filas[1:]):
        cabecera, filas = [_etiqueta(c or "") for c in filas[0]], filas[1:]
    filas = _rellenar_combinadas(filas)

    for fila in filas:
        valores = _valores(fila)
        if not valores:
            continue
        if len(valores) == 1:
            # Un solo valor: subtítulo dentro de la tabla ("FRUTAS", "Para la salsa") o dato sin etiqueta
            v = valores[0]
            es_seccion = len(fila) > 1 and _es_mayusculas(v.split("(")[0]) and len(v) <= 80
            lineas.append(f"## {v}" if es_seccion else v)
        elif cabecera:
            lineas.append(" · ".join(f"{h}: {v}" if h else v for h, v in zip(cabecera, fila) if v))
        elif len(fila) == 2:
            lineas.append(f"{fila[0]}: {fila[1]}")              # tabla clave-valor
        else:
            lineas.append(" | ".join(valores))
    return "\n".join(lineas)


# ============================================================== Word
def _extraer_docx(contenido: bytes) -> Extraccion:
    import docx
    from docx.table import Table, _Cell
    from docx.text.paragraph import Paragraph

    documento = docx.Document(io.BytesIO(contenido))
    partes = []
    # Recorrer el cuerpo en orden: así cada tabla queda junto a la receta a la que pertenece
    for elemento in documento.element.body.iterchildren():
        etiqueta = elemento.tag.rsplit("}", 1)[-1]
        if etiqueta == "p":
            parrafo = Paragraph(elemento, documento)
            texto = parrafo.text.strip()
            if not texto:
                continue
            estilo = (parrafo.style.name if parrafo.style is not None else "").lower()
            es_titulo = estilo.startswith(("heading", "título", "titulo", "title"))
            partes.append(f"## {texto}" if es_titulo else texto)
        elif etiqueta == "tbl":
            tabla = Table(elemento, documento)
            filas = []
            for fila in tabla.rows:
                celdas = []
                for tc in fila._tr.tc_lst:
                    # vMerge="continue": la celda sigue a la de arriba (combinación vertical)
                    continua = getattr(tc, "vMerge", None) == "continue"
                    celdas.append(None if continua else _Cell(tc, tabla).text)
                    # gridSpan: una celda combinada horizontalmente ocupa varias columnas, su texto va una vez
                    celdas.extend([""] * (int(tc.grid_span or 1) - 1))
                filas.append(celdas)
            texto = tabla_a_texto(filas)
            if texto:
                partes.append(texto)
    return Extraccion(texto="\n".join(partes))


# ============================================================== PDF
@dataclass
class _Pieza:
    x0: float
    y0: float
    x1: float
    y1: float
    texto: str = ""
    titulo: bool = False
    filas: list | None = None          # si es una tabla: filas (None = combinada con la de arriba)
    columnas: list | None = None       # si es una tabla: rangos x de sus columnas

    @property
    def es_tabla(self) -> bool:
        return self.filas is not None


def _abrir_pdf(contenido: bytes):
    try:
        import pymupdf
    except ImportError:            # versiones antiguas solo exponen el módulo como 'fitz'
        import fitz as pymupdf
    return pymupdf, pymupdf.open(stream=contenido, filetype="pdf")


def _lineas_visuales(palabras: list) -> list[list]:
    """Agrupa palabras (x0, y0, x1, y1, texto, ...) en líneas visuales, de arriba abajo y de izquierda a derecha."""
    lineas = []
    for p in sorted(palabras, key=lambda p: ((p[1] + p[3]) / 2, p[0])):
        centro, alto = (p[1] + p[3]) / 2, p[3] - p[1]
        if lineas and abs(centro - lineas[-1][0]) <= max(alto, 1) * 0.5:
            lineas[-1][1].append(p)
        else:
            lineas.append([centro, [p]])
    return [sorted(ps, key=lambda p: p[0]) for _, ps in lineas]


def _palabras_en(zona, palabras: list) -> list:
    x0, y0, x1, y1 = zona
    return [p for p in palabras if x0 <= (p[0] + p[2]) / 2 <= x1 and y0 <= (p[1] + p[3]) / 2 <= y1]


def _texto_en(zona, palabras: list) -> str:
    return _unir_lineas([" ".join(p[4] for p in linea) for linea in _lineas_visuales(_palabras_en(zona, palabras))])


def _fila_cruza_columnas(zona_fila, columnas: list, palabras: list) -> bool:
    """True si el texto de la fila fluye por encima de los límites de columna: es una celda combinada
    que el detector ha partido (p. ej. un título que ocupa todo el ancho sin bordes internos)."""
    limites = [c[1] for c in columnas[:-1] if c]
    for linea in _lineas_visuales(_palabras_en(zona_fila, palabras)):
        for a, b in zip(linea, linea[1:]):
            for limite in limites:
                if a[2] <= limite + 1 and b[0] >= limite - 1 and b[0] - a[2] < 5:
                    return True
        for p in linea:
            if any(p[0] < limite - 2 and p[2] > limite + 2 for limite in limites):
                return True
    return False


def _tabla_pdf(tabla, palabras: list) -> tuple[list[list], list]:
    """Filas de una tabla de PyMuPDF con el texto de cada celda y las combinaciones resueltas."""
    filas_geo = tabla.rows
    referencia = max(filas_geo, key=lambda f: sum(c is not None for c in f.cells))
    columnas = [(c[0], c[2]) if c is not None else None for c in referencia.cells]
    n = len(columnas)

    filas = []
    for fila in filas_geo:
        celdas = list(fila.cells) + [None] * (n - len(fila.cells))
        if _fila_cruza_columnas(fila.bbox, columnas, palabras):
            filas.append([_texto_en(fila.bbox, palabras)] + [""] * (n - 1))
            continue
        valores = []
        for j, celda in enumerate(celdas):
            if celda is not None:
                valores.append(_texto_en(celda, palabras))
                continue
            # Celda cubierta: por la de su izquierda (combinación horizontal) o por la de arriba (vertical)
            izquierda = next((celdas[k] for k in range(j - 1, -1, -1) if celdas[k] is not None), None)
            limite_columna = columnas[j][1] if columnas[j] else None
            if izquierda is not None and limite_columna is not None and izquierda[2] >= limite_columna - 2:
                valores.append("")
            else:
                valores.append(None)
        filas.append(valores)
    return filas, columnas


def _piezas_texto(pagina, zonas: list, tam_cuerpo: float) -> list[_Pieza]:
    """Texto fuera de las tablas, por tramos de líneas; las líneas en letra grande se marcan como título."""
    piezas = []
    for bloque in pagina.get_text("dict")["blocks"]:
        if bloque.get("type") != 0:
            continue                                           # imágenes
        tramo, titulo_tramo = [], None

        def cerrar():
            if tramo:
                x0 = min(l[0] for l in tramo); y0 = min(l[1] for l in tramo)
                x1 = max(l[2] for l in tramo); y1 = max(l[3] for l in tramo)
                piezas.append(_Pieza(x0, y0, x1, y1, _unir_lineas([l[4] for l in tramo]), bool(titulo_tramo)))
                tramo.clear()

        for linea in bloque.get("lines", []):
            x0, y0, x1, y1 = linea["bbox"]
            texto = "".join(s["text"] for s in linea["spans"]).strip()
            dentro = any(z[0] <= (x0 + x1) / 2 <= z[2] and z[1] <= (y0 + y1) / 2 <= z[3] for z in zonas)
            if not texto or dentro:
                cerrar()
                continue
            spans = [s for s in linea["spans"] if s["text"].strip()]
            tam = max((s["size"] for s in spans), default=tam_cuerpo)
            negrita = bool(spans) and all(s["flags"] & 16 or re.search(r"bold|black|heavy", s["font"], re.I)
                                          for s in spans)
            # Título: letra claramente mayor, algo mayor y en negrita, o negrita en mayúsculas ("INGREDIENTES")
            es_titulo = len(texto) <= 100 and (
                tam >= tam_cuerpo * 1.3
                or (negrita and tam >= tam_cuerpo * 1.1)
                or (negrita and tam >= tam_cuerpo * 0.95 and _es_mayusculas(texto) and len(texto) <= 60)
            )
            if tramo and es_titulo != titulo_tramo:
                cerrar()                                       # un título no se mezcla con el párrafo
            titulo_tramo = es_titulo
            tramo.append((x0, y0, x1, y1, texto))
        cerrar()
    return piezas


def _absorber_continuacion(piezas: list[_Pieza], tabla_previa: _Pieza, palabras: list) -> list[_Pieza]:
    """Si la página empieza con filas sueltas alineadas con las columnas de la última tabla de la página
    anterior, son la continuación de esa tabla: se añaden a ella en lugar de quedar como texto suelto."""
    primera_tabla = min((p.y0 for p in piezas if p.es_tabla), default=float("inf"))
    iniciales = [p for p in piezas if not p.es_tabla and p.y1 <= primera_tabla]
    if not iniciales or any(p.titulo for p in iniciales):
        return piezas
    zona = (min(p.x0 for p in iniciales) - 1, min(p.y0 for p in iniciales) - 1,
            max(p.x1 for p in iniciales) + 1, max(p.y1 for p in iniciales) + 1)
    columnas = tabla_previa.columnas
    filas = []
    for linea in _lineas_visuales(_palabras_en(zona, palabras)):
        fila = [[] for _ in columnas]
        for p in linea:
            centro = (p[0] + p[2]) / 2
            j = next((j for j, c in enumerate(columnas) if c and c[0] - 2 <= centro <= c[1] + 2), None)
            if j is None:
                return piezas                                  # algo no encaja en ninguna columna
            fila[j].append(p[4])
        if sum(bool(c) for c in fila) < 2:
            return piezas                                      # no parece una fila de la tabla
        filas.append([_unir_lineas([" ".join(c)]) for c in fila])

    # Celdas vacías en columnas que venían combinadas (p. ej. CENA) siguen a la de arriba
    ultima = tabla_previa.filas[-1] if tabla_previa.filas else []
    for fila in filas:
        for j, valor in enumerate(fila):
            if not valor and j < len(ultima) and ultima[j] is None:
                fila[j] = None
    tabla_previa.filas.extend(filas)
    return [p for p in piezas if p not in iniciales]


def _agrupar_etiquetas(piezas: list[_Pieza]) -> list[_Pieza]:
    """Une 'etiqueta | valor' maquetados uno al lado del otro (sin tabla): 'FRUTAS: Albaricoque, ...'."""
    resultado, usadas = [], set()
    textos = [p for p in piezas if not p.es_tabla and not p.titulo]
    for etiqueta in textos:
        if id(etiqueta) in usadas or len(etiqueta.texto) > 40:
            continue
        alto = etiqueta.y1 - etiqueta.y0
        for valor in textos:
            if valor is etiqueta or id(valor) in usadas:
                continue
            solape = min(etiqueta.y1, valor.y1) - max(etiqueta.y0, valor.y0)
            hueco = valor.x0 - etiqueta.x1
            if 0 <= hueco < 60 and solape >= alto * 0.5:
                texto = etiqueta.texto.rstrip(":")
                valor.texto = f"{texto}: {valor.texto}"
                valor.x0, valor.y0 = etiqueta.x0, min(etiqueta.y0, valor.y0)
                usadas.add(id(etiqueta))
                break
    return [p for p in piezas if id(p) not in usadas]


def _ordenar(piezas: list[_Pieza], ancho_pagina: float) -> list[_Pieza]:
    """Orden de lectura con columnas: lo que ocupa todo el ancho separa franjas; dentro de cada franja,
    primero la columna izquierda entera y después la derecha."""
    medio, margen = ancho_pagina / 2, ancho_pagina * 0.05

    def lado(p: _Pieza) -> str:
        if p.x1 <= medio + margen:
            return "izq"
        if p.x0 >= medio - margen:
            return "der"
        return "ancho"

    orden, franja = [], []

    def volcar():
        for l in ("izq", "der"):
            orden.extend(sorted((p for p in franja if lado(p) == l), key=lambda p: (p.y0, p.x0)))
        franja.clear()

    for p in sorted(piezas, key=lambda p: (p.y0, p.x0)):
        if lado(p) == "ancho":
            volcar()
            orden.append(p)
        else:
            franja.append(p)
    volcar()
    return orden


def _tam_cuerpo(documento) -> float:
    """Tamaño de letra más habitual del documento (ponderado por caracteres)."""
    tamanos = []
    for pagina in documento:
        for bloque in pagina.get_text("dict")["blocks"]:
            if bloque.get("type") == 0:
                for linea in bloque["lines"]:
                    for s in linea["spans"]:
                        tamanos.extend([round(s["size"], 1)] * len(s["text"].strip()))
    return statistics.median(tamanos) if tamanos else 10.0


def _ocr_pagina(pagina) -> str:
    """OCR de una página escaneada (necesita Tesseract y su idioma español en el contenedor).
    Si falta el paquete de español, se intenta en inglés (reconoce el texto, aunque peor las tildes)."""
    for idioma in ("spa+eng", "eng"):
        try:
            tp = pagina.get_textpage_ocr(language=idioma, dpi=200, full=True)
            texto = pagina.get_text(textpage=tp).strip()
            if texto:
                return texto
        except Exception:
            continue
    return ""


def _extraer_pdf(contenido: bytes) -> Extraccion:
    pymupdf, documento = _abrir_pdf(contenido)
    resultado = Extraccion()
    with documento:
        tam_cuerpo = _tam_cuerpo(documento)
        todas: list[_Pieza] = []
        for n, pagina in enumerate(documento, start=1):
            palabras = pagina.get_text("words")
            try:
                tablas = pagina.find_tables().tables if hasattr(pagina, "find_tables") else []
            except Exception:
                tablas = []

            piezas = []
            for tabla in tablas:
                filas, columnas = _tabla_pdf(tabla, palabras)
                x0, y0, x1, y1 = tabla.bbox
                piezas.append(_Pieza(x0, y0, x1, y1, filas=filas, columnas=columnas))
            zonas_tabla = [(p.x0, p.y0, p.x1, p.y1) for p in piezas]
            piezas += _piezas_texto(pagina, zonas_tabla, tam_cuerpo)

            if not piezas and pagina.get_images():
                # Página sin texto pero con imágenes: probablemente escaneada
                texto = _ocr_pagina(pagina)
                if not texto:
                    resultado.avisos.append(f"La página {n} es una imagen (documento escaneado) y no se pudo leer: "
                                            "hace falta OCR (Tesseract) en el backend.")
                    continue
                r = pagina.rect
                piezas = [_Pieza(r.x0, r.y0, r.x1, r.y1, texto)]
                resultado.avisos.append(f"La página {n} se ha leído con OCR: revisa que el texto sea correcto.")

            # Tabla que viene de la página anterior
            if todas and todas[-1].es_tabla:
                piezas = _absorber_continuacion(piezas, todas[-1], palabras)

            piezas = _agrupar_etiquetas(piezas)
            todas.extend(_ordenar(piezas, pagina.rect.width))

    partes = []
    for p in todas:
        if p.es_tabla:
            texto = tabla_a_texto(p.filas)
        else:
            texto = f"## {p.texto}" if p.titulo else p.texto
        if texto:
            partes.append(texto)
    resultado.texto = _unir_titulos("\n".join(partes))
    return resultado


def _unir_titulos(texto: str) -> str:
    """Títulos consecutivos ('## LÁCTEOS VEGETALES' + '## (preferible...)') en una sola línea."""
    lineas = []
    for linea in texto.splitlines():
        if linea.startswith("## ") and lineas and lineas[-1].startswith("## ") \
                and (linea[3:4] in "([" or lineas[-1].endswith(("-", ","))):
            lineas[-1] = f"{lineas[-1]} {linea[3:]}"
        else:
            lineas.append(linea)
    return "\n".join(lineas)


# ============================================================== Excel / CSV
def _hojas_xlsx(contenido: bytes) -> dict[str, list[list]]:
    from openpyxl import load_workbook

    # Sin read_only para poder leer las celdas combinadas
    libro = load_workbook(io.BytesIO(contenido), data_only=True)
    hojas = {}
    for hoja in libro.worksheets:
        filas = [["" if v is None else v for v in fila] for fila in hoja.iter_rows(values_only=True)]
        for rango in hoja.merged_cells.ranges:
            for r in range(rango.min_row, rango.max_row + 1):
                for c in range(rango.min_col, rango.max_col + 1):
                    if (r, c) == (rango.min_row, rango.min_col) or r > len(filas) or c > len(filas[r - 1]):
                        continue
                    # Debajo de la celda principal y en su columna: sigue a la de arriba; el resto, vacío
                    filas[r - 1][c - 1] = None if c == rango.min_col else ""
        hojas[hoja.title] = filas
    libro.close()
    return hojas


def _hojas_csv(contenido: bytes) -> dict[str, list[list]]:
    texto = _decodificar(contenido)
    try:
        dialecto = csv.Sniffer().sniff(texto[:5000], delimiters=",;\t|")
    except csv.Error:
        # Excel en español exporta con ';'
        primera = texto.splitlines()[0] if texto else ""
        dialecto = csv.excel
        if primera.count(";") > primera.count(","):
            dialecto = type("PuntoYComa", (csv.excel,), {"delimiter": ";"})
    return {"CSV": list(csv.reader(io.StringIO(texto), dialecto))}


def _recetario(filas: list[list]) -> list[str] | None:
    """Si la tabla es un recetario (columna de nombre + ingredientes/pasos), una receta por fila."""
    if len(filas) < 2:
        return None
    cabecera = [c or "" for c in filas[0]]
    normalizadas = [_normalizar(c) for c in cabecera]
    col_nombre = next((i for i, c in enumerate(normalizadas) if c in COLUMNAS_NOMBRE), None)
    if col_nombre is None or not any(c in COLUMNAS_CONTENIDO for c in normalizadas):
        return None

    recetas = []
    for fila in _rellenar_combinadas(filas[1:]):
        nombre = fila[col_nombre].strip() if col_nombre < len(fila) else ""
        if not nombre:
            continue
        lineas = [nombre]                      # primera línea = nombre del plato (título de la idea)
        for i, (titulo, valor) in enumerate(zip(cabecera, fila)):
            if i != col_nombre and str(valor).strip():
                lineas.append(f"{titulo or 'Detalle'}: {str(valor).strip()}")
        recetas.append("\n".join(lineas))
    return recetas


def _celdas_hoja(filas: list[list]) -> list[list]:
    """Celdas limpias conservando los saltos de línea (útiles en recetas) y las marcas None."""
    limpias = []
    for fila in filas:
        celdas = [c if c is None else (c.strip() if isinstance(c, str) else _limpiar_celda(c)) for c in fila]
        if any(celdas):
            limpias.append(celdas)
    if not limpias:
        return []
    ancho = max(len(f) for f in limpias)
    limpias = [f + [""] * (ancho - len(f)) for f in limpias]
    columnas = [j for j in range(ancho) if any(f[j] for f in limpias)]
    return [[f[j] for j in columnas] for f in limpias]


def _extraer_hojas(hojas: dict[str, list[list]]) -> Extraccion:
    resultado = Extraccion()
    textos = []
    for nombre_hoja, filas in hojas.items():
        celdas = _celdas_hoja(filas)
        if not celdas:
            continue
        recetas = _recetario(celdas)
        if recetas:
            resultado.recetas.extend(recetas)
        else:
            texto = tabla_a_texto(celdas)
            if texto:
                textos.append(f"## {nombre_hoja}\n{texto}" if len(hojas) > 1 else texto)
    resultado.texto = "\n".join(textos)
    return resultado


# ============================================================== texto plano
def _decodificar(contenido: bytes) -> str:
    for codificacion in ("utf-8-sig", "cp1252"):
        try:
            return contenido.decode(codificacion)
        except UnicodeDecodeError:
            continue
    return contenido.decode("latin-1")


def _extraer_texto_plano(contenido: bytes, markdown: bool) -> Extraccion:
    texto = _decodificar(contenido)
    if markdown:
        # Títulos de cualquier nivel como '## ' y tablas Markdown como filas con cabecera
        lineas, tabla = [], []

        def volcar_tabla():
            if tabla:
                filas = [[c.strip() for c in l.strip().strip("|").split("|")] for l in tabla
                         if not re.fullmatch(r"\|?[\s:|-]+\|?", l.strip())]
                lineas.append(tabla_a_texto(filas))
                tabla.clear()

        for linea in texto.splitlines():
            if linea.strip().startswith("|"):
                tabla.append(linea)
                continue
            volcar_tabla()
            m = re.match(r"\s*#{1,6}\s+(.*)", linea)
            lineas.append(f"## {m.group(1).strip()}" if m else linea)
        volcar_tabla()
        texto = "\n".join(lineas)
    return Extraccion(texto=texto)


# ============================================================== entrada
def extraer_documento(nombre: str, contenido: bytes) -> Extraccion:
    """Extrae el contenido según la extensión. Lanza ValueError si el formato no está soportado."""
    extension = "." + nombre.lower().rsplit(".", 1)[-1] if "." in nombre else ""
    if extension == ".pdf":
        return _extraer_pdf(contenido)
    if extension == ".docx":
        return _extraer_docx(contenido)
    if extension == ".xlsx":
        return _extraer_hojas(_hojas_xlsx(contenido))
    if extension == ".csv":
        return _extraer_hojas(_hojas_csv(contenido))
    if extension in (".txt", ".md"):
        return _extraer_texto_plano(contenido, markdown=extension == ".md")
    if extension == ".xls":
        raise ValueError("El formato .xls antiguo no está soportado: guárdalo como .xlsx desde Excel.")
    if extension == ".doc":
        raise ValueError("El formato .doc antiguo no está soportado: guárdalo como .docx desde Word.")
    raise ValueError(f"Formato no soportado. Sube un archivo {', '.join(FORMATOS)}.")