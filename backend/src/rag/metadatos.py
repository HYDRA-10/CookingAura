"""Metadatos de las recetas: cómo se obtienen, cómo se normalizan y cómo se convierten en filtros de Milvus.

Cada fragmento indexado lleva, además del texto, estos campos (vacíos o 0 si no se conocen):

    nombre          nombre del plato
    tipo_comida     lista: desayuno, brunch, almuerzo, comida, merienda, cena, postre
    categoria       texto libre normalizado (legumbres, pescados, postres...)
    raciones        entero (0 = desconocido)
    tiempo_min      entero, minutos (0 = desconocido)
    dificultad      facil | media | dificil
    dieta           lista: vegana, vegetariana, pescetariana, sin gluten, sin lactosa, omnivora
    alergenos       lista de los 14 alérgenos oficiales (gluten, lacteos, huevo, crustaceos...)
    ingredientes    lista de nombres normalizados (sin tildes, en singular)
    origen_meta     de dónde salen: "tabla" (columnas de un recetario), "texto" (líneas "Etiqueta: valor") o "ninguno"

Fuentes:
- Recetarios en tabla (Excel, CSV): cada columna conocida se lee tal cual. Es lo más fiable.
- Texto con líneas "Tipo de comida: Cena", "Alérgenos: gluten"...: se leen esas líneas.
- Texto libre sin etiquetas: sin metadatos (origen "ninguno").

Al buscar, un criterio se aplica en dos pasos: primero las recetas que lo CUMPLEN y, si faltan resultados,
las que NO TIENEN ese dato. Las que lo incumplen no salen nunca (un desayuno nunca devuelve una comida).
"""
import json
import re
import unicodedata

# ------------------------------------------------------------------ vocabularios
TIPOS_COMIDA = ["desayuno", "brunch", "almuerzo", "comida", "merienda", "cena", "postre"]
_SINONIMOS_TIPO = {
    "desayunar": "desayuno", "desayunos": "desayuno",
    "almorzar": "almuerzo", "almuerzos": "almuerzo",
    "comer": "comida", "comidas": "comida", "plato principal": "comida", "principal": "comida",
    "merendar": "merienda", "meriendas": "merienda",
    "cenar": "cena", "cenas": "cena",
    "postres": "postre", "dulce": "postre",
}
# Qué etiquetas valen para cada tipo pedido (el almuerzo de mediodía es una comida; un postre vale de merienda)
COMPATIBLES = {
    "desayuno": ["desayuno", "brunch"],
    "brunch": ["brunch", "desayuno"],
    "almuerzo": ["almuerzo", "comida"],
    "comida": ["comida", "almuerzo"],
    "merienda": ["merienda", "postre"],
    "cena": ["cena"],
    "postre": ["postre"],
}

# Los 14 alérgenos de declaración obligatoria (UE) y las palabras que los delatan
ALERGENOS = {
    "gluten": r"gluten|trigo|cebada|centeno|avena|espelta|kamut|harina|\bpan\b|pasta|cuscus|celiac",
    "crustaceos": r"crustace|marisco|gamba|langostino|cigala|cangrejo|bogavante|langosta|necora|percebe",
    "huevo": r"huevo|\bclara|\byema|mayonesa",
    "pescado": r"pescado|merluza|salmon|atun|bacalao|sardina|anchoa|boqueron|lubina|dorada|trucha|caballa|rape",
    "cacahuete": r"cacahuete|mani\b",
    "soja": r"\bsoja|tofu|edamame|tempeh",
    "lacteos": r"lact|leche|queso|yogur|nata\b|mantequilla|kefir|requeson",
    "frutos de cascara": r"frutos (secos|de cascara)|nuez|nueces|almendra|avellana|anacardo|pistacho|pecana|macadamia",
    "apio": r"\bapio",
    "mostaza": r"mostaza",
    "sesamo": r"sesamo|tahini",
    "sulfitos": r"sulfito|\bvino\b|vinagre",
    "altramuces": r"altramuz|altramuces",
    "moluscos": r"molusco|marisco|mejillon|almeja|calamar|sepia|pulpo|berberecho|ostra|chipiron|navaja",
}
_CELDA_SIN_ALERGENOS = r"^(ninguno|ninguna|no|sin alergenos|-|n/?a)\b"

DIETAS = {
    "vegana": r"vegan",
    "vegetariana": r"vegetarian",
    "pescetariana": r"pescetarian|pescatarian",
    "sin gluten": r"sin gluten|celiac",
    "sin lactosa": r"sin lactosa",
    "omnivora": r"omnivor",
}
# Qué etiquetas de dieta cumplen cada dieta pedida (una receta vegana vale para un vegetariano)
DIETAS_COMPATIBLES = {
    "vegana": ["vegana"],
    "vegetariana": ["vegetariana", "vegana"],
    "pescetariana": ["pescetariana", "vegetariana", "vegana"],
    "sin gluten": ["sin gluten"],
    "sin lactosa": ["sin lactosa", "vegana"],
}

DIFICULTADES = {"facil": r"facil|sencill|baja", "media": r"media|intermedi|moderad", "dificil": r"dificil|alta|avanzad"}

# Cabeceras (o etiquetas "X: valor" en el texto) que corresponden a cada campo
CABECERAS = {
    "nombre": {"nombre", "receta", "plato", "titulo", "nombre de la receta", "nombre del plato"},
    "tipo_comida": {"tipo de comida", "tipo comida", "momento", "momento del dia", "ocasion", "comida del dia"},
    "categoria": {"categoria", "tipo de plato", "grupo", "seccion"},
    "raciones": {"raciones", "porciones", "comensales", "personas", "para"},
    "tiempo_min": {"tiempo", "tiempo total", "tiempo de preparacion", "duracion", "minutos", "tiempo (min)"},
    "dificultad": {"dificultad", "nivel"},
    "dieta": {"dieta", "tipo de dieta", "apto para"},
    "alergenos": {"alergenos", "alergias", "contiene", "alergenos que contiene"},
    "ingredientes": {"ingredientes"},
}

# Límites de los campos en Milvus
MAX_NOMBRE, MAX_CATEGORIA, MAX_ETIQUETA = 256, 64, 64
MAX_TIPOS, MAX_DIETAS, MAX_ALERGENOS, MAX_INGREDIENTES = 8, 8, 16, 64


# ------------------------------------------------------------------ utilidades
def normalizar(texto) -> str:
    texto = unicodedata.normalize("NFD", str(texto or "").lower())
    return " ".join("".join(c for c in texto if unicodedata.category(c) != "Mn").split())


def campo_de_cabecera(cabecera: str) -> str | None:
    """'Tiempo (min)' -> 'tiempo_min'; 'Alérgenos' -> 'alergenos'; desconocida -> None."""
    c = normalizar(cabecera).strip(" :")
    sin_parentesis = re.sub(r"\s*\(.*?\)", "", c).strip()
    for campo, nombres in CABECERAS.items():
        if c in nombres or sin_parentesis in nombres:
            return campo
    return None


def vacio() -> dict:
    return {"nombre": "", "tipo_comida": [], "categoria": "", "raciones": 0, "tiempo_min": 0, "dificultad": "",
            "dieta": [], "alergenos": [], "ingredientes": [], "origen_meta": "ninguno"}


def _entero(valor) -> int:
    m = re.search(r"\d+", str(valor or ""))
    return int(m.group()) if m else 0


def _minutos(valor) -> int:
    """'45', '45 min', '1 h 30 min', '2 horas', '1,5 h' -> minutos."""
    t = normalizar(valor)
    if not t:
        return 0
    horas = re.search(r"(\d+(?:[.,]\d+)?)\s*h", t)
    minutos = re.search(r"(\d+)\s*(?:min|')", t)
    if horas or minutos:
        total = float(horas.group(1).replace(",", ".")) * 60 if horas else 0
        return int(total + (int(minutos.group(1)) if minutos else 0))
    return _entero(t)


def tipos_comida(valor) -> list[str]:
    """'Comida o cena' -> ['comida', 'cena']; 'Desayunos' -> ['desayuno']."""
    texto = normalizar(valor)
    encontrados = []
    for trozo in re.split(r",|;|/|\by\b|\bo\b|\+", texto):
        trozo = trozo.strip(" .")
        tipo = _SINONIMOS_TIPO.get(trozo, trozo)
        if tipo in TIPOS_COMIDA and tipo not in encontrados:
            encontrados.append(tipo)
    if not encontrados:                         # texto libre: "Ideal para el desayuno"
        encontrados = [t for t in TIPOS_COMIDA if re.search(rf"\b{t}", texto)]
    return encontrados[:MAX_TIPOS]


def alergenos(valor) -> list[str]:
    """Celda 'Gluten, lácteos (lactosa), crustáceos (marisco)' -> ['gluten', 'lacteos', 'crustaceos']."""
    texto = normalizar(valor)
    if not texto or re.match(_CELDA_SIN_ALERGENOS, texto):
        return []
    encontrados = []
    for parte in re.split(r",|;|/|\by\b", texto):
        parte_sin_detalle = re.sub(r"\(.*?\)", "", parte).strip() or parte
        # El detalle entre paréntesis se ignora: "Crustáceos (marisco)" es solo crustáceos,
        # mientras que "Marisco" a secas marca crustáceos y moluscos
        for nombre, patron in ALERGENOS.items():
            if re.search(patron, parte_sin_detalle) and nombre not in encontrados:
                encontrados.append(nombre)
    return encontrados[:MAX_ALERGENOS]


def alergenos_de_preferencias(alergias: list[str]) -> list[str]:
    """Alergias del usuario ('marisco', 'nueces', 'lactosa') -> alérgenos oficiales para el filtro."""
    resultado = []
    for alergia in alergias or []:
        a = normalizar(alergia)
        for nombre, patron in ALERGENOS.items():
            if (re.search(patron, a) or nombre in a) and nombre not in resultado:
                resultado.append(nombre)
    return resultado


def dietas(valor) -> list[str]:
    texto = normalizar(valor)
    return [d for d, patron in DIETAS.items() if re.search(patron, texto)][:MAX_DIETAS]


def dificultad(valor) -> str:
    texto = normalizar(valor)
    return next((d for d, patron in DIFICULTADES.items() if re.search(patron, texto)), "")


# Palabras de medida al principio de un ingrediente: no forman parte de su nombre
_MEDIDAS = re.compile(
    r"^(?:\d+\s*)?(?:cucharadas?|cucharaditas?|dientes?|rebanadas?|trozos?|pizcas?|latas?|vasos?|tazas?|"
    r"manojos?|ramas?|ramitas?|hojas?|punados?|chorritos?|chorros?|sobres?|lonchas?|filetes?|lomos?|botes?|"
    r"paquetes?|medidas?(?: de vaso de yogur)?|unidades?|piezas?|cabezas?|tiras?|vaso de yogur)\s+(?:de\s+)?")


def ingredientes(valor) -> list[str]:
    """Celda de ingredientes ('- 600 g de patatas\\n- 6 huevos') -> ['patata', 'huevo'] (sin tildes, singular)."""
    from src.data import sqlitebd
    nombres = []
    for linea in re.split(r"\n|;|•", str(valor or "")):
        linea = linea.strip(" -*·\t")
        if len(linea) < 2:
            continue
        for nombre, _, _ in sqlitebd.parsear_lista(linea):
            nombre = re.sub(r"\(.*?\)", "", nombre).strip(" .")
            nombre = _MEDIDAS.sub("", normalizar(nombre)).strip()       # "2 dientes de ajo" -> "ajo"
            clave = sqlitebd._clave(nombre)[:MAX_ETIQUETA]
            if len(clave) > 1 and clave not in nombres:
                nombres.append(clave)
    return nombres[:MAX_INGREDIENTES]


def _aplicar(meta: dict, campo: str, valor) -> None:
    if campo == "nombre":
        meta["nombre"] = str(valor).strip()[:MAX_NOMBRE]
    elif campo == "tipo_comida":
        meta["tipo_comida"] = tipos_comida(valor)
    elif campo == "categoria":
        meta["categoria"] = normalizar(valor)[:MAX_CATEGORIA]
    elif campo == "raciones":
        meta["raciones"] = _entero(valor)
    elif campo == "tiempo_min":
        meta["tiempo_min"] = _minutos(valor)
    elif campo == "dificultad":
        meta["dificultad"] = dificultad(valor)
    elif campo == "dieta":
        meta["dieta"] = dietas(valor)
    elif campo == "alergenos":
        meta["alergenos"] = alergenos(valor)
    elif campo == "ingredientes":
        meta["ingredientes"] = ingredientes(valor)


# ------------------------------------------------------------------ obtención
def desde_fila(cabecera: list[str], fila: list) -> dict:
    """Metadatos de una fila de un recetario en tabla (fuente más fiable)."""
    meta = vacio()
    for titulo, valor in zip(cabecera, fila):
        campo = campo_de_cabecera(titulo or "")
        if campo and str(valor or "").strip():
            _aplicar(meta, campo, valor)
    meta["origen_meta"] = "tabla"
    return meta


_ETIQUETA = re.compile(r"^\s*(?:[-*•]\s*)?\**([A-Za-zÁÉÍÓÚáéíóúÑñ ()]{3,40}?)\**\s*:\s*(.+)$")


def desde_texto(texto: str) -> dict:
    """Metadatos a partir de líneas 'Etiqueta: valor' de un fragmento ('Tipo de comida: Cena', 'Tiempo: 30 min').
    La lista de ingredientes puede ocupar varias líneas tras 'Ingredientes:'."""
    meta = vacio()
    lineas = (texto or "").splitlines()
    encontrados = 0
    for i, linea in enumerate(lineas):
        m = _ETIQUETA.match(linea)
        if not m:
            continue
        campo = campo_de_cabecera(m.group(1))
        if not campo or campo == "nombre":
            continue
        valor = m.group(2)
        if campo == "ingredientes":
            # Las líneas siguientes que empiezan por guion también son ingredientes
            for siguiente in lineas[i + 1:]:
                if not re.match(r"^\s*[-*•]", siguiente):
                    break
                valor += "\n" + siguiente
        _aplicar(meta, campo, valor)
        encontrados += 1
    if encontrados:
        meta["origen_meta"] = "texto"
    return meta


def para_fragmento(texto: str, meta_receta: dict | None = None) -> dict:
    """Metadatos que se guardan con un fragmento: los de la receta (tabla) o, si no hay, los de su texto."""
    if meta_receta and meta_receta.get("origen_meta") == "tabla":
        return dict(meta_receta)
    return desde_texto(texto)


# ------------------------------------------------------------------ filtros para Milvus
def _lista(valores: list[str]) -> str:
    return json.dumps(list(valores), ensure_ascii=False)


def criterios(tipo_comida: str | None = None, alergias: list[str] | None = None, dieta: str | None = None,
              tiempo_max: int | None = None, fuente: str | None = None) -> list[tuple[str, str]]:
    """Criterios de búsqueda como pares (cumple, desconocido) de expresiones de Milvus.
    'desconocido' es None cuando el criterio no admite recetas sin el dato."""
    resultado = []
    if tipo_comida:
        tipo = _SINONIMOS_TIPO.get(normalizar(tipo_comida), normalizar(tipo_comida))
        if tipo in COMPATIBLES:
            resultado.append((f"ARRAY_CONTAINS_ANY(tipo_comida, {_lista(COMPATIBLES[tipo])})",
                              "ARRAY_LENGTH(tipo_comida) == 0"))
    oficiales = alergenos_de_preferencias(alergias or [])
    if oficiales:
        # Las alergias excluyen siempre: no hay paso "desconocido" (eso lo cubre la revisión del texto)
        resultado.append((f"not ARRAY_CONTAINS_ANY(alergenos, {_lista(oficiales)})", None))
    if dieta:
        d = normalizar(dieta)
        if d in DIETAS_COMPATIBLES:
            resultado.append((f"ARRAY_CONTAINS_ANY(dieta, {_lista(DIETAS_COMPATIBLES[d])})",
                              "ARRAY_LENGTH(dieta) == 0"))
    if tiempo_max:
        resultado.append((f"(tiempo_min > 0 and tiempo_min <= {int(tiempo_max)})", "tiempo_min == 0"))
    if fuente:
        resultado.append((f'fuente == {json.dumps(fuente, ensure_ascii=False)}', None))
    return resultado


def filtros_en_dos_pasos(lista_criterios: list[tuple[str, str]]) -> tuple[str, str | None]:
    """(estricto, flexible): el estricto exige cumplir todo; el flexible admite 'no se sabe' donde se permite.
    flexible es None si no aporta nada distinto."""
    if not lista_criterios:
        return "", None
    estricto = " and ".join(cumple for cumple, _ in lista_criterios)
    if not any(desconocido for _, desconocido in lista_criterios):
        return estricto, None
    flexible = " and ".join(f"({cumple} or {desconocido})" if desconocido else cumple
                            for cumple, desconocido in lista_criterios)
    return estricto, flexible


def cumple_tipo_en_texto(texto: str, tipo_comida: str | None) -> bool:
    """Para colecciones sin metadatos: descarta un fragmento cuya línea 'Tipo de comida' no encaja.
    Sin esa línea, se acepta (no se sabe)."""
    if not tipo_comida:
        return True
    tipo = _SINONIMOS_TIPO.get(normalizar(tipo_comida), normalizar(tipo_comida))
    etiquetados = desde_texto(texto)["tipo_comida"]
    return not etiquetados or bool(set(etiquetados) & set(COMPATIBLES.get(tipo, [tipo])))