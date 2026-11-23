"""Reglas rápidas de intención (sin LLM) para los mensajes más habituales del chat.

Se aplican antes que el clasificador de Ollama: son instantáneas y no fallan con frases como
"Hola, ¿qué puedo comer hoy?" (un saludo seguido de una petición).
"""
import re
import unicodedata

# Saludo al principio del mensaje ("Hola,", "Buenas tardes!", "Hey, qué tal")
_SALUDO = re.compile(
    r"^[\s¡¿!?,.]*(hola+|buenas(\s+(tardes|noches))?|buenos\s+dias|hey|ey|que\s+tal|saludos|holi)\b[\s,.!¡¿?:;-]*"
)

# Pide ideas de qué comer o cocinar -> flujo de ideas con botones
_IDEAS = re.compile(
    r"\bque\s+(puedo|podria|podemos|me\s+recomiendas|me\s+sugieres|me\s+aconsejas|hay\s+para|deberia)?\s*"
    r"(comer|cenar|cocinar|desayunar|merendar|almorzar|preparar|hacer)\b"
    r"|\bque\s+(como|ceno|cocino|desayuno|meriendo|almuerzo|preparo|hago\s+de\s+comer)\b"
    r"|\bque\s+hay\s+(hoy\s+)?(de|para)\s+(comer|cenar|desayunar|merendar)\b"
    r"|\b(dame|dime|proponme|sugiereme|recomiendame)\s+(alguna[s]?\s+|algun\s+|un[ao]?s?\s+)?(ideas?|platos?|algo\s+(de|para)\s+(comer|cenar))"
    r"|\bideas?\s+(de|para)\s+(comer|cenar|desayunar|merendar|cocinar|platos?|comida|cena|desayuno)"
    r"|\btengo\s+hambre\b|\bme\s+apetece\s+(algo|comer|cenar)\b"
)

# Acciones y consultas sobre los datos del usuario -> agente con herramientas, sin pasar por el clasificador.
# Se aceptan errores frecuentes al escribir o dictar ("dispensa", "añadé", "agregá").
_LUGAR = r"(despensa|dispensa|despensas|nevera|frigo\w*|congelador|alacena|armario|lista(\s+de\s+(la\s+)?compra)?|compra)"
_ASISTENTE = re.compile(
    # "añade unas patatas a la despensa", "quita la leche de la lista", "apunta tomates en la compra"
    r"\b(anad\w*|agreg\w*|apunt\w*|met\w*|pon\w*|guard\w*|quit\w*|elimin\w*|borr\w*|sac\w*|"
    r"descuent\w*|descont\w*|rest\w*|actualiz\w*|cambi\w*)\b.*\b" + _LUGAR + r"\b"
    # "¿me quedan huevos?", "¿qué tengo en la nevera?", "¿hay leche en la despensa?"
    r"|\b(me\s+queda\w*|queda\w*|tengo|hay)\b.*\b" + _LUGAR + r"\b"
    r"|\bme\s+queda\w*\b"
    r"|\b" + _LUGAR + r"\b.*\b(tengo|hay|queda\w*|no\s+(tengo|hay|queda\w*))\b"
    # "tengo cero gramos de aceitunas", "me quedan 2 huevos", "no tengo leche", "no queda aceite"
    r"|\b(tengo|quedan?|me\s+quedan?)\s+(cero|nada\s+de|ningun\w*|medio|media|un[oa]?s?|dos|tres|cuatro|cinco|"
    r"seis|siete|ocho|nueve|diez|\d+([.,]\d+)?)\b"
    r"|\bno\s+(me\s+)?(tengo|queda\w*|hay)\s+(nada\s+de\s+)?\w+"
    # Compra y consumo: "he comprado 2 kg de patatas", "he gastado la leche"
    r"|\bhe\s+(comprado|gastado|usado|terminado|acabado|cocinado|hecho|preparado|comido|cenado|desayunado)\b"
    r"|\bse\s+me\s+ha\w*\s+(acabado|terminado|agotado)\b"
    # Preferencias: "soy alérgico al marisco", "no me gusta el cilantro", "soy vegetariano"
    r"|\bsoy\s+(alergic\w*|intolerante|celiac\w*|vegan\w*|vegetarian\w*)\b"
    r"|\b(alergi\w*|alergic\w*|intoleran\w*)\b|\bno\s+me\s+gusta\w*\b|\bodio\b"
    # Historial y recetas guardadas
    r"|\b(historial|mis\s+recetas|recetas\s+guardadas|recetario)\b"
    # "¿Tengo alguna receta de lentejas guardada?"
    r"|\breceta\w*\b.*\bguardad\w*|\bguardad\w*\b.*\breceta\w*"
)

# Palabras de cocina: si aparecen, un "OTRO" del clasificador se corrige a ASISTENTE
_COCINA = re.compile(
    r"com(er|ida|o)\b|cocin|\bcen(a|ar|o)\b|desayun|merend|almorz|almuerzo|receta|ingredient|plato|despens|dispens|"
    r"nevera|frigo|hambre|menu|dieta|alerg|intoler|compra|horno|sarten|hervir|freir|asar|guiso|postre|"
    r"huevo|carne|pescado|verdura|fruta|legumbre|pasta|arroz|cocido|tortilla|salsa|patata|tomate|cebolla|ajo\b|"
    r"leche|queso|yogur|pollo|ternera|cerdo|jamon|atun|merluza|salmon|lenteja|garbanzo|judia|alubia|harina|"
    r"azucar|aceite|pan\b|mantequilla|zanahoria|lechuga|pimiento|calabac|berenjena|champi|seta|naranja|manzana|"
    r"platano|fresa|limon|especia|sal\b|pimienta|"
    # Nutrición y alimentación
    r"calori|nutri|proteina|vitamina|hidrato|grasa|fibra|saludable|sano\b|sana\b|alimento|alimentacion|"
    r"engord|adelgaz|digest|"
    # Más alimentos y platos frecuentes
    r"aguacate|chocolate|galleta|bizcocho|tarta|pizza|hamburguesa|sopa|crema\b|ensalada|bocadillo|sandwich|"
    r"marisco|gamba|mejillon|calamar|bacalao|sardina|chorizo|salchicha|embutido|pavo|cordero|conejo|"
    r"espinaca|brocoli|coliflor|pepino|lechuga|kiwi|uva|melon|sandia|pera|melocoton|nuez|almendra|"
    r"cafe\b|zumo|batido|infusion|vino\b|cerveza|bebida|cereal|avena|quinoa|tofu|seitan|"
    r"cuchar|vaso\b|taza\b|gramo|kilo|litro|racion|cocer|hornear|saltear|freir|plancha|microondas"
)


# "Prepárame una receta de ternera con teriyaki", "dame la receta del gazpacho", "¿cómo se hace una tortilla?"
_PLATO = re.compile(
    r"\breceta\s+(?:de|del|para\s+hacer|para\s+preparar)\s+(?:un[oa]?s?\s+|el\s+|la\s+|los\s+|las\s+)?(?P<plato>.+)"
    r"|\bcomo\s+(?:se\s+)?(?:hace|hago|hacer|prepara|preparo|preparar|elabora|elaboro)\s+"
    r"(?:un[oa]?s?\s+|el\s+|la\s+|los\s+|las\s+)?(?P<plato2>.+)"
)
# Preguntas sobre las recetas guardadas: las atiende el agente con su búsqueda, no se generan
_SOBRE_GUARDADAS = re.compile(r"guardad|mis\s+recetas|recetario|documento|tengo\s+alguna|hay\s+alguna")
# Coletillas finales que no forman parte del nombre del plato
_COLETILLAS = re.compile(
    r"[\s,]*(\b(aqui|ahora|ya|por\s+favor|porfa|porfavor|gracias|hoy|rapido|rapidamente|"
    r"para\s+(hoy|ahora|manana|esta\s+noche|cenar|comer|desayunar|merendar|almorzar|"
    r"la\s+cena|la\s+comida|el\s+desayuno|la\s+merienda|\w+\s+personas?)|para)\b[\s,.!?¿¡]*)+$"
)


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFD", (texto or "").lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def quitar_saludo(texto: str) -> str:
    """'Hola, ¿qué puedo comer hoy?' -> 'qué puedo comer hoy?' (conserva mayúsculas y tildes).
    Quita saludos encadenados: 'Hey, qué tal, ...'."""
    texto = (texto or "").strip()
    while True:
        m = _SALUDO.match(normalizar(texto))     # la normalización no cambia la longitud del texto
        if not m or m.end() == 0:
            return texto
        texto = texto[m.end():].strip()


def intencion_por_reglas(texto: str) -> str | None:
    """SALUDO, RECETA, ASISTENTE o None (None = que decida el clasificador)."""
    resto = normalizar(quitar_saludo(texto)).strip(" ,.!¡¿?")
    if not resto:
        return "SALUDO"
    if _IDEAS.search(resto):
        return "RECETA"
    if _ASISTENTE.search(resto):
        return "ASISTENTE"
    return None


def plato_pedido(texto: str) -> str | None:
    """Nombre del plato si el mensaje pide una receta concreta; None en otro caso.
    'Prepárame una receta de ternera con verduras y salsa teriyaki aquí' -> 'Ternera con verduras y salsa teriyaki'"""
    original = quitar_saludo(texto)
    normal = normalizar(original)                 # misma longitud que el original: los índices coinciden
    if _SOBRE_GUARDADAS.search(normal):
        return None
    m = _PLATO.search(normal)
    if not m:
        return None
    grupo = "plato" if m.group("plato") else "plato2"
    inicio, fin = m.span(grupo)
    fin_limpio = inicio + len(_COLETILLAS.sub("", normal[inicio:fin].rstrip(" .,!?¿¡")).rstrip(" .,!?¿¡"))
    plato = original[inicio:fin_limpio].strip(" .,!?¿¡")
    if len(plato) < 3:
        return None
    return plato[0].upper() + plato[1:]


def habla_de_cocina(texto: str) -> bool:
    return bool(_COCINA.search(normalizar(texto)))