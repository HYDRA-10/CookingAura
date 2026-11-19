"""Selección de ingredientes principales de la despensa según el tipo de comida.

Cuando el usuario pide ideas sin indicar ingredientes, pasar toda la despensa al LLM hace el prompt
enorme y mezcla cosas que no encajan (aceite, sal, especias, chocolate para una cena...). Aquí se eligen
hasta N ingredientes **principales** (sin condimentos ni básicos), **adecuados al tipo de comida** y
**variados** (de grupos distintos), al azar para que cada petición dé ideas diferentes.
"""
import random
import re
import unicodedata

# Grupos de ingredientes, por palabras clave del nombre (y de la categoría de la despensa).
# El orden importa: el primer grupo que coincide gana ("judía verde" es verdura antes que legumbre).
GRUPOS = [
    ("condimento", r"\bsal\b|pimienta|aceite|vinagre|azucar|edulcorante|sacarina|especia|oregano|comino|pimenton|"
                   r"canela|laurel|curry|nuez moscada|tomillo|romero|perejil|albahaca|cilantro|eneldo|jengibre|"
                   r"\bajo\b|ajos\b|caldo|pastilla|levadura|bicarbonato|maicena|harina|salsa|ketchup|mayonesa|"
                   r"mostaza|\bagua\b|hielo|colorante|azafran|vainilla|guindilla|clavo|cayena|tahini|vino\b|"
                   r"mantequilla|margarina|\bnata\b|manteca"),
    ("verdura", r"judia verde|judias verdes|tomate|cebolla|pimiento|calabacin|berenjena|zanahoria|lechuga|"
                r"espinaca|brocoli|coliflor|champinon|seta|puerro|pepino|calabaza|alcachofa|esparrago|\bcol\b|"
                r"repollo|acelga|apio|remolacha|maiz|guisante|canonigo|rucula|kale|verdura|hortaliza|ajete"),
    ("carne", r"pollo|pavo|ternera|cerdo|lomo|cordero|conejo|carne|hamburguesa|salchicha|chorizo|jamon|bacon|"
              r"panceta|costilla|solomillo|pechuga|muslo|contramuslo|albondiga|morcilla|fiambre|embutido|guanciale"),
    ("pescado", r"pescado|merluza|salmon|atun|bacalao|sardina|boqueron|lubina|dorada|gamba|langostino|mejillon|"
                r"calamar|sepia|pulpo|almeja|marisco|trucha|caballa|rape|anchoa|surimi|bonito"),
    ("huevo", r"huevo"),
    ("legumbre", r"lenteja|garbanzo|alubia|judia|frijol|soja|tofu|tempeh|seitan|legumbre|hummus|edamame"),
    ("pasta_arroz", r"arroz|pasta|macarron|espagueti|fideo|tallarin|lasana|quinoa|cuscus|tortilla de trigo|wrap"),
    ("pan_cereal", r"avena|\bpan\b|tostada|biscote|cereal|muesli|granola|tortita|bizcote|cereales"),
    ("tuberculo", r"patata|papa\b|boniato|batata|yuca"),
    ("lacteo", r"leche|yogur|queso|nata|mantequilla|kefir|requeson|cuajada|bebida vegetal|lacteo"),
    ("fruta", r"manzana|pera\b|platano|naranja|mandarina|fresa|kiwi|uva|melon|sandia|melocoton|nectarina|pina|"
              r"mango|limon|frutos rojos|arandano|frambuesa|mora\b|aguacate|cereza|ciruela|higo|granada|fruta"),
    ("dulce", r"chocolate|cacao|galleta|mermelada|miel|nuez|nueces|almendra|avellana|cacahuete|pistacho|"
              r"anacardo|frutos secos|bizcocho|crema de|dulce|pasa\b|pasas"),
]

# Grupos que encajan en cada tipo de comida. None = todo el grupo; un patrón = solo lo que coincide
# (en el desayuno, de la carne solo el fiambre; en la cena, de los lácteos solo queso y yogur...).
TODO = None
ADECUADOS = {
    "desayuno": {"lacteo": TODO, "fruta": TODO, "pan_cereal": TODO, "huevo": TODO, "dulce": TODO,
                 "carne": r"jamon|pavo|fiambre|lomo embuchado|bacon", "verdura": r"tomate|champinon|espinaca",
                 "pescado": r"salmon ahumado"},
    "brunch": {"lacteo": TODO, "fruta": TODO, "pan_cereal": TODO, "huevo": TODO, "verdura": TODO,
               "carne": TODO, "pescado": r"salmon|atun"},
    "comida": {"carne": TODO, "pescado": TODO, "huevo": TODO, "legumbre": TODO, "pasta_arroz": TODO,
               "tuberculo": TODO, "verdura": TODO},
    "almuerzo": {"carne": TODO, "pescado": TODO, "huevo": TODO, "legumbre": TODO, "pasta_arroz": TODO,
                 "tuberculo": TODO, "verdura": TODO},
    "cena": {"verdura": TODO, "pescado": TODO, "huevo": TODO, "legumbre": TODO,
             "carne": r"pollo|pavo|pechuga|lomo|ternera|conejo|solomillo|hamburguesa",
             "lacteo": r"queso|yogur|requeson|kefir"},
    "merienda": {"fruta": TODO, "dulce": TODO, "pan_cereal": TODO,
                 "lacteo": r"leche|yogur|queso|kefir|requeson|cuajada",
                 "carne": r"jamon|pavo|fiambre", "verdura": r"tomate"},
}
MINIMO = 6       # si para el tipo de comida hay menos, se completa con otros ingredientes principales


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFD", (texto or "").lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def grupo(nombre: str, categoria: str = "") -> str:
    """Grupo del ingrediente ('verdura', 'carne', 'condimento'...). 'otro' si no se reconoce."""
    texto = _normalizar(nombre)
    for nombre_grupo, patron in GRUPOS:
        if re.search(patron, texto):
            return nombre_grupo
    texto_categoria = _normalizar(categoria)
    for nombre_grupo, patron in GRUPOS:
        if texto_categoria and re.search(patron, texto_categoria):
            return nombre_grupo
    return "otro"


def _adecuado(ingrediente: dict, tipo: str | None) -> bool:
    if not tipo or tipo not in ADECUADOS:
        return True
    g = grupo(ingrediente["nombre"], ingrediente.get("categoria", ""))
    if g not in ADECUADOS[tipo]:
        return False
    patron = ADECUADOS[tipo][g]
    return patron is None or bool(re.search(patron, _normalizar(ingrediente["nombre"])))


def _variado(ingredientes: list[dict], n: int) -> list[dict]:
    """Hasta n ingredientes al azar, repartidos entre grupos (uno de cada grupo por turno)."""
    por_grupo: dict[str, list[dict]] = {}
    for i in ingredientes:
        por_grupo.setdefault(grupo(i["nombre"], i.get("categoria", "")), []).append(i)
    for lista in por_grupo.values():
        random.shuffle(lista)
    grupos = list(por_grupo)
    random.shuffle(grupos)
    elegidos = []
    while len(elegidos) < n and any(por_grupo.values()):
        for g in grupos:
            if por_grupo[g] and len(elegidos) < n:
                elegidos.append(por_grupo[g].pop())
    return elegidos


def seleccionar(despensa: list[dict], tipo_comida: str | None = None, n: int = 20,
                evitar: list[str] | None = None) -> list[dict]:
    """Hasta n ingredientes principales de la despensa, adecuados al tipo de comida y variados.
    despensa: [{"nombre", "cantidad", "unidad", "categoria"}]. evitar: alergias y lo que no le gusta."""
    evitar_norm = [_normalizar(e) for e in (evitar or []) if e]
    principales = [
        i for i in despensa
        if (i.get("cantidad") or 0) > 0
        and grupo(i["nombre"], i.get("categoria", "")) != "condimento"
        and not any(e and e in _normalizar(i["nombre"]) for e in evitar_norm)
    ]
    adecuados = [i for i in principales if _adecuado(i, tipo_comida)]
    elegidos = _variado(adecuados, n)
    if len(elegidos) < MINIMO:
        # Pocos ingredientes del tipo pedido: se completan con otros principales para que haya ideas
        resto = [i for i in principales if i not in elegidos]
        elegidos += _variado(resto, MINIMO - len(elegidos))
    return sorted(elegidos, key=lambda i: _normalizar(i["nombre"]))


def formatear(ingredientes: list[dict]) -> str:
    """Mismo formato que la herramienta ver_despensa (lo entiende el resto del grafo)."""
    def numero(n):
        return str(int(n)) if float(n).is_integer() else f"{round(n, 2):g}"
    if not ingredientes:
        return "No se encontraron ingredientes en la despensa."
    return "Ingredientes disponibles en la despensa:\n" + "\n".join(
        f"- {i['nombre'].capitalize()} ({numero(i['cantidad'])} {i.get('unidad') or ''})".replace(" )", ")")
        for i in ingredientes)