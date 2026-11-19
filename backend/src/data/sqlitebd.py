"""Acceso a los datos del usuario en SQLite (data/ingredients.db).

Lo usan las herramientas MCP (para el agente del chat) y los nodos del grafo.
Tablas: ingredients (despensa), preferencias, lista_compra, historial.
"""
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

RUTA_BD = Path(__file__).resolve().parents[2] / "data" / "ingredients.db"

TIPOS_PREFERENCIA = {
    "alergia": "Alergias e intolerancias",
    "dieta": "Dieta",
    "no_me_gusta": "No le gusta",
    "me_gusta": "Le gusta",
}

# Conversión entre unidades de la misma magnitud (a gramos o mililitros)
_UNIDADES = {
    "g": ("masa", 1), "gr": ("masa", 1), "gramo": ("masa", 1), "gramos": ("masa", 1),
    "kg": ("masa", 1000), "kilo": ("masa", 1000), "kilos": ("masa", 1000),
    "ml": ("volumen", 1), "cl": ("volumen", 10), "dl": ("volumen", 100),
    "l": ("volumen", 1000), "litro": ("volumen", 1000), "litros": ("volumen", 1000),
    "unidad": ("unidad", 1), "unidades": ("unidad", 1), "ud": ("unidad", 1), "uds": ("unidad", 1),
}


@contextmanager
def conexion():
    RUTA_BD.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(RUTA_BD)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ingredients (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, quantity TEXT NOT NULL,
                category TEXT, unit TEXT);
            CREATE TABLE IF NOT EXISTS preferencias (
                id INTEGER PRIMARY KEY AUTOINCREMENT, tipo TEXT NOT NULL, valor TEXT NOT NULL,
                UNIQUE (tipo, valor));
            CREATE TABLE IF NOT EXISTS lista_compra (
                id INTEGER PRIMARY KEY AUTOINCREMENT, producto TEXT NOT NULL UNIQUE, fecha TEXT);
            CREATE TABLE IF NOT EXISTS historial (
                id INTEGER PRIMARY KEY AUTOINCREMENT, plato TEXT NOT NULL, fecha TEXT NOT NULL, comentario TEXT);
        """)
        yield conn
        conn.commit()
    finally:
        conn.close()


# ------------------------------------------------------------------ utilidades
def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFD", (texto or "").lower().strip())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def _singular(palabra: str) -> str:
    """Raíz que iguala singular y plural: tomate/tomates -> tomat, limón/limones -> limon, nuez/nueces -> nuez."""
    if palabra.endswith("ces") and len(palabra) > 4:
        return palabra[:-3] + "z"
    if palabra.endswith("s") and len(palabra) > 3:
        palabra = palabra[:-1]
    if palabra.endswith("e") and len(palabra) > 3:
        palabra = palabra[:-1]
    return palabra


def _clave(nombre: str) -> str:
    """'Huevos', 'huevo', 'HUEVOS ' -> 'huevo': para encontrar un ingrediente aunque cambie el plural o las tildes."""
    return " ".join(_singular(p) for p in _normalizar(nombre).split())


def a_numero(valor) -> float | None:
    try:
        return float(str(valor).replace(",", ".").strip())
    except (TypeError, ValueError):
        return None


def formatear_numero(n: float) -> str:
    return str(int(n)) if float(n).is_integer() else f"{round(n, 2):g}"


def _convertir(cantidad: float, desde: str, hacia: str) -> float | None:
    """Convierte entre unidades de la misma magnitud (g↔kg, ml↔l). None si no es posible."""
    a, b = _UNIDADES.get(_normalizar(desde)), _UNIDADES.get(_normalizar(hacia))
    if not a or not b or a[0] != b[0]:
        return None
    return cantidad * a[1] / b[1]


def parsear_lista(texto: str) -> list[tuple[str, float | None, str]]:
    """'huevos 2, 0.5 litros de leche, harina 200 g, sal' -> [(nombre, cantidad, unidad), ...]"""
    elementos = []
    for trozo in re.split(r"[,;\n]+|\s+y\s+", texto or ""):
        trozo = trozo.strip(" .")
        if not trozo:
            continue
        numero = r"(\d+(?:[.,]\d+)?)"
        # "2 huevos", "0.5 litros de leche", "200 g de harina", "2 huevos camperos"
        m = re.match(rf"^{numero}\s*(?:([a-zA-Záéíóú]+)\s+)?(?:de\s+)?(.+)$", trozo)
        if m:
            primera, resto = m.group(2), m.group(3).strip()
            if primera and _normalizar(primera) in _UNIDADES:
                elementos.append((resto, a_numero(m.group(1)), primera))
            else:
                # Lo que seguía al número no era una unidad: forma parte del nombre ("2 huevos camperos")
                elementos.append((f"{primera} {resto}".strip() if primera else resto, a_numero(m.group(1)), ""))
            continue
        # "huevos 2", "leche 0.5 litros", "harina 200 g"
        m = re.match(rf"^(.+?)\s+{numero}\s*([a-zA-Záéíóú]+)?$", trozo)
        if m:
            elementos.append((m.group(1).strip(), a_numero(m.group(2)), m.group(3) or ""))
            continue
        elementos.append((trozo, None, ""))
    return elementos


# ------------------------------------------------------------------ despensa
def _fila(conn, nombre: str):
    """Busca un ingrediente por nombre (sin importar plural, mayúsculas ni tildes)."""
    clave = _clave(nombre)
    for fila in conn.execute("SELECT * FROM ingredients"):
        if _clave(fila["name"]) == clave:
            return fila
    return None


def listar_despensa(categoria: str | None = None, incluir_agotados: bool = False) -> list[dict]:
    with conexion() as conn:
        filas = conn.execute("SELECT * FROM ingredients ORDER BY name").fetchall()
    resultado = []
    for f in filas:
        cantidad = a_numero(f["quantity"]) or 0
        if not incluir_agotados and cantidad <= 0:
            continue
        if categoria and _normalizar(categoria) not in _normalizar(f["category"] or ""):
            continue
        resultado.append({"nombre": f["name"], "cantidad": cantidad, "unidad": f["unit"] or "",
                          "categoria": f["category"] or ""})
    return resultado


def buscar_ingrediente(nombre: str) -> list[dict]:
    """Ingredientes cuyo nombre contiene el buscado ('pollo' encuentra 'pechuga de pollo')."""
    clave = _clave(nombre)
    return [i for i in listar_despensa(incluir_agotados=True) if clave in _clave(i["nombre"])]


def anadir_ingrediente(nombre: str, cantidad: float, unidad: str = "unidades", categoria: str = "Otros") -> str:
    """Suma la cantidad si ya existe (convirtiendo g↔kg, ml↔l); si no, lo crea."""
    with conexion() as conn:
        fila = _fila(conn, nombre)
        if fila is None:
            conn.execute("INSERT INTO ingredients (name, quantity, category, unit) VALUES (?, ?, ?, ?)",
                         (nombre.lower().strip(), formatear_numero(cantidad), categoria, unidad))
            return f"Añadido a la despensa: {nombre} ({formatear_numero(cantidad)} {unidad})."
        actual = a_numero(fila["quantity"]) or 0
        unidad_actual = fila["unit"] or unidad
        convertida = cantidad if _normalizar(unidad) == _normalizar(unidad_actual) else _convertir(cantidad, unidad, unidad_actual)
        if convertida is None:
            # Unidades incompatibles: se sustituye para no mezclar magnitudes
            conn.execute("UPDATE ingredients SET quantity = ?, unit = ? WHERE id = ?",
                         (formatear_numero(cantidad), unidad, fila["id"]))
            return (f"{fila['name']} estaba en {unidad_actual}; ahora queda {formatear_numero(cantidad)} {unidad} "
                    f"(no se pueden sumar {unidad} y {unidad_actual}).")
        nueva = actual + convertida
        conn.execute("UPDATE ingredients SET quantity = ? WHERE id = ?", (formatear_numero(nueva), fila["id"]))
        return f"{fila['name']}: {formatear_numero(actual)} → {formatear_numero(nueva)} {unidad_actual}."


def fijar_ingrediente(nombre: str, cantidad: float, unidad: str | None = None) -> str:
    with conexion() as conn:
        fila = _fila(conn, nombre)
        if fila is None:
            return f"No hay ningún ingrediente llamado '{nombre}' en la despensa."
        unidad = unidad or fila["unit"] or ""
        conn.execute("UPDATE ingredients SET quantity = ?, unit = ? WHERE id = ?",
                     (formatear_numero(cantidad), unidad, fila["id"]))
        return f"{fila['name']} actualizado a {formatear_numero(cantidad)} {unidad}."


def consumir(elementos: list[tuple[str, float | None, str]]) -> list[str]:
    """Resta cantidades de la despensa. Devuelve una línea de resumen por ingrediente."""
    resumen = []
    with conexion() as conn:
        for nombre, cantidad, unidad in elementos:
            fila = _fila(conn, nombre)
            if fila is None:
                resumen.append(f"{nombre}: no estaba en la despensa.")
                continue
            actual = a_numero(fila["quantity"]) or 0
            unidad_actual = fila["unit"] or ""
            if cantidad is None:
                nueva, aviso = 0, " (sin cantidad: se marca como agotado)"
            else:
                restar = cantidad if not unidad or _normalizar(unidad) == _normalizar(unidad_actual) \
                    else _convertir(cantidad, unidad, unidad_actual)
                aviso = ""
                if restar is None:
                    restar, aviso = cantidad, f" (unidades distintas: {unidad} y {unidad_actual})"
                nueva = max(actual - restar, 0)
            conn.execute("UPDATE ingredients SET quantity = ? WHERE id = ?", (formatear_numero(nueva), fila["id"]))
            estado = "agotado" if nueva <= 0 else f"quedan {formatear_numero(nueva)} {unidad_actual}"
            resumen.append(f"{fila['name']}: {estado}{aviso}.")
    return resumen


def eliminar_ingrediente(nombre: str) -> str:
    with conexion() as conn:
        fila = _fila(conn, nombre)
        if fila is None:
            return f"No hay ningún ingrediente llamado '{nombre}' en la despensa."
        conn.execute("DELETE FROM ingredients WHERE id = ?", (fila["id"],))
        return f"{fila['name']} eliminado de la despensa."


# ------------------------------------------------------------------ preferencias
def obtener_preferencias() -> dict[str, list[str]]:
    with conexion() as conn:
        filas = conn.execute("SELECT tipo, valor FROM preferencias ORDER BY tipo, valor").fetchall()
    resultado = {t: [] for t in TIPOS_PREFERENCIA}
    for f in filas:
        resultado.setdefault(f["tipo"], []).append(f["valor"])
    return resultado


def guardar_preferencia(tipo: str, valor: str) -> str:
    tipo = _normalizar(tipo).replace(" ", "_")
    if tipo not in TIPOS_PREFERENCIA:
        return f"Tipo no válido: usa uno de {', '.join(TIPOS_PREFERENCIA)}."
    with conexion() as conn:
        conn.execute("INSERT OR IGNORE INTO preferencias (tipo, valor) VALUES (?, ?)", (tipo, valor.lower().strip()))
    return f"Guardado en {TIPOS_PREFERENCIA[tipo].lower()}: {valor}."


def eliminar_preferencia(tipo: str, valor: str) -> str:
    tipo = _normalizar(tipo).replace(" ", "_")
    with conexion() as conn:
        borradas = conn.execute("DELETE FROM preferencias WHERE tipo = ? AND valor = ?",
                                (tipo, valor.lower().strip())).rowcount
    return f"Eliminado de {TIPOS_PREFERENCIA.get(tipo, tipo).lower()}: {valor}." if borradas \
        else f"No había '{valor}' en {TIPOS_PREFERENCIA.get(tipo, tipo).lower()}."


def texto_restricciones(preferencias: dict[str, list[str]] | None = None) -> str:
    """Frase con las restricciones del usuario para los prompts ('' si no hay ninguna)."""
    p = preferencias if preferencias is not None else obtener_preferencias()
    partes = []
    if p.get("alergia"):
        partes.append(f"alergias o intolerancias: {', '.join(p['alergia'])} (no puede tomarlos de ninguna forma)")
    if p.get("dieta"):
        partes.append(f"dieta: {', '.join(p['dieta'])}")
    if p.get("no_me_gusta"):
        partes.append(f"no le gusta: {', '.join(p['no_me_gusta'])}")
    return "; ".join(partes)


# ------------------------------------------------------------------ lista de la compra
def ver_lista_compra() -> list[str]:
    with conexion() as conn:
        return [f["producto"] for f in conn.execute("SELECT producto FROM lista_compra ORDER BY id")]


def anadir_a_lista(productos: list[str]) -> list[str]:
    anadidos = []
    with conexion() as conn:
        existentes = {_clave(f["producto"]) for f in conn.execute("SELECT producto FROM lista_compra")}
        for producto in (p.strip() for p in productos):
            if producto and _clave(producto) not in existentes:
                conn.execute("INSERT INTO lista_compra (producto, fecha) VALUES (?, ?)",
                             (producto, datetime.now().isoformat(timespec="seconds")))
                existentes.add(_clave(producto))
                anadidos.append(producto)
    return anadidos


def quitar_de_lista(producto: str) -> str:
    with conexion() as conn:
        if _normalizar(producto) in ("todo", "toda", "todos", "*"):
            n = conn.execute("DELETE FROM lista_compra").rowcount
            return f"Lista de la compra vaciada ({n} productos)."
        clave = _clave(producto)
        filas = [f for f in conn.execute("SELECT * FROM lista_compra") if clave in _clave(f["producto"])]
        for f in filas:
            conn.execute("DELETE FROM lista_compra WHERE id = ?", (f["id"],))
    return f"Quitado de la lista: {', '.join(f['producto'] for f in filas)}." if filas \
        else f"'{producto}' no estaba en la lista de la compra."


# ------------------------------------------------------------------ historial
def registrar_plato(plato: str, comentario: str = "") -> str:
    with conexion() as conn:
        conn.execute("INSERT INTO historial (plato, fecha, comentario) VALUES (?, ?, ?)",
                     (plato.strip(), datetime.now().isoformat(timespec="seconds"), comentario.strip()))
    return f"Registrado en tu historial: {plato}."


def ver_historial(dias: int = 7) -> list[dict]:
    desde = (datetime.now() - timedelta(days=max(dias, 1))).isoformat(timespec="seconds")
    with conexion() as conn:
        filas = conn.execute("SELECT plato, fecha, comentario FROM historial WHERE fecha >= ? ORDER BY fecha DESC",
                             (desde,)).fetchall()
    return [dict(f) for f in filas]