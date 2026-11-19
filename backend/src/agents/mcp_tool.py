"""Servidor MCP de CookingAura: herramientas que usan el grafo y el agente del chat libre.

Se ejecuta como subproceso del backend (stdio). Los datos del usuario están en SQLite (src/data/sqlitebd.py);
las recetas guardadas se consultan a través de la API del propio backend, que ya tiene el modelo
de embeddings cargado.
"""
import json
import os
import sys
from pathlib import Path

# El subproceso arranca con la carpeta de este fichero en sys.path: se añade la raíz para importar 'src'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
from mcp.server.fastmcp import FastMCP
from pymilvus import MilvusClient

from src.data import sqlitebd, seleccion_ingredientes

milvus_host = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_URI = f"http://{milvus_host}:19530"
# El backend escucha en el mismo contenedor que este subproceso
BACKEND_INTERNO = os.environ.get("BACKEND_INTERNO_URL", "http://localhost:8080").rstrip("/")

mcp = FastMCP("CookingAura")


def _cantidad(valor: str) -> float | None:
    return sqlitebd.a_numero(valor)


# ====================================================================== DESPENSA
@mcp.tool()
def ver_despensa(nombre: str = "", categoria: str = "", incluir_agotados: bool = False) -> str:
    """Muestra los ingredientes de la despensa del usuario con su cantidad.
    nombre: busca un ingrediente concreto ('pollo'). categoria: filtra por categoría ('verduras').
    incluir_agotados: True para ver también los que están a 0."""
    try:
        if nombre:
            filas = sqlitebd.buscar_ingrediente(nombre)
            if not filas:
                return f"No hay '{nombre}' en la despensa."
        else:
            filas = sqlitebd.listar_despensa(categoria or None, incluir_agotados)
            if not filas:
                return "No se encontraron ingredientes en la despensa."
        lineas = [f"- {f['nombre'].capitalize()} ({sqlitebd.formatear_numero(f['cantidad'])} {f['unidad']})"
                  + (" [agotado]" if f["cantidad"] <= 0 else "") for f in filas]
        return "Ingredientes disponibles en la despensa:\n" + "\n".join(lineas) + "\n"
    except Exception as e:
        return f"Error al consultar la base de datos: {e}"


@mcp.tool()
def ingredientes_principales(tipo_comida: str = "", cantidad: int = 20) -> str:
    """Uso interno: hasta `cantidad` ingredientes principales de la despensa (sin condimentos), adecuados al
    tipo de comida (desayuno, comida, cena, merienda) y variados, elegidos al azar. Excluye alergias y
    lo que no le gusta al usuario."""
    try:
        preferencias = sqlitebd.obtener_preferencias()
        evitar = (preferencias.get("alergia") or []) + (preferencias.get("no_me_gusta") or [])
        elegidos = seleccion_ingredientes.seleccionar(sqlitebd.listar_despensa(), tipo_comida or None,
                                                      max(1, int(cantidad)), evitar)
        return seleccion_ingredientes.formatear(elegidos)
    except Exception as e:
        return f"Error al consultar la base de datos: {e}"


@mcp.tool()
def anadir_ingrediente(nombre: str, cantidad: str, unidad: str = "unidades", categoria: str = "Otros") -> str:
    """Añade un ingrediente a la despensa o suma la cantidad si ya existe (p. ej. tras hacer la compra).
    cantidad: número ('2', '0.5'). unidad: g, kg, ml, litros, unidades, paquete..."""
    n = _cantidad(cantidad)
    if n is None or n < 0:
        return f"Cantidad no válida: '{cantidad}'. Indica un número."
    return sqlitebd.anadir_ingrediente(nombre, n, unidad or "unidades", categoria or "Otros")


@mcp.tool()
def actualizar_ingrediente(nombre: str, cantidad: str, unidad: str = "") -> str:
    """Fija la cantidad exacta que queda de un ingrediente ('me quedan 3 huevos'). unidad es opcional."""
    n = _cantidad(cantidad)
    if n is None or n < 0:
        return f"Cantidad no válida: '{cantidad}'. Indica un número."
    return sqlitebd.fijar_ingrediente(nombre, n, unidad or None)


@mcp.tool()
def consumir_ingredientes(ingredientes: str) -> str:
    """Resta de la despensa lo que el usuario ha gastado o cocinado.
    ingredientes: lista separada por comas, p. ej. 'huevos 2, leche 0.5 litros, harina 200 g'.
    Un ingrediente sin cantidad se marca como agotado."""
    elementos = sqlitebd.parsear_lista(ingredientes)
    if not elementos:
        return "No he entendido qué ingredientes descontar."
    return "Despensa actualizada:\n" + "\n".join(f"- {l}" for l in sqlitebd.consumir(elementos))


@mcp.tool()
def eliminar_ingrediente(nombre: str) -> str:
    """Elimina por completo un ingrediente de la despensa."""
    return sqlitebd.eliminar_ingrediente(nombre)


# ====================================================================== RECETAS GUARDADAS
@mcp.tool()
def buscar_recetas_guardadas(consulta: str, cantidad: int = 3) -> str:
    """Busca en las recetas que el usuario ha subido (sus documentos) y devuelve los fragmentos más parecidos.
    consulta: lo que se busca ('lentejas', 'postre con chocolate', 'cena ligera con pescado')."""
    try:
        r = httpx.get(f"{BACKEND_INTERNO}/recetas/buscar",
                      params={"query": consulta, "modo": "hibrida", "top_k": max(1, min(int(cantidad), 5))},
                      timeout=120)
        r.raise_for_status()
        resultados = r.json().get("resultados", [])
    except Exception as e:
        return f"No se pudo buscar en las recetas guardadas: {e}"
    if not resultados:
        return f"No hay recetas guardadas que encajen con '{consulta}'."
    bloques = [f"{i}. [{r.get('fuente') or 'receta'}]\n{(r.get('texto') or '')[:900]}"
               for i, r in enumerate(resultados, start=1)]
    return "Recetas guardadas encontradas:\n\n" + "\n\n".join(bloques)


@mcp.tool()
def listar_documentos_recetas() -> str:
    """Lista los documentos de recetas que el usuario ha subido a la aplicación."""
    try:
        r = httpx.get(f"{BACKEND_INTERNO}/recetas/documentos", timeout=60)
        r.raise_for_status()
        documentos = r.json().get("documentos", [])
    except Exception as e:
        return f"No se pudieron consultar los documentos: {e}"
    if not documentos:
        return "El usuario no ha subido ningún documento de recetas."
    return "Documentos de recetas:\n" + "\n".join(f"- {d['nombre']} ({d['fragmentos']} fragmentos)" for d in documentos)


@mcp.tool()
def estado_recetas() -> str:
    """Uso interno: si la colección de recetas existe y cuántos fragmentos tiene (JSON)."""
    try:
        client = MilvusClient(uri=MILVUS_URI)
        if not client.has_collection("recetas"):
            return json.dumps({"existe": False, "cantidad": 0, "mensaje": "La colección 'recetas' no existe."})
        client.load_collection("recetas")
        resultados = client.query(collection_name="recetas", filter="", output_fields=["count(*)"])
        cantidad = resultados[0].get("count(*)", 0) if resultados else 0
        return json.dumps({"existe": True, "cantidad": cantidad,
                           "mensaje": f"La colección 'recetas' tiene {cantidad} documentos."})
    except Exception as e:
        return json.dumps({"existe": False, "cantidad": 0, "mensaje": f"Error con Milvus: {e}"})


# ====================================================================== PREFERENCIAS
@mcp.tool()
def ver_preferencias() -> str:
    """Muestra las alergias, la dieta y los gustos guardados del usuario. Consúltalo antes de recomendar comida."""
    p = sqlitebd.obtener_preferencias()
    lineas = [f"- {titulo}: {', '.join(p.get(tipo, [])) or 'ninguno'}" for tipo, titulo in sqlitebd.TIPOS_PREFERENCIA.items()]
    return "Preferencias del usuario:\n" + "\n".join(lineas)


@mcp.tool()
def guardar_preferencia(tipo: str, valor: str) -> str:
    """Guarda una preferencia. tipo: 'alergia' (alergias e intolerancias), 'dieta' (vegetariana, sin gluten...),
    'no_me_gusta' o 'me_gusta'. valor: el ingrediente o la dieta ('marisco', 'vegetariana')."""
    return sqlitebd.guardar_preferencia(tipo, valor)


@mcp.tool()
def eliminar_preferencia(tipo: str, valor: str) -> str:
    """Elimina una preferencia guardada. tipo: 'alergia', 'dieta', 'no_me_gusta' o 'me_gusta'."""
    return sqlitebd.eliminar_preferencia(tipo, valor)


# ====================================================================== LISTA DE LA COMPRA
@mcp.tool()
def ver_lista_compra() -> str:
    """Muestra la lista de la compra del usuario."""
    productos = sqlitebd.ver_lista_compra()
    return "Lista de la compra:\n" + "\n".join(f"- {p}" for p in productos) if productos \
        else "La lista de la compra está vacía."


@mcp.tool()
def anadir_a_lista_compra(productos: str) -> str:
    """Añade productos a la lista de la compra. productos: separados por comas ('tomates, 1 kg de arroz, leche')."""
    anadidos = sqlitebd.anadir_a_lista([p for p in productos.replace(";", ",").split(",")])
    return f"Añadido a la lista: {', '.join(anadidos)}." if anadidos else "Esos productos ya estaban en la lista."


@mcp.tool()
def quitar_de_lista_compra(producto: str) -> str:
    """Quita un producto de la lista de la compra (p. ej. porque ya se ha comprado). 'todo' vacía la lista."""
    return sqlitebd.quitar_de_lista(producto)


# ====================================================================== HISTORIAL
@mcp.tool()
def registrar_plato_cocinado(plato: str, comentario: str = "") -> str:
    """Apunta en el historial un plato que el usuario ha cocinado o comido. comentario: opcional ('le faltó sal')."""
    return sqlitebd.registrar_plato(plato, comentario)


@mcp.tool()
def ver_historial(dias: int = 7) -> str:
    """Muestra los platos que el usuario ha cocinado en los últimos días."""
    filas = sqlitebd.ver_historial(int(dias))
    if not filas:
        return f"No hay platos registrados en los últimos {dias} días."
    return "Historial:\n" + "\n".join(
        f"- {f['fecha'][:10]}: {f['plato']}" + (f" ({f['comentario']})" if f["comentario"] else "") for f in filas)


# Mantiene el proceso vivo escuchando peticiones por stdio
if __name__ == "__main__":
    mcp.run()