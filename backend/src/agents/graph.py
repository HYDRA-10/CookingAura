import ast
import json
import os
import re
import sys
import time
import functools
import inspect
from contextlib import AsyncExitStack
from typing import TypedDict

import httpx
from langgraph.graph import StateGraph, END

# Importaciones módulos app
from .rag_node import (buscar_recetas, buscar_recetas_en_milvus, buscar_con_criterios,
                       limpiar_ingredientes, obtener_receta_por_id,
                       get_client, COLLECTION_NAME)
from .web_search_node import buscar_receta_internet
from .ideas_node import generar_ideas_creativas, titulo_y_resumen
from .coincidencias import separar_ingredientes, contar_coincidencias
from .mcp_client import herramientas_mcp, gestor_mcp, texto_resultado_mcp as _texto_resultado_mcp
from .agente_chat_generico import agente_eventos, ejecutar_agente, hay_tema
from .intencion import intencion_por_reglas, quitar_saludo, habla_de_cocina, plato_pedido
from src.data import sqlitebd, seleccion_ingredientes
from src.rag import metadatos
from src.llms.ollama_client import evaluar_intencion, evaluar_ingredientes, generar_respuesta_final_ollama, stream_respuesta_ollama
from src.llms.vllm_client import generar_respuesta_final_vllm, stream_respuesta_vllm, VLLM_URL
from src.llms.prompts import normalizar_tipo_comida, detectar_tipo_comida, construir_prompt_generador, tipo_comida_con_articulo
from src.config import (IDEAS_TOTAL, IDEAS_MAX_LOCALES, EVAL_MIN_COINCIDENCIAS, GENERADOR_MAX_CHARS,
                        DESPENSA_MAX_INGREDIENTES, WEB_RECETA_DIRECTA, GEMINI_TIMEOUT)


def cronometrar(fn):
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def wrapper(state):
            t = time.perf_counter(); r = await fn(state)
            print(f"⏱ {fn.__name__}: {time.perf_counter()-t:.2f}s"); return r
    else:
        @functools.wraps(fn)
        def wrapper(state):
            t = time.perf_counter(); r = fn(state)
            print(f"⏱ {fn.__name__}: {time.perf_counter()-t:.2f}s"); return r
    return wrapper


class AgentState(TypedDict):
    input_usuario: str
    intencion: str
    tipo_comida: str             # "desayuno", "comida", "cena"... o ausente
    fase: str                    # "ideas" (por defecto) | "receta"
    # --- Ingredientes ---
    ingredientes_mcp: str
    ingredientes_manual: str
    improvisar: bool             # no hay ingredientes: se proponen platos con básicos
    hay_documentos: bool
    # --- Fase 1: ideas ---
    fuente_busqueda: str         # "auto" (propias + creativas) | "local" (solo propias) | "web" (solo creativas)
    ids_excluidos: list[int]     # recetas propias ya propuestas como idea
    recetas_descartadas: list[str]  # nombres de platos ya propuestos
    ideas: list[dict]            # [{"id", "nombre", "descripcion", "origen", "ref_id", "fuente"}]
    # --- Fase 2: receta ---
    idea_elegida: dict
    recetas_encontradas: str
    ids_recetas_usadas: list[int]
    origen_receta: str           # "local" | "web" | "idea"
    receta_generada: bool
    respuesta_final: str
    # --- Usuario y chat libre ---
    preferencias: dict           # {"alergia": [...], "dieta": [...], "no_me_gusta": [...], "me_gusta": [...]}
    historial: list[dict]        # mensajes anteriores del chat [{"role", "content"}]
    agente_pendiente: bool       # en streaming, el agente se ejecuta fuera del grafo
    despensa_modificada: bool
    herramientas_usadas: list[str]
    # --- Streaming ---
    modo_streaming: bool
    prompt_generador: str
    motor_generador: str         # "vllm" | "ollama"


def vllm_disponible() -> bool:
    """vLLM corre en su propio contenedor: se comprueba si responde, no la GPU del backend."""
    try:
        return httpx.get(f"{VLLM_URL}/models", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


# 1. DEFINICIÓN DE LOS NODOS
def nodo_respuesta_directa(state: AgentState):
    print("-> Ejecutando Nodo: Respuesta Directa")
    if state["intencion"] == "SALUDO":
        return {"respuesta_final": "¡Hola! Soy CookingAura 👨‍🍳. ¿En qué puedo ayudarte?"}
    else:
        return {"respuesta_final": "Eso se me escapa: soy un asistente de cocina 👨‍🍳. Puedo proponerte qué comer, "
                                   "darte recetas o ayudarte con tu despensa, tu lista de la compra o tus alergias. "
                                   "¿Te propongo algo para hoy?"}


def _menciona_despensa(mensaje: str) -> bool:
    """True si el mensaje nombra algún ingrediente de la despensa (amplía el vocabulario de cocina)."""
    try:
        claves = {sqlitebd._clave(i["nombre"]) for i in sqlitebd.listar_despensa(incluir_agotados=True)}
        texto = sqlitebd._clave(mensaje)
        return any(c and c in texto for c in claves)
    except Exception:
        return False


async def nodo_intencion(state: AgentState):
    mensaje = state["input_usuario"]

    # 0) Receta de un plato concreto ("prepárame una receta de ternera teriyaki"): directamente a la receta
    plato = plato_pedido(mensaje)
    if plato:
        print(f"   Intención por reglas: PLATO «{plato}»")
        return {"intencion": "PLATO", "fase": "receta",
                "idea_elegida": {"id": "pedida", "nombre": plato, "descripcion": "", "origen": "pedida",
                                 "ref_id": None, "fuente": ""}}

    # 1) Reglas rápidas: saludos solos y peticiones de ideas ("Hola, ¿qué puedo comer hoy?")
    regla = intencion_por_reglas(mensaje)
    if regla:
        print(f"   Intención por reglas: {regla}")
        return {"intencion": regla}

    # 2) Sin ninguna señal de cocina ni conversación previa: fuera de tema sin gastar una llamada al LLM
    #    ("¿Qué tiempo va a hacer mañana?"). Con conversación previa sí se pregunta al LLM, porque un
    #    "¿y para 6 personas?" no lleva palabras de cocina pero continúa lo anterior.
    if (not state.get("historial") and not habla_de_cocina(mensaje) and not hay_tema(mensaje)
            and not _menciona_despensa(mensaje)):
        print("   Intención por reglas: OTRO (sin relación con la cocina)")
        return {"intencion": "OTRO"}

    # 3) Clasificador LLM, sin el saludo inicial (que confunde a los modelos pequeños)
    print("-> Ejecutando Categorización de Intención del usuario en Ollama")
    ultimo_asistente = next((m.get("content", "") for m in reversed(state.get("historial") or [])
                             if m.get("role") == "assistant"), "")
    intencion = await evaluar_intencion(quitar_saludo(mensaje) or mensaje, ultimo_asistente)

    # 4) Red de seguridad: un mensaje que habla de cocina nunca se rechaza como "fuera de tema"
    if intencion == "OTRO" and (habla_de_cocina(mensaje) or _menciona_despensa(mensaje)):
        print("   El clasificador dijo OTRO, pero el mensaje habla de cocina: se pasa al asistente")
        intencion = "ASISTENTE"
    print(f"   Intención: {intencion}")
    return {"intencion": intencion}


def _preferencias(state: AgentState) -> dict:
    """Preferencias del estado o, si aún no se han leído, de la base de datos."""
    if state.get("preferencias") is not None:
        return state["preferencias"]
    try:
        return sqlitebd.obtener_preferencias()
    except Exception as e:
        print(f"⚠️  No se pudieron leer las preferencias: {e}", flush=True)
        return {}


def _despensa_directa(tipo_comida: str | None = None) -> str:
    """Misma salida que la herramienta ingredientes_principales, leyendo SQLite directamente (si MCP no responde)."""
    preferencias = sqlitebd.obtener_preferencias()
    evitar = (preferencias.get("alergia") or []) + (preferencias.get("no_me_gusta") or [])
    elegidos = seleccion_ingredientes.seleccionar(sqlitebd.listar_despensa(), tipo_comida,
                                                  DESPENSA_MAX_INGREDIENTES, evitar)
    return seleccion_ingredientes.formatear(elegidos)


def _estado_milvus_directo() -> dict:
    """Misma salida que la herramienta estado_recetas, consultando Milvus directamente."""
    try:
        client = get_client()
        if not client.has_collection(COLLECTION_NAME):
            return {"cantidad": 0, "mensaje": "La colección 'recetas' no existe."}
        client.load_collection(COLLECTION_NAME)
        filas = client.query(collection_name=COLLECTION_NAME, filter="", output_fields=["count(*)"])
        cantidad = filas[0].get("count(*)", 0) if filas else 0
        return {"cantidad": cantidad, "mensaje": f"La colección 'recetas' tiene {cantidad} documentos."}
    except Exception as e:
        return {"cantidad": 0, "mensaje": f"Error con Milvus: {e}"}


async def nodo_mcp(state: AgentState):
    print("-> Ejecutando Nodo MCP: despensa + estado de Milvus")
    # Si ya vienen ingredientes (seleccionados en el frontend o manuales validados), no se vuelven a pedir
    ingredientes = "" if state.get("intencion") == "RECETA" else (state.get("ingredientes_mcp") or "")

    try:
        # Sesión MCP compartida (abierta al arrancar el backend); temporal solo si no está activa
        async with herramientas_mcp() as mcp_tools:
            herramientas = {t.name: t for t in mcp_tools}
            if not ingredientes:
                # Sin ingredientes indicados: los principales de la despensa, adecuados al tipo de comida
                bruto = await herramientas["ingredientes_principales"].ainvoke(
                    {"tipo_comida": state.get("tipo_comida") or "", "cantidad": DESPENSA_MAX_INGREDIENTES})
                ingredientes = _texto_resultado_mcp(bruto)
                print(f"   Ingredientes elegidos ({state.get('tipo_comida') or 'cualquier comida'}): {ingredientes}")

            bruto_milvus = await herramientas["estado_recetas"].ainvoke({})
            try:
                estado_milvus = json.loads(_texto_resultado_mcp(bruto_milvus))
            except json.JSONDecodeError:
                estado_milvus = {"cantidad": 0, "mensaje": "Respuesta de Milvus no válida"}
    except Exception as e:
        # El flujo de ideas no depende de MCP: se leen los mismos datos directamente
        print(f"⚠️  {e}\n   Se continúa leyendo la despensa y Milvus sin MCP.", flush=True)
        if not ingredientes:
            ingredientes = _despensa_directa(state.get("tipo_comida"))
            print(f"   Ingredientes elegidos (sin MCP): {ingredientes}")
        estado_milvus = _estado_milvus_directo()

    hay_documentos = estado_milvus.get("cantidad", 0) > 0
    print(f"   Milvus: {estado_milvus.get('mensaje')}")

    preferencias = _preferencias(state)
    restricciones = sqlitebd.texto_restricciones(preferencias)
    if restricciones:
        print(f"   Restricciones del usuario: {restricciones}")

    # Un fallo de la herramienta llega como texto: no debe tratarse como lista de ingredientes
    error_herramienta = ingredientes.startswith(("Error executing tool", "Error al consultar"))
    if error_herramienta:
        print(f"⚠️  La herramienta de despensa ha fallado: {ingredientes}", flush=True)

    sin_ingredientes = (error_herramienta or not ingredientes or ingredientes == "None"
                        or ingredientes.startswith("No se encontraron"))
    if sin_ingredientes:
        print("   Sin ingredientes: se improvisarán las ideas")
        return {"ingredientes_mcp": "", "improvisar": True, "hay_documentos": hay_documentos,
                "preferencias": preferencias}

    return {"ingredientes_mcp": ingredientes, "hay_documentos": hay_documentos, "preferencias": preferencias}


async def nodo_evaluar_ingredientes(state: AgentState):
    print("-> Ejecutando Categorización de Ingredientes")
    respuesta_llm = await evaluar_ingredientes(state["ingredientes_manual"])

    try:
        # Greedy hasta el ÚLTIMO ']' para capturar también la lista interna: ['SI', ['A', 'B']]
        match = re.search(r"\[\s*['\"](SI|NO)['\"].*\]", respuesta_llm, re.IGNORECASE | re.DOTALL)
        if match:
            lista_eval = ast.literal_eval(match.group(0))
            evaluacion = lista_eval[0].upper()
            ingredientes_validos = ", ".join(lista_eval[1]) if len(lista_eval) > 1 else ""
        else:
            raise ValueError("No se encontró el patrón esperado de lista.")

    except Exception as e:
        print(f"Error al analizar la respuesta de Ollama: {e}")
        evaluacion = "NO"
        ingredientes_validos = ""

    if evaluacion == "SI" and ingredientes_validos:
        return {"ingredientes_mcp": ingredientes_validos}
    else:
        return {"respuesta_final": "Los ingredientes proporcionados no son válidos. Revísalos y vuelve a intentarlo."}


def nodo_ideas_locales(state: AgentState):
    """Ideas sacadas de las recetas propias (Milvus), filtradas por ingredientes en común."""
    print("-> Ejecutando Nodo Ideas Locales: buscando en tus recetas")
    fuente = state.get("fuente_busqueda") or "auto"
    max_ideas = IDEAS_TOTAL if fuente == "local" else IDEAS_MAX_LOCALES
    if max_ideas <= 0:
        return {"ideas": []}

    ingredientes = limpiar_ingredientes(state.get("ingredientes_mcp") or "")
    candidatos = max_ideas * 3
    preferencias = _preferencias(state)
    evitar = (preferencias.get("alergia") or []) + (preferencias.get("no_me_gusta") or [])
    # Filtros por metadatos: tipo de comida, alergias (alérgenos oficiales) y dieta
    dieta = next((d for d in metadatos.dietas(" ".join(preferencias.get("dieta") or []))
                  if d in metadatos.DIETAS_COMPATIBLES), None)
    filtros = {"tipo_comida": state.get("tipo_comida"), "alergias": preferencias.get("alergia") or [],
               "dieta": dieta}
    if state.get("improvisar") or not ingredientes:
        # Sin ingredientes: recetas propias que encajen con el momento del día
        momento = tipo_comida_con_articulo(state.get("tipo_comida"))
        consulta = f"Receta para {momento}" if momento else "Receta sencilla"
        resultados = buscar_con_criterios(consulta, state.get("tipo_comida") or "receta", top_k=candidatos,
                                          excluir_ids=state.get("ids_excluidos"), **filtros)
        minimo = 0
    else:
        resultados = buscar_recetas_en_milvus(ingredientes, state.get("tipo_comida"), top_k=candidatos,
                                              excluir_ids=state.get("ids_excluidos"),
                                              alergias=filtros["alergias"], dieta=dieta)
        minimo = max(1, min(EVAL_MIN_COINCIDENCIAS, len(separar_ingredientes(ingredientes))))
    ideas, vistos = [], {n.strip().lower() for n in (state.get("recetas_descartadas") or [])}
    for r in resultados:
        if r.get("coincidencias", 0) < minimo:
            continue
        if evitar and contar_coincidencias(evitar, r["texto"]):
            print(f"   Descartada por preferencias: {titulo_y_resumen(r['texto'])[0]}")
            continue
        nombre, descripcion = titulo_y_resumen(r["texto"])
        if nombre.lower() in vistos:
            continue
        vistos.add(nombre.lower())
        ideas.append({"id": f"local-{r['id']}", "nombre": nombre, "descripcion": descripcion,
                      "origen": "local", "ref_id": r["id"], "fuente": r.get("fuente") or ""})
        if len(ideas) >= max_ideas:
            break

    print(f"   {len(ideas)} ideas de tus recetas (mínimo {minimo} ingredientes en común)")
    return {"ideas": ideas}


def nodo_ideas_creativas(state: AgentState):
    """Completa la lista con platos inventados por el LLM (o la llena entera en modo 'web')."""
    print("-> Ejecutando Nodo Ideas Creativas")
    actuales = state.get("ideas") or []
    n = IDEAS_TOTAL - len(actuales)
    if n <= 0:
        return {}

    descartadas = (state.get("recetas_descartadas") or []) + [i["nombre"] for i in actuales]
    nuevas = generar_ideas_creativas(
        limpiar_ingredientes(state.get("ingredientes_mcp") or ""),
        state.get("tipo_comida"), n, descartadas, bool(state.get("improvisar")),
        restricciones=sqlitebd.texto_restricciones(_preferencias(state)),
    )
    ideas = actuales + [
        {"id": f"creativa-{k}", "nombre": i["nombre"], "descripcion": i["descripcion"],
         "origen": "creativa", "ref_id": None, "fuente": ""}
        for k, i in enumerate(nuevas, start=1)
    ]
    return {"ideas": ideas}


def nodo_presentar_ideas(state: AgentState):
    """Texto que acompaña a la lista (los botones para elegir los pinta el frontend)."""
    ideas = state.get("ideas") or []
    if not ideas:
        return {"respuesta_final": "😕 No se me ha ocurrido ningún plato esta vez. "
                                   "Prueba con otros ingredientes o vuelve a intentarlo."}

    momento = tipo_comida_con_articulo(state.get("tipo_comida"))
    para = f" para {momento}" if momento else ""
    lista_ingredientes = separar_ingredientes(limpiar_ingredientes(state.get("ingredientes_mcp") or ""))
    if state.get("improvisar") or not lista_ingredientes:
        intro = f"🥄 No tengo ingredientes con los que trabajar, así que he improvisado estas ideas{para}"
    elif len(lista_ingredientes) > 6:
        intro = f"Con lo que tienes en la despensa se me ocurren estas ideas{para}"
    else:
        intro = f"Con {', '.join(lista_ingredientes).lower()} se me ocurren estas ideas{para}"

    lineas = [f"{intro}:", ""]
    for idea in ideas:
        icono = "📚" if idea["origen"] == "local" else "✨"
        descripcion = f" — {idea['descripcion']}" if idea.get("descripcion") else ""
        lineas.append(f"- {icono} **{idea['nombre']}**{descripcion}")
    lineas += ["", "Elige una y te preparo la receta 👇 (📚 de tus recetas · ✨ idea nueva)"]
    return {"respuesta_final": "\n".join(lineas)}


def nodo_sin_recetas_locales(state: AgentState):
    """El usuario pidió ideas solo de sus recetas y no queda ninguna que encaje."""
    print("-> Ejecutando Nodo: Sin recetas locales")
    if not state.get("hay_documentos"):
        msj = ("📚 Todavía no tienes recetas guardadas. Súbelas desde la pestaña **Recetas** "
               "o pide ideas más creativas ✨.")
    elif state.get("ids_excluidos"):
        msj = ("📚 Ya te he propuesto todas las recetas de tus documentos que encajan con estos "
               "ingredientes. ¿Probamos con ideas más creativas? ✨")
    else:
        msj = ("📚 No he encontrado ninguna receta en tus documentos que encaje con estos ingredientes. "
               "¿Probamos con ideas más creativas? ✨")
    return {"respuesta_final": msj}


def nodo_preparar_receta(state: AgentState):
    """Fase 2: si la idea elegida es una receta propia, se recupera de Milvus."""
    idea = state.get("idea_elegida") or {}
    print(f"-> Ejecutando Nodo Preparar Receta: {idea.get('nombre')} ({idea.get('origen')})")
    if idea.get("origen") == "local" and idea.get("ref_id") is not None:
        try:
            fila = obtener_receta_por_id(idea["ref_id"])
        except Exception as e:
            print(f"⚠️  No se pudo recuperar la receta {idea['ref_id']}: {e}", flush=True)
            fila = None
        if fila and fila.get("texto"):
            return {
                "recetas_encontradas": f"[{fila.get('fuente') or 'receta'}]\n{fila['texto']}",
                "ids_recetas_usadas": [idea["ref_id"]],
                "origen_receta": "local",
            }
    if idea.get("origen") == "pedida":
        # Plato pedido por nombre: si está entre tus recetas (todas sus palabras en el título), se usa esa
        try:
            palabras = separar_ingredientes(idea["nombre"].replace(" con ", ", ").replace(" y ", ", "))
            for r in buscar_recetas(idea["nombre"], top_k=3):
                titulo, _ = titulo_y_resumen(r["texto"])
                if palabras and contar_coincidencias(palabras, titulo) == len(palabras):
                    print(f"   Encontrada en tus recetas: «{titulo}»")
                    return {"recetas_encontradas": f"[{r.get('fuente') or 'receta'}]\n{r['texto']}",
                            "ids_recetas_usadas": [r["id"]], "origen_receta": "local"}
        except Exception as e:
            print(f"⚠️  No se pudo buscar el plato en tus recetas: {e}", flush=True)
    return {}


def nodo_busqueda_web(state: AgentState):
    """Busca en internet la receta del plato elegido.
    - Con WEB_RECETA_DIRECTA, la receta de Gemini llega ya redactada y se muestra tal cual (sin pasar por el
      generador local, que tardaría ~100 s más en CPU).
    - Si Gemini falla o tarda más de GEMINI_TIMEOUT segundos, el generador redacta la receta desde la idea."""
    idea = state.get("idea_elegida") or {}
    print(f"-> Ejecutando Nodo Búsqueda Web: receta de «{idea.get('nombre')}» (límite {GEMINI_TIMEOUT} s)")
    try:
        receta_red = buscar_receta_internet(
            limpiar_ingredientes(state.get("ingredientes_mcp") or ""),
            state.get("tipo_comida"),
            plato=idea.get("nombre"),
            restricciones=sqlitebd.texto_restricciones(_preferencias(state)),
            formato_final=WEB_RECETA_DIRECTA,
        )
        if WEB_RECETA_DIRECTA and receta_red and not receta_red.startswith("No se pudo"):
            print("   Receta de Gemini mostrada directamente (sin reescribirla con el modelo local)")
            # receta_generada marca la respuesta como receta para el frontend (✅ en la idea, origen "web")
            return {"recetas_encontradas": receta_red, "origen_receta": "web", "respuesta_final": receta_red,
                    "receta_generada": True}
        return {"recetas_encontradas": receta_red, "origen_receta": "web"}
    except Exception as e:
        print(f"⚠️  Búsqueda web no disponible ({type(e).__name__}: {e}); "
              "la receta se redacta a partir de la idea.", flush=True)
        referencia = f"Plato: {idea.get('nombre', '')}. {idea.get('descripcion', '')}".strip()
        return {"recetas_encontradas": referencia, "origen_receta": "idea"}


def nodo_generador(state: AgentState):
    args = (
        limpiar_ingredientes(state.get("ingredientes_mcp") or ""),
        (state.get("recetas_encontradas") or "")[:GENERADOR_MAX_CHARS],
        state.get("tipo_comida"),
        state.get("input_usuario"),
        state.get("recetas_descartadas"),
        sqlitebd.texto_restricciones(_preferencias(state)),
    )
    motor = "vllm" if vllm_disponible() else "ollama"

    if state.get("modo_streaming"):
        # El texto lo emite /chat/stream token a token; aquí solo se prepara el prompt
        print(f"-> Ejecutando Nodo Generador: prompt preparado para streaming con {motor}")
        return {"prompt_generador": construir_prompt_generador(*args),
                "motor_generador": motor, "receta_generada": True}

    if motor == "vllm":
        print("-> Ejecutando Nodo Generador: vLLM")
        respuesta = generar_respuesta_final_vllm(*args)
    else:
        print("-> Ejecutando Nodo Generador: vLLM no disponible, usando Ollama")
        respuesta = generar_respuesta_final_ollama(*args)

    return {"respuesta_final": respuesta, "receta_generada": True}


async def nodo_agente_chat(state: AgentState):
    """Chat libre: el agente conversa y usa las herramientas MCP (despensa, compra, preferencias...)."""
    if state.get("modo_streaming"):
        # En streaming lo ejecuta el endpoint, para ir mostrando las herramientas que usa
        return {"agente_pendiente": True}
    print("-> Ejecutando Nodo Agente Chat")
    final = await ejecutar_agente(state["input_usuario"], state.get("historial"), state.get("intencion"))
    return {"respuesta_final": final.get("texto", ""),
            "despensa_modificada": final.get("despensa_modificada", False),
            "herramientas_usadas": final.get("herramientas", [])}


# 2. DEFINICIÓN DEL ENRUTAMIENTO CONDICIONAL
def decidir_ruta_nodo_inicial(state: AgentState) -> str:
    """Fase 2 va directa a preparar la receta; la fase 1 sigue el camino de siempre."""
    if state.get("fase") == "receta":
        return "preparar_receta"
    if state.get("intencion") == "RECETA" or state.get("ingredientes_mcp"):
        return "mcp"
    if state.get("ingredientes_manual"):
        return "evaluar_ingredientes"
    return "intencion"


def decidir_ruta_intencion(state: AgentState) -> str:
    """Dirige el grafo en base a la intención detectada."""
    if state["intencion"] == "RECETA":
        return "mcp"
    if state["intencion"] == "PLATO":
        return "preparar_receta"
    if state["intencion"] == "SALUDO":
        return "respuesta_directa"
    if (state["intencion"] == "OTRO" and not hay_tema(state["input_usuario"])
            and not state.get("historial")):
        # Ni palabras de cocina, ni temas con herramientas, ni conversación previa: respuesta directa, sin LLM
        return "respuesta_directa"
    # ASISTENTE, u OTRO que toca algún tema o continúa la conversación ("¿y para 6 personas?"):
    # lo atiende el agente (sin herramientas si no hay tema).
    # Así un error del clasificador nunca deja una petición sin hacer.
    return "agente"


def decidir_continuar_mcp(state: AgentState):
    if state.get("respuesta_final"):
        return END
    fuente = state.get("fuente_busqueda") or "auto"
    if fuente == "web":
        return "ideas_creativas"
    if fuente == "local":
        return "ideas_locales" if state.get("hay_documentos") else "sin_recetas_locales"
    return "ideas_locales" if state.get("hay_documentos") else "ideas_creativas"


def decidir_continuar_evaluacion(state: AgentState):
    """Si la evaluación de ingredientes falló, termina; si no, pasa por MCP para comprobar Milvus."""
    if state.get("respuesta_final"):
        return END
    return "mcp"


def decidir_tras_ideas_locales(state: AgentState):
    ideas = state.get("ideas") or []
    if (state.get("fuente_busqueda") or "auto") == "local":
        return "presentar_ideas" if ideas else "sin_recetas_locales"
    return "presentar_ideas" if len(ideas) >= IDEAS_TOTAL else "ideas_creativas"


def decidir_tras_preparar(state: AgentState):
    return "generador" if state.get("recetas_encontradas") else "busqueda_web"

def decidir_tras_web(state: AgentState):
    return END if state.get("respuesta_final") else "generador"


# 3. CONSTRUCCIÓN Y COMPILACIÓN DEL GRAFO
workflow = StateGraph(AgentState)

workflow.add_node("intencion", cronometrar(nodo_intencion))
workflow.add_node("respuesta_directa", cronometrar(nodo_respuesta_directa))
workflow.add_node("mcp", cronometrar(nodo_mcp))
workflow.add_node("evaluar_ingredientes", cronometrar(nodo_evaluar_ingredientes))
workflow.add_node("ideas_locales", cronometrar(nodo_ideas_locales))
workflow.add_node("ideas_creativas", cronometrar(nodo_ideas_creativas))
workflow.add_node("presentar_ideas", cronometrar(nodo_presentar_ideas))
workflow.add_node("sin_recetas_locales", cronometrar(nodo_sin_recetas_locales))
workflow.add_node("preparar_receta", cronometrar(nodo_preparar_receta))
workflow.add_node("busqueda_web", cronometrar(nodo_busqueda_web))
workflow.add_node("generador", cronometrar(nodo_generador))
workflow.add_node("agente", cronometrar(nodo_agente_chat))

# El mapa de destinos permite además dibujar el grafo correctamente (dibujar_grafo.py)
workflow.set_conditional_entry_point(
    decidir_ruta_nodo_inicial,
    {
        "preparar_receta": "preparar_receta",
        "mcp": "mcp",
        "evaluar_ingredientes": "evaluar_ingredientes",
        "intencion": "intencion",
    }
)

workflow.add_conditional_edges("intencion", decidir_ruta_intencion,
                               {"mcp": "mcp", "agente": "agente", "respuesta_directa": "respuesta_directa",
                                "preparar_receta": "preparar_receta"})
workflow.add_edge("agente", END)
workflow.add_edge("respuesta_directa", END)

workflow.add_conditional_edges("evaluar_ingredientes", decidir_continuar_evaluacion,
                               {END: END, "mcp": "mcp"})

# --- Fase 1: ideas ---
workflow.add_conditional_edges("mcp", decidir_continuar_mcp, {
    END: END,
    "ideas_locales": "ideas_locales",
    "ideas_creativas": "ideas_creativas",
    "sin_recetas_locales": "sin_recetas_locales",
})
workflow.add_conditional_edges("ideas_locales", decidir_tras_ideas_locales, {
    "presentar_ideas": "presentar_ideas",
    "ideas_creativas": "ideas_creativas",
    "sin_recetas_locales": "sin_recetas_locales",
})
workflow.add_edge("ideas_creativas", "presentar_ideas")
workflow.add_edge("presentar_ideas", END)
workflow.add_edge("sin_recetas_locales", END)

# --- Fase 2: receta del plato elegido ---
workflow.add_conditional_edges("preparar_receta", decidir_tras_preparar,
                               {"generador": "generador", "busqueda_web": "busqueda_web"})
workflow.add_conditional_edges(
    "busqueda_web",
    decidir_tras_web,
    {END: END, "generador": "generador"},
)
workflow.add_edge("generador", END)

app = workflow.compile()


# 4. FUNCIONES DE EJECUCIÓN (ENTRYPOINTS)
def _estado_inicial(pregunta: str, intencion: str = None, ingredientes_db: str = None,
                    ingredientes_manual: str = None, tipo_comida: str = None,
                    fuente_busqueda: str = None, ids_excluidos: list[int] = None,
                    recetas_descartadas: list[str] = None, fase: str = None,
                    idea: dict = None, historial: list[dict] = None) -> dict:
    estado = {"input_usuario": pregunta, "fase": "receta" if fase == "receta" and idea else "ideas"}
    if intencion:
        estado["intencion"] = intencion
    if ingredientes_db:
        estado["ingredientes_mcp"] = ingredientes_db
    if ingredientes_manual:
        estado["ingredientes_manual"] = ingredientes_manual
    if fuente_busqueda in ("auto", "local", "web"):
        estado["fuente_busqueda"] = fuente_busqueda
    if ids_excluidos:
        estado["ids_excluidos"] = list(ids_excluidos)
    if recetas_descartadas:
        estado["recetas_descartadas"] = list(recetas_descartadas)
    if idea:
        estado["idea_elegida"] = dict(idea)
    if historial:
        estado["historial"] = [dict(m) for m in historial]

    # Tipo de comida: el explícito del frontend manda; si no hay, se intenta deducir del texto
    tipo = normalizar_tipo_comida(tipo_comida) or detectar_tipo_comida(pregunta)
    if tipo:
        estado["tipo_comida"] = tipo
        print(f"   Tipo de comida: {tipo}")
    return estado


def _contexto(resultado: dict) -> dict:
    """Lo que necesita el frontend para mostrar las ideas y pedir la receta o más ideas."""
    es_receta = bool(resultado.get("receta_generada"))
    return {
        "fase": resultado.get("fase"),
        "es_receta": es_receta,
        "origen": resultado.get("origen_receta") if es_receta else None,
        "ideas": [] if es_receta else (resultado.get("ideas") or []),
        "improvisado": bool(resultado.get("improvisar")),
        # Ingredientes ya resueltos (despensa, selección o manuales validados): se reutilizan tal cual
        "ingredientes": resultado.get("ingredientes_mcp") or None,
        "tipo_comida": resultado.get("tipo_comida"),
        # Chat libre
        "agente": bool(resultado.get("agente_pendiente") or resultado.get("herramientas_usadas") is not None),
        "despensa_modificada": bool(resultado.get("despensa_modificada")),
        "herramientas": resultado.get("herramientas_usadas") or [],
    }


async def ejecutar_asistente(**kwargs) -> dict:
    """Ejecuta el grafo completo y devuelve {"respuesta", "contexto"} de una vez."""
    resultado = await app.ainvoke(_estado_inicial(**kwargs))
    return {"respuesta": resultado.get("respuesta_final", ""), "contexto": _contexto(resultado)}


# Mensaje que ve el usuario cuando termina cada nodo
PASOS = {
    "intencion": "🧭 Petición entendida",
    "mcp": "🥫 Despensa revisada",
    "evaluar_ingredientes": "🔍 Ingredientes comprobados",
    "ideas_locales": "📚 Tus recetas revisadas",
    "ideas_creativas": "✨ Ideas nuevas pensadas",
    "presentar_ideas": "🍽️ Ideas listas",
    "preparar_receta": "📖 Plato elegido",
    "busqueda_web": "🌐 Receta buscada en internet",
    "generador": "👨‍🍳 Redactando la receta",
}


async def ejecutar_asistente_stream(**kwargs):
    """Ejecuta el grafo emitiendo eventos:
    {"tipo": "estado", "texto", "nodo", "segundos"} al terminar cada nodo, {"tipo": "token", "texto"}
    con la respuesta a medida que se genera y {"tipo": "fin", "respuesta", "contexto", "metricas"} al acabar."""
    estado = _estado_inicial(**kwargs)
    estado["modo_streaming"] = True
    final = dict(estado)
    inicio = anterior = time.perf_counter()

    # stream_mode="updates" devuelve {nodo: cambios} cada vez que termina un nodo
    async for actualizacion in app.astream(estado, stream_mode="updates"):
        ahora = time.perf_counter()
        for nodo, cambios in actualizacion.items():
            if cambios:
                final.update(cambios)
            if nodo in PASOS:
                yield {"tipo": "estado", "texto": PASOS[nodo], "nodo": nodo,
                       "segundos": round(ahora - anterior, 2)}
        anterior = ahora

    prompt = final.get("prompt_generador")
    if final.get("agente_pendiente"):
        t_agente = time.perf_counter()
        async for evento in agente_eventos(final["input_usuario"], final.get("historial"), final.get("intencion")):
            if evento["tipo"] == "estado":
                yield evento
            elif evento["tipo"] == "respuesta":
                final["respuesta_final"] = evento["texto"]
                final["despensa_modificada"] = evento.get("despensa_modificada", False)
                final["herramientas_usadas"] = evento.get("herramientas", [])
        yield {"tipo": "token", "texto": final["respuesta_final"]}
        metricas_gen = {"motor": "agente", "generacion_s": round(time.perf_counter() - t_agente, 2),
                        "herramientas": len(final.get("herramientas_usadas") or [])}
    elif prompt:
        generar = stream_respuesta_vllm if final.get("motor_generador") == "vllm" else stream_respuesta_ollama
        partes, t_gen, primer_token = [], time.perf_counter(), None
        async for trozo in generar(prompt):
            if primer_token is None:
                primer_token = time.perf_counter() - t_gen
            partes.append(trozo)
            yield {"tipo": "token", "texto": trozo}
        final["respuesta_final"] = "".join(partes) or "No he podido generar una respuesta."
        metricas_gen = {"motor": final.get("motor_generador"),
                        "generacion_s": round(time.perf_counter() - t_gen, 2),
                        "primer_token_llm_s": round(primer_token or 0, 2),
                        "trozos": len(partes), "caracteres": len(final["respuesta_final"])}
    else:
        # Respuestas sin generador (saludo, sin ingredientes, sin recetas locales...): de una vez
        final["respuesta_final"] = final.get("respuesta_final") or "No he podido generar una respuesta."
        yield {"tipo": "token", "texto": final["respuesta_final"]}
        metricas_gen = {}

    yield {"tipo": "fin", "respuesta": final["respuesta_final"], "contexto": _contexto(final),
           "metricas": {"total_s": round(time.perf_counter() - inicio, 2), **metricas_gen}}