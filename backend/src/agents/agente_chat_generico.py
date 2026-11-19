"""Agente del chat libre: un LLM con herramientas MCP que conversa con el usuario y actúa sobre sus datos
(despensa, lista de la compra, preferencias, historial y recetas guardadas).

El bucle es explícito (sin depender de create_react_agent, cuya API cambia entre versiones):
el modelo responde o pide herramientas; se ejecutan, se le devuelven los resultados y vuelve a decidir.
"""
import re
import time
import unicodedata
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from src.config import AGENTE_MODELO, AGENTE_MAX_PASOS, AGENTE_HISTORIAL, AGENTE_TIMEOUT
from src.llms.ollama_client import OLLAMA_URL
from .mcp_client import herramientas_mcp, gestor_mcp, ErrorArranqueMCP, texto_resultado_mcp

SISTEMA = """Eres CookingAura, un asistente de cocina amable y práctico. Hoy es {fecha}.

Ayudas con la despensa del usuario, su lista de la compra, sus alergias y gustos, el historial de lo que
ha cocinado y sus recetas guardadas, además de cualquier duda de cocina (técnicas, sustituciones,
tiempos, conservación, adaptar raciones...).

Reglas:
- Para saber qué hay en la despensa, la lista de la compra, sus preferencias o su historial, usa SIEMPRE
  la herramienta correspondiente. Nunca lo inventes.
- Modifica datos (añadir, actualizar, gastar, eliminar, guardar) solo cuando el usuario lo pida claramente.
  Si falta un dato imprescindible (por ejemplo, la cantidad), pregúntalo antes.
- Si dice que ha cocinado un plato, ofrécele descontar los ingredientes de la despensa y apuntarlo en su
  historial; hazlo cuando lo confirme.
- Antes de recomendar platos o ingredientes, consulta sus preferencias y respeta sus alergias.
- En esta conversación NO hay botones: nunca le pidas al usuario que pulse o haga clic en nada.
  Si te pide una receta o ideas de qué cocinar, respóndele tú directamente (ingredientes y pasos breves).
- Cantidades de la despensa: si dice cuánto tiene o le queda de algo (aunque sea cero), usa
  actualizar_ingrediente; si lo ha comprado, anadir_ingrediente; si lo ha gastado, consumir_ingredientes.
- Tras modificar algo, confirma en una frase qué has hecho.
- Responde en español, breve y claro. Si te preguntan algo ajeno a la cocina, dilo con amabilidad."""

# Herramientas solo para el grafo (no tienen sentido en la conversación)
HERRAMIENTAS_INTERNAS = {"estado_recetas", "ingredientes_principales"}
MODIFICAN_DESPENSA = {"anadir_ingrediente", "actualizar_ingrediente", "consumir_ingredientes", "eliminar_ingrediente"}

DESCRIPCIONES = {
    "ver_despensa": "🥫 Consultando la despensa",
    "anadir_ingrediente": "➕ Añadiendo a la despensa",
    "actualizar_ingrediente": "✏️ Actualizando la despensa",
    "consumir_ingredientes": "➖ Descontando de la despensa",
    "eliminar_ingrediente": "🗑️ Eliminando de la despensa",
    "buscar_recetas_guardadas": "📚 Buscando en tus recetas",
    "listar_documentos_recetas": "🗂️ Revisando tus documentos",
    "ver_preferencias": "🚫 Consultando tus preferencias",
    "guardar_preferencia": "💾 Guardando una preferencia",
    "eliminar_preferencia": "🗑️ Quitando una preferencia",
    "ver_lista_compra": "🛒 Consultando la lista de la compra",
    "anadir_a_lista_compra": "🛒 Añadiendo a la lista de la compra",
    "quitar_de_lista_compra": "🛒 Quitando de la lista de la compra",
    "registrar_plato_cocinado": "📝 Apuntando en tu historial",
    "ver_historial": "📅 Consultando tu historial",
}

# Grupos de herramientas según el tema del mensaje. Con un modelo pequeño en CPU, ofrecer solo las
# relevantes acorta mucho el prompt (cada herramienta añade su descripción) y reduce los errores.
GRUPOS = [   # patrones sin tildes: el texto se normaliza antes de compararlo
    ({"ver_despensa", "anadir_ingrediente", "actualizar_ingrediente", "consumir_ingredientes", "eliminar_ingrediente"},
     r"despensa|dispensa|nevera|frigo|me queda|quedan|tengo|hay |anad|agreg|compre|he comprado|gast|use\b|quit|elimin|borr|"
     r"cantidad|\bkg\b|gramos|litro|unidad|agotad|cocine|he hecho|he preparado"),
    ({"ver_lista_compra", "anadir_a_lista_compra", "quitar_de_lista_compra", "ver_despensa"},
     r"\b(la\s+compra|lista|comprar|supermercado|super|mercado)\b|me\s+falta|necesito\s+comprar"),
    ({"ver_preferencias", "guardar_preferencia", "eliminar_preferencia"},
     r"alerg|intoler|celiac|gluten|lactosa|vegan|vegetarian|dieta|no me gusta|odio|me encanta|me gusta|prefer|evit"),
    ({"registrar_plato_cocinado", "ver_historial", "consumir_ingredientes"},
     r"cocine|he cocinado|he hecho|he preparado|\bcomi\b|\bcene\b|desayune|historial|ayer|semana|ultimamente"),
    ({"buscar_recetas_guardadas", "listar_documentos_recetas", "ver_preferencias"},
     r"receta|recetario|documento|guardad|mis recetas|libro"),
]


def _sin_tildes(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def hay_tema(texto: str) -> bool:
    """True si el mensaje toca algún tema con herramientas (despensa, compra, preferencias, historial, recetas)."""
    texto = _sin_tildes(texto)
    return any(re.search(patron, texto) for _, patron in GRUPOS)


def _seleccionar(herramientas: list, texto: str, solo_charla: bool = False) -> list:
    disponibles = [h for h in herramientas if h.name not in HERRAMIENTAS_INTERNAS]
    texto = _sin_tildes(texto)
    nombres = set()
    for grupo, patron in GRUPOS:
        if re.search(patron, texto):
            nombres |= grupo
    if nombres:
        return [h for h in disponibles if h.name in nombres]
    # Tema no reconocido: pregunta suelta o ajena a la cocina -> sin herramientas (rápido);
    # continuación de una conversación ("sí", "vale, hazlo") -> todas
    return [] if solo_charla else disponibles


def _mensajes(mensaje: str, historial: list[dict] | None) -> list:
    fecha = datetime.now().strftime("%A %d/%m/%Y")
    mensajes = [SystemMessage(SISTEMA.format(fecha=fecha))]
    for m in (historial or [])[-AGENTE_HISTORIAL:] if AGENTE_HISTORIAL else []:
        contenido = (m.get("content") or "")[:1500]
        if not contenido:
            continue
        mensajes.append(HumanMessage(contenido) if m.get("role") == "user" else AIMessage(contenido))
    mensajes.append(HumanMessage(mensaje))
    return mensajes


async def agente_eventos(mensaje: str, historial: list[dict] | None = None, intencion: str | None = None):
    """Conversa con el usuario usando herramientas. Emite eventos:
    {"tipo": "estado", "texto", "nodo", "segundos"} por cada paso y, al final,
    {"tipo": "respuesta", "texto", "herramientas", "despensa_modificada"}."""
    try:
        # Sesión MCP compartida (abierta al arrancar el backend); temporal solo si no está activa
        async with herramientas_mcp() as herramientas:
            # Sin tema reconocible, las herramientas solo se ofrecen si continúa una conversación ("sí, hazlo");
            # una pregunta suelta ("¿cuántas calorías tiene un aguacate?") se responde sin ellas: mucho más rápido
            solo_charla = not historial
            async for evento in _conversar(herramientas, mensaje, historial, solo_charla=solo_charla):
                yield evento
    except ErrorArranqueMCP as e:
        print(f"⚠️  Agente sin herramientas: {e}", flush=True)
        yield {"tipo": "respuesta", "herramientas": [], "despensa_modificada": False,
               "texto": "Ahora mismo no puedo acceder a tu despensa ni al resto de tus datos: el servidor de "
                        "herramientas no ha arrancado. Revisa los logs del backend e inténtalo de nuevo."}


async def _conversar(herramientas: list, mensaje: str, historial: list[dict] | None, solo_charla: bool = False):
    """Bucle del agente: el modelo responde o pide herramientas, se ejecutan y vuelve a decidir."""
    # El tema se decide con el mensaje y lo último de la conversación (para respuestas como "sí, hazlo")
    ultimos = " ".join(m.get("content", "")[:300] for m in (historial or [])[-2:])
    seleccion = _seleccionar(herramientas, f"{mensaje} {ultimos}", solo_charla)
    # Se ofrecen solo las del tema, pero si el modelo pide otra que existe, también se ejecuta
    por_nombre = {h.name: h for h in herramientas if h.name not in HERRAMIENTAS_INTERNAS}
    print(f"-> Agente: {len(seleccion)} herramientas ofrecidas ({', '.join(sorted(h.name for h in seleccion))})",
          flush=True)

    llm = ChatOllama(model=AGENTE_MODELO, base_url=OLLAMA_URL, temperature=0.3, num_predict=700,
                     keep_alive=-1, client_kwargs={"timeout": AGENTE_TIMEOUT})
    if seleccion:
        llm = llm.bind_tools(seleccion)
    mensajes = _mensajes(mensaje, historial)
    usadas, modificada = [], False

    for paso in range(1, AGENTE_MAX_PASOS + 1):
        t0 = time.perf_counter()
        try:
            respuesta = await llm.ainvoke(mensajes)
        except Exception as e:
            # Normalmente un timeout: el modelo no ha contestado a tiempo (CPU saturada o falta de memoria)
            espera = time.perf_counter() - t0
            print(f"⚠️  Agente: el modelo '{AGENTE_MODELO}' no ha respondido en el paso {paso} "
                  f"({type(e).__name__} tras {espera:.0f} s)", flush=True)
            hecho = f" Antes de eso he usado: {', '.join(dict.fromkeys(usadas))}." if usadas else ""
            yield {"tipo": "respuesta", "herramientas": usadas, "despensa_modificada": modificada,
                   "texto": f"⏳ El modelo está tardando demasiado en responder ({espera:.0f} s) y he tenido que "
                            f"parar.{hecho} Suele deberse a que el ordenador va justo de memoria o de CPU. "
                            "Inténtalo de nuevo en un momento."}
            return
        pensar = time.perf_counter() - t0
        mensajes.append(respuesta)

        if not respuesta.tool_calls:
            yield {"tipo": "estado", "texto": "💬 Respuesta redactada", "nodo": "agente", "segundos": round(pensar, 2)}
            texto = (respuesta.content or "").strip() or "Hecho. ¿Te ayudo con algo más?"
            yield {"tipo": "respuesta", "texto": texto, "herramientas": usadas, "despensa_modificada": modificada}
            return

        yield {"tipo": "estado", "texto": f"🧠 Decidiendo qué hacer (paso {paso})", "nodo": "agente",
               "segundos": round(pensar, 2)}
        for llamada in respuesta.tool_calls:
            nombre, argumentos = llamada["name"], llamada.get("args") or {}
            t1 = time.perf_counter()
            herramienta = por_nombre.get(nombre)
            if herramienta is None:
                resultado = f"Error: la herramienta '{nombre}' no está disponible."
            else:
                try:
                    resultado = texto_resultado_mcp(await herramienta.ainvoke(argumentos))
                except Exception as e:
                    resultado = f"Error al ejecutar {nombre}: {e}"
                    # Si el servidor se ha caído, se reabre la sesión y se repite la llamada una vez
                    if gestor_mcp.notificar_error(e) and await gestor_mcp.esperar_activo():
                        por_nombre = {h.name: h for h in gestor_mcp.herramientas
                                      if h.name not in HERRAMIENTAS_INTERNAS}
                        try:
                            resultado = texto_resultado_mcp(await por_nombre[nombre].ainvoke(argumentos))
                        except Exception as e2:
                            resultado = f"Error al ejecutar {nombre}: {e2}"
            print(f"   🔧 {nombre}({argumentos}) -> {resultado[:150]!r}", flush=True)
            usadas.append(nombre)
            modificada |= nombre in MODIFICAN_DESPENSA
            yield {"tipo": "estado", "texto": DESCRIPCIONES.get(nombre, f"🔧 {nombre}"),
                   "nodo": f"herramienta:{nombre}", "segundos": round(time.perf_counter() - t1, 2)}
            mensajes.append(ToolMessage(content=resultado[:4000], tool_call_id=llamada.get("id") or nombre,
                                        name=nombre))

    yield {"tipo": "respuesta", "texto": "He hecho varias consultas, pero no he llegado a una respuesta clara. "
                                         "¿Me lo puedes concretar un poco más?",
           "herramientas": usadas, "despensa_modificada": modificada}


async def ejecutar_agente(mensaje: str, historial: list[dict] | None = None, intencion: str | None = None) -> dict:
    """Versión sin streaming: devuelve el evento final de respuesta."""
    final = {}
    async for evento in agente_eventos(mensaje, historial, intencion):
        if evento["tipo"] == "respuesta":
            final = evento
    return final