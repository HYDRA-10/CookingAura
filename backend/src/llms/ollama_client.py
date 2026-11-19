from langchain_ollama import ChatOllama
import os

from src.llms.prompts import construir_prompt_generador, tipo_comida_con_articulo

# Conexión local al contenedor de Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODELO_EVALUADOR = os.getenv("OLLAMA_MODELO_EVALUADOR", "llama3")
MODELO_GENERADOR = os.getenv("OLLAMA_MODELO_GENERADOR", "llama3")


def get_ollama(evaluator: bool = True, num_predict: int | None = None):
    """Devuelve la instancia del LLM local. num_predict es el límite real de tokens en Ollama."""
    modelo = MODELO_EVALUADOR if evaluator else MODELO_GENERADOR
    return ChatOllama(
        model=modelo,
        base_url=OLLAMA_URL,
        temperature=0.1 if evaluator else 0.7,
        num_predict=num_predict or (10 if evaluator else 450),
        keep_alive=-1,  # Mantiene el modelo en memoria permanentemente
        client_kwargs={"timeout": 300},  # en CPU leer un prompt largo puede tardar; sin esto, sin límite
    )


# GUARDARRAÍLES SEMÁNTICOS DE ENTRADA
async def evaluar_intencion(user_input: str, contexto: str = "") -> str:
    """Clasifica el mensaje en RECETA (quiere ideas de platos), ASISTENTE (cualquier otra cosa de cocina,
    despensa, compra, preferencias...), SALUDO u OTRO (ajeno a la cocina)."""
    llm = get_ollama(True, num_predict=6)
    previo = f"\nÚltimo mensaje del asistente (contexto): {contexto[:300]}\n" if contexto else ""

    prompt = f"""
    Clasifica el mensaje del usuario a un asistente de cocina.{previo}
    Mensaje del usuario: {user_input}

    Opciones:
    'RECETA' si pide que le propongas platos o ideas de qué cocinar o comer ("¿qué ceno?", "ideas con pollo").
    'ASISTENTE' para cualquier otra cosa de cocina o de su cocina: su despensa (qué tiene, añadir, gastar),
    la lista de la compra, sus alergias o gustos, lo que ha cocinado, sus recetas guardadas, dudas sobre una
    receta o técnica, o una respuesta a lo que le dijo el asistente ("sí", "para 6 personas").
    'SALUDO' si solo saluda.
    'OTRO' si no tiene nada que ver con la cocina ni con la comida.

    Responde ÚNICAMENTE con una opción.
    """

    print("Enviando contexto a Ollama para evaluar la intención...")
    respuesta = await llm.ainvoke(prompt)
    texto = respuesta.content.strip().upper()
    print(f"   Clasificador ({MODELO_EVALUADOR}) respondió: {texto!r}", flush=True)
    # El modelo a veces añade comillas o puntuación: nos quedamos con la etiqueta
    for opcion in ("ASISTENTE", "RECETA", "SALUDO", "OTRO"):
        if opcion in texto:
            return opcion
    return "ASISTENTE"     # ante la duda, mejor que lo atienda el agente que rechazarlo


async def evaluar_ingredientes(ingredientes: str) -> str:
    """Pide a Ollama que filtre los ingredientes que no son reales."""
    llm = get_ollama(True, num_predict=200)

    prompt = f"""
    Eres un juez culinario estricto y directo.

    Ingredientes disponibles en la despensa:
    {ingredientes}

    ¿Son en realidad todos los ingredientes disponibles en la despensa reales?

    En el caso de que sí lo sean, devuelve la respuesta en el siguiente formato ['SI', una lista de los ingredientes válidos]. No des explicaciones adicionales.
    En el caso de que lo sean parcialmente, es decir, algunos sí lo son y otros no, devuelve la respuesta en el siguiente formato ['SI', una lista de los ingredientes válidos]. No des explicaciones adicionales.
    En el caso de que no lo sean, devuelve la respuesta en el siguiente formato ['NO', []]. No des explicaciones adicionales.
    """

    print(f"Enviando contexto a Ollama para evaluación... Con los ingredientes: {ingredientes}")
    respuesta = await llm.ainvoke(prompt)
    return respuesta.content.strip().upper()


# GUARDARRAÍLES SEMÁNTICOS DE SALIDA
def evaluar_recetas(ingredientes: str, recetas: str, tipo_comida: str | None = None) -> str:
    """Decide si alguna receta encontrada sirve con los ingredientes (y para el tipo de comida)."""
    llm = get_ollama(True, num_predict=5)
    momento = tipo_comida_con_articulo(tipo_comida)
    condicion_momento = f" y que además sea apropiada para {momento}" if momento else ""

    prompt = f"""
    Eres un juez culinario estricto y directo.

    Ingredientes disponibles en la despensa:
    {ingredientes}

    Recetas sugeridas de la base de datos:
    {recetas}

    ¿Alguna de estas recetas se puede cocinar PRINCIPALMENTE usando los ingredientes disponibles{condicion_momento}?
    Responde ÚNICAMENTE con la palabra 'SI' o con la palabra 'NO'. No des explicaciones adicionales.
    """

    print(f"Enviando contexto a Ollama para evaluación... Ingredientes: {ingredientes} | Tipo: {tipo_comida}")
    respuesta = llm.invoke(prompt)
    return respuesta.content.strip().upper()


# NODO GENERADOR
def generar_respuesta_final_ollama(ingredientes: str, receta_elegida: str,
                                   tipo_comida: str | None = None,
                                   peticion: str | None = None,
                                   descartadas: list[str] | None = None,
                                   restricciones: str = "") -> str:
    llm = get_ollama(False)
    prompt = construir_prompt_generador(ingredientes, receta_elegida, tipo_comida, peticion, descartadas, restricciones)
    print("Enviando contexto a Ollama para generación de respuesta...")
    respuesta = llm.invoke(prompt)
    return respuesta.content


async def stream_respuesta_ollama(prompt: str):
    """Genera la respuesta token a token (para el endpoint /chat/stream)."""
    llm = get_ollama(False)
    print("Generando respuesta en streaming con Ollama...")
    async for trozo in llm.astream(prompt):
        if trozo.content:
            yield trozo.content


def generar_json_ollama(prompt: str, num_predict: int = 350) -> str:
    """Generación con salida JSON garantizada (format="json"), para las ideas de platos."""
    llm = ChatOllama(
        model=MODELO_GENERADOR,
        base_url=OLLAMA_URL,
        temperature=0.8,
        num_predict=num_predict,
        format="json",
        keep_alive=-1,
        client_kwargs={"timeout": 300},
    )
    print("Generando ideas de platos con Ollama...")
    return llm.invoke(prompt).content


async def precalentar_modelos(modelos: list[str]) -> None:
    """Carga en memoria los modelos de Ollama al arrancar (una generación de 1 token).
    Así la primera petición del usuario no paga la carga del modelo desde disco."""
    import httpx
    import time
    async with httpx.AsyncClient(base_url=OLLAMA_URL, timeout=600) as cliente:
        for modelo in dict.fromkeys(m for m in modelos if m):
            t0 = time.perf_counter()
            try:
                r = await cliente.post("/api/generate", json={"model": modelo, "prompt": "hola", "stream": False,
                                                               "keep_alive": -1, "options": {"num_predict": 1}})
                r.raise_for_status()
                carga = (r.json().get("load_duration") or 0) / 1e9
                print(f"🔥 Ollama: '{modelo}' listo en {time.perf_counter() - t0:.1f} s (carga {carga:.1f} s)", flush=True)
            except Exception as e:
                print(f"⚠️  Ollama: no se pudo precargar '{modelo}': {type(e).__name__} {e}", flush=True)