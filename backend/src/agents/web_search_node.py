import concurrent.futures
import os
import time

from google import genai
from google.genai import types

from src.config import GEMINI_TIMEOUT
from src.llms.prompts import tipo_comida_con_articulo, formatear_descartadas

# Hilos para las llamadas a Gemini: así se puede dejar de esperar sin bloquear la petición
_HILOS = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="gemini")

# Formato de la receta cuando se muestra directamente al usuario (sin reescribirla con el modelo local)
FORMATO_RECETA = (
    "\n\nEscribe la receta lista para mostrarla al usuario, en Markdown y en español, con este formato:\n"
    "1. Una frase breve de presentación.\n"
    "2. **Ingredientes** con sus cantidades (indica para cuántas raciones).\n"
    "3. **Preparación** en un máximo de 6 pasos numerados, claros y concretos.\n"
    "4. Opcionalmente, un consejo breve al final.\n"
    "Si la receta original lleva algo que el usuario no puede tomar, sustitúyelo y avisa en una línea. "
    "No incluyas enlaces ni nombres de páginas en el texto: las fuentes se añaden aparte."
)


def _con_limite(funcion, *args, **kwargs):
    """Ejecuta una llamada a Gemini con un tiempo máximo (GEMINI_TIMEOUT). Si se supera, lanza TimeoutError."""
    futuro = _HILOS.submit(funcion, *args, **kwargs)
    try:
        return futuro.result(timeout=GEMINI_TIMEOUT)
    except concurrent.futures.TimeoutError:
        futuro.cancel()
        raise TimeoutError(f"Gemini no ha respondido en {GEMINI_TIMEOUT} s")

MODELO = "gemini-3.8-flash"


def get_vertex_search_agent():
    """Crea el cliente de Vertex AI. gemini-3.8-flash solo está en global/us/eu, no en regiones sueltas."""
    return genai.Client(
        vertexai=True,
        project=os.getenv("GCP_PROJECT_ID"),
        location="global",
        # Límite también en el cliente HTTP (milisegundos), para que la llamada no siga abierta en segundo plano
        http_options=types.HttpOptions(api_version="v1", timeout=(GEMINI_TIMEOUT + 5) * 1000),
    )


def _extraer_fuentes(respuesta) -> list[str]:
    """Saca las fuentes reales del grounding_metadata (sin depender de lo que escriba el modelo)."""
    fuentes, vistos = [], set()
    if not respuesta.candidates:
        return fuentes
    metadata = respuesta.candidates[0].grounding_metadata
    if not metadata or not metadata.grounding_chunks:
        return fuentes
    for chunk in metadata.grounding_chunks:
        if chunk.web and chunk.web.uri and chunk.web.uri not in vistos:
            vistos.add(chunk.web.uri)
            fuentes.append(f"{chunk.web.title or 'Fuente'}: {chunk.web.uri}")
    return fuentes


def buscar_receta_internet(ingredientes: str, tipo_comida: str | None = None,
                           descartadas: list[str] | None = None, creativa: bool = False,
                           plato: str | None = None, restricciones: str = "", formato_final: bool = False) -> str:
    """Busca una receta en internet. Con 'plato', busca la receta de ese plato concreto.
    Con formato_final, la pide ya redactada para mostrarla al usuario tal cual.
    Si Gemini tarda más de GEMINI_TIMEOUT segundos, lanza TimeoutError."""
    print(f"Buscando en Internet vía Vertex AI Grounding... (plato: {plato or '-'})")

    client = get_vertex_search_agent()

    momento = tipo_comida_con_articulo(tipo_comida)
    para_momento = f" apropiada para {momento}" if momento else ""

    if plato:
        prompt = f"Busca y devuelve una receta de «{plato}»{para_momento}."
        if ingredientes:
            prompt += f" Si es posible, que aproveche estos ingredientes: {ingredientes}."
    else:
        estilo = ("una receta original o poco habitual (evita los platos más típicos)"
                  if creativa else "una receta breve y sencilla")
        if ingredientes:
            prompt = (f"Busca y devuelve {estilo}{para_momento} que se pueda cocinar "
                      f"principalmente con estos ingredientes: {ingredientes}.")
        else:
            prompt = f"Busca y devuelve {estilo}{para_momento} con ingredientes básicos y fáciles de conseguir."
        lista_descartadas = formatear_descartadas(descartadas)
        if lista_descartadas:
            prompt += f"\n\nEl usuario ya ha descartado estas propuestas; busca un plato distinto:\n{lista_descartadas}"
    if restricciones:
        prompt += f" Debe respetar estas restricciones del usuario: {restricciones}."
    prompt += FORMATO_RECETA if formato_final else " Responde en español e indica de qué página web la has obtenido."

    t0 = time.perf_counter()
    respuesta = _con_limite(
        client.models.generate_content,
        model=MODELO,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.9 if creativa else 0.3,
        ),
    )
    print(f"   Gemini respondió en {time.perf_counter() - t0:.1f} s", flush=True)

    texto = respuesta.text or "No se pudo obtener una receta."
    fuentes = _extraer_fuentes(respuesta)
    if fuentes:
        texto += "\n\nFuentes:\n" + "\n".join(f"- {f}" for f in fuentes)
    return texto


def generar_json_gemini(prompt: str) -> str:
    """Generación con salida JSON (sin búsqueda), para las ideas de platos."""
    print("Generando ideas de platos con Gemini...")
    respuesta = _con_limite(
        get_vertex_search_agent().models.generate_content,
        model=MODELO,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.8),
    )
    return respuesta.text or ""