import os
from google import genai
from google.genai import types

from src.llms.prompts import tipo_comida_con_articulo, formatear_descartadas

MODELO = "gemini-3.8-flash"


def get_vertex_search_agent():
    """Crea el cliente de Vertex AI. gemini-3.8-flash solo está en global/us/eu, no en regiones sueltas."""
    return genai.Client(
        vertexai=True,
        project=os.getenv("GCP_PROJECT_ID"),
        location="global",
        http_options=types.HttpOptions(api_version="v1"),
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
                           plato: str | None = None, restricciones: str = "") -> str:
    """Busca una receta en internet. Con 'plato', busca la receta de ese plato concreto."""
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
    prompt += " Responde en español e indica de qué página web la has obtenido."

    respuesta = client.models.generate_content(
        model=MODELO,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.9 if creativa else 0.3,
        ),
    )

    texto = respuesta.text or "No se pudo obtener una receta."
    fuentes = _extraer_fuentes(respuesta)
    if fuentes:
        texto += "\n\nFuentes:\n" + "\n".join(f"- {f}" for f in fuentes)
    return texto


def generar_json_gemini(prompt: str) -> str:
    """Generación con salida JSON (sin búsqueda), para las ideas de platos."""
    print("Generando ideas de platos con Gemini...")
    respuesta = get_vertex_search_agent().models.generate_content(
        model=MODELO,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.8),
    )
    return respuesta.text or ""