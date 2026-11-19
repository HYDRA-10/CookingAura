"""Fase 1: ideas de platos. Salen de las recetas propias (Milvus) y/o se generan con un LLM."""
import json
import re

from src.config import IDEAS_MOTOR
from src.llms.prompts import tipo_comida_con_articulo, formatear_descartadas
from src.llms.ollama_client import generar_json_ollama
from src.agents.web_search_node import generar_json_gemini


def titulo_y_resumen(texto: str, max_titulo: int = 80, max_resumen: int = 160) -> tuple[str, str]:
    """Primera línea del fragmento como nombre del plato y el resto, resumido, como descripción."""
    lineas = [l.strip(" #*-•\t") for l in (texto or "").splitlines() if l.strip(" #*-•\t")]
    if not lineas:
        return "Receta sin título", ""
    titulo = lineas[0] if len(lineas[0]) <= max_titulo else lineas[0][:max_titulo].rstrip() + "…"
    resumen = " ".join(lineas[1:])
    if len(resumen) > max_resumen:
        resumen = resumen[:max_resumen].rstrip() + "…"
    return titulo, resumen


def _prompt_ideas(ingredientes: str, tipo_comida: str | None, n: int,
                  descartadas: list[str] | None, improvisar: bool, restricciones: str = "") -> str:
    momento = tipo_comida_con_articulo(tipo_comida)
    partes = [f"Eres un chef creativo. Propón {n} ideas de platos distintos entre sí"
              f"{f' para {momento}' if momento else ''}."]
    if improvisar or not ingredientes:
        partes.append("El usuario no tiene ingredientes concretos: improvisa platos sencillos "
                      "con ingredientes básicos y fáciles de conseguir.")
    else:
        partes.append(f"Deben poder prepararse principalmente con estos ingredientes: {ingredientes}. "
                      "Se pueden añadir básicos de cocina (sal, aceite, especias).")
    if restricciones:
        partes.append(f"Respeta siempre estas restricciones del usuario: {restricciones}.")
    lista = formatear_descartadas(descartadas, max_items=15)
    if lista:
        partes.append(f"No repitas ninguno de estos platos:\n{lista}")
    partes.append('Responde SOLO con JSON con este formato, en español: '
                  '{"ideas": [{"nombre": "nombre del plato", "descripcion": "una frase corta"}]}')
    return "\n\n".join(partes)


def _parsear_ideas(texto: str) -> list[dict]:
    """Extrae [{"nombre", "descripcion"}] aunque el modelo envuelva el JSON en texto."""
    try:
        datos = json.loads(texto)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}|\[.*\]", texto or "", re.DOTALL)
        try:
            datos = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            datos = {}
    ideas = datos.get("ideas", []) if isinstance(datos, dict) else datos if isinstance(datos, list) else []
    resultado = []
    for idea in ideas:
        if isinstance(idea, dict) and str(idea.get("nombre", "")).strip():
            resultado.append({
                "nombre": str(idea["nombre"]).strip()[:80],
                "descripcion": str(idea.get("descripcion", "")).strip()[:200],
            })
    return resultado


def generar_ideas_creativas(ingredientes: str, tipo_comida: str | None, n: int,
                            descartadas: list[str] | None = None, improvisar: bool = False,
                            restricciones: str = "") -> list[dict]:
    """Ideas inventadas por el LLM. Si Gemini falla, se intenta con Ollama."""
    prompt = _prompt_ideas(ingredientes, tipo_comida, n, descartadas, improvisar, restricciones)
    ya_vistos = {d.strip().lower() for d in (descartadas or [])}

    motores = ["gemini", "ollama"] if IDEAS_MOTOR == "gemini" else ["ollama"]
    for motor in motores:
        try:
            texto = generar_json_gemini(prompt) if motor == "gemini" else generar_json_ollama(prompt)
            ideas = [i for i in _parsear_ideas(texto) if i["nombre"].lower() not in ya_vistos]
            if ideas:
                return ideas[:n]
            print(f"⚠️  {motor} no devolvió ideas válidas: {texto[:200]!r}")
        except Exception as e:
            print(f"⚠️  Ideas con {motor} fallaron: {e}", flush=True)
    return []