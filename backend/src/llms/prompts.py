"""Utilidades compartidas para el tipo de comida y los prompts de generación."""
import re

# Tipo de comida -> forma con artículo para usar en los prompts
_ARTICULOS = {
    "desayuno": "el desayuno",
    "brunch": "el brunch",
    "almuerzo": "el almuerzo",
    "comida": "la comida",
    "merienda": "la merienda",
    "cena": "la cena",
}

# Lo que puede llegar del frontend ("Desayunar", "Cenar"...) -> valor canónico
_SINONIMOS = {
    "desayunar": "desayuno",
    "almorzar": "almuerzo",
    "comer": "comida",
    "merendar": "merienda",
    "cenar": "cena",
}

# Detección en texto libre (chat o audio). El orden importa: lo más específico primero.
_PATRONES = [
    (r"\bdesayun\w*", "desayuno"),
    (r"\bbrunch\b", "brunch"),
    (r"\balm(?:uerz|orz)\w*", "almuerzo"),
    (r"\bmeriend\w*|\bmerend\w*", "merienda"),
    (r"\bcen(?:a|as|ar|o|amos)\b", "cena"),
    (r"\bcomida\b|\bpara comer\b", "comida"),
]


def normalizar_tipo_comida(valor: str | None) -> str | None:
    """'Desayunar' -> 'desayuno', 'cena' -> 'cena', valores desconocidos -> None."""
    if not valor:
        return None
    v = valor.strip().lower()
    v = _SINONIMOS.get(v, v)
    return v if v in _ARTICULOS else None


def detectar_tipo_comida(texto: str | None) -> str | None:
    """Intenta deducir el tipo de comida de un mensaje libre del usuario."""
    if not texto:
        return None
    texto = texto.lower()
    for patron, tipo in _PATRONES:
        if re.search(patron, texto):
            return tipo
    return None


def tipo_comida_con_articulo(tipo_comida: str | None) -> str | None:
    """'cena' -> 'la cena'; None si no hay tipo."""
    return _ARTICULOS.get(tipo_comida) if tipo_comida else None


def formatear_descartadas(descartadas: list[str] | None, max_items: int = 5) -> str:
    """Lista breve de propuestas que el usuario ya ha rechazado, para meterla en un prompt."""
    if not descartadas:
        return ""
    return "\n".join(f"- {d.strip()[:300]}" for d in descartadas[-max_items:] if d and d.strip())


def construir_prompt_generador(ingredientes: str, receta: str,
                               tipo_comida: str | None = None,
                               peticion: str | None = None,
                               descartadas: list[str] | None = None,
                               restricciones: str = "") -> str:
    """Prompt común para el generador final (Ollama y vLLM)."""
    momento = tipo_comida_con_articulo(tipo_comida)

    lineas = ["Eres un chef amigable y entusiasta."]
    if peticion:
        lineas.append(f'El usuario ha pedido: "{peticion}"')
    if momento:
        lineas.append(
            f"La receta es para {momento}: ajusta el plato, las raciones y el tiempo de "
            f"preparación a lo que es apropiado para {momento}. Si la receta de referencia "
            f"no encaja del todo para {momento}, adáptala y explica brevemente el cambio."
        )
    lista_descartadas = formatear_descartadas(descartadas)
    if lista_descartadas:
        lineas.append(
            "El usuario NO quiere ninguna de estas propuestas anteriores. Propón un plato "
            f"claramente distinto (no repitas ni el plato ni su nombre):\n{lista_descartadas}"
        )
    if restricciones:
        lineas.append(f"IMPORTANTE, restricciones del usuario: {restricciones}. Si la receta de referencia usa "
                      "algo de eso, sustitúyelo por una alternativa adecuada y avisa del cambio.")
    if ingredientes:
        lineas.append(f"Ingredientes disponibles: {ingredientes}")
    else:
        lineas.append("El usuario no ha indicado ingredientes concretos: usa los propios del plato, "
                      "fáciles de conseguir.")
    lineas.append(f"Receta de referencia:\n{receta}")
    lineas.append("Redacta en español un mensaje breve y entusiasta animando al usuario a cocinarla: "
                  "la lista de ingredientes con cantidades y, como máximo, 6 pasos cortos. "
                  "Sin introducciones largas ni consejos extra.")
    return "\n\n".join(lineas)