"""Evaluación sin LLM: cuántos ingredientes del usuario aparecen en el texto de una receta.

Sustituye a la llamada de Ollama del nodo evaluador (en CPU costaba ~1 minuto por receta)."""
import re
import unicodedata

_STOPWORDS = {"de", "del", "la", "el", "los", "las", "y", "e", "en", "con", "al", "a",
              "un", "una", "unos", "unas", "o", "sin", "para", "por"}


def _normalizar(texto: str) -> str:
    """Minúsculas y sin tildes: 'Limón' -> 'limon'."""
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def _raiz(palabra: str) -> str:
    """Raíz aproximada para igualar singular y plural en español.
    huevos/huevo -> huevo · tomates/tomate -> tomat · limones/limon -> limon · nueces/nuez -> nuez"""
    if palabra.endswith("ces") and len(palabra) > 4:
        palabra = palabra[:-3] + "z"
    elif palabra.endswith("s") and len(palabra) > 3:
        palabra = palabra[:-1]
    if palabra.endswith("e") and len(palabra) > 3:
        palabra = palabra[:-1]
    return palabra


def raices(texto: str) -> set[str]:
    return {_raiz(p) for p in re.findall(r"[a-z]+", _normalizar(texto))
            if len(p) > 2 and p not in _STOPWORDS}


def separar_ingredientes(texto: str) -> list[str]:
    """'huevos, patatas, aceite de oliva' -> ['huevos', 'patatas', 'aceite de oliva']"""
    return [i.strip() for i in (texto or "").split(",") if i.strip()]


def contar_coincidencias(ingredientes: list[str], texto_receta: str) -> int:
    """Número de ingredientes del usuario que aparecen en la receta.
    Un ingrediente compuesto ('pechuga de pollo') cuenta si aparece cualquiera de sus palabras."""
    raices_receta = raices(texto_receta)
    return sum(1 for ingrediente in ingredientes if raices(ingrediente) & raices_receta)