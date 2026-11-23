"""Configuración del backend leída del fichero .env (y del entorno).

Se importa lo primero en main.py: así cualquier os.getenv() posterior en otros módulos
(RAG_*, OLLAMA_*...) también ve los valores del .env.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # sin python-dotenv solo se usan las variables del entorno (p. ej. env_file del compose)
    load_dotenv = None

# En Docker el .env está montado en /app/.env (= padre de src/); en local suele estar un nivel más arriba
_SRC = Path(__file__).resolve().parent
for _candidato in (_SRC.parent / ".env", _SRC.parent.parent / ".env"):
    if load_dotenv and _candidato.is_file():
        # override=True: el .env es la única fuente de configuración. Como está montado en el
        # contenedor, editarlo y hacer `docker compose restart cookingaura-app-backend` basta.
        load_dotenv(_candidato, override=True)
        print(f"⚙️  Configuración cargada desde {_candidato}")
        break


def _entero(nombre: str, defecto: int, minimo: int | None = None) -> int:
    valor = os.getenv(nombre, "").strip()
    try:
        resultado = int(valor) if valor else defecto
    except ValueError:
        print(f"⚠️  {nombre}='{valor}' no es un número; se usa {defecto}")
        resultado = defecto
    return max(minimo, resultado) if minimo is not None else resultado


def _opcion(nombre: str, defecto: str, validas: set[str]) -> str:
    valor = os.getenv(nombre, defecto).strip().lower()
    if valor not in validas:
        print(f"⚠️  {nombre}='{valor}' no es válido ({', '.join(sorted(validas))}); se usa '{defecto}'")
        return defecto
    return valor


# --- Ideas de platos (fase 1) ---
# Número total de ideas que se proponen
IDEAS_TOTAL = _entero("IDEAS_TOTAL", 5, minimo=2)
# En modo automático, cuántas de esas ideas pueden salir de tus recetas (el resto son creativas)
IDEAS_MAX_LOCALES = _entero("IDEAS_MAX_LOCALES", 3, minimo=0)
# Motor de las ideas creativas: ollama (local) | gemini (Vertex AI, mucho más rápido si estás en CPU)
IDEAS_MOTOR = _opcion("IDEAS_MOTOR", "ollama", {"ollama", "gemini"})
# Ingredientes de la despensa que se usan cuando el usuario pide ideas sin indicarlos
# (los principales y adecuados al tipo de comida, elegidos al azar). También los que ofrece la selección rápida.
DESPENSA_MAX_INGREDIENTES = _entero("DESPENSA_MAX_INGREDIENTES", 20, minimo=5)
# Mínimo de ingredientes del usuario que debe usar una receta propia para proponerla como idea
EVAL_MIN_COINCIDENCIAS = _entero("EVAL_MIN_COINCIDENCIAS", 2, minimo=1)

# --- Generador ---
# Tope de caracteres de la receta de referencia que recibe el generador
GENERADOR_MAX_CHARS = _entero("GENERADOR_MAX_CHARS", 2000, minimo=200)

# --- Agente del chat libre (usa las herramientas MCP) ---
# Modelo de Ollama con soporte de herramientas (qwen2.5, llama3.1...). Vacío = el del generador
AGENTE_MODELO = (os.getenv("AGENTE_MODELO") or os.getenv("OLLAMA_MODELO_GENERADOR") or "llama3.1").strip()
# Máximo de rondas de herramientas por mensaje (evita bucles)
AGENTE_MAX_PASOS = _entero("AGENTE_MAX_PASOS", 5, minimo=1)
# Mensajes anteriores del chat que se envían como contexto
AGENTE_HISTORIAL = _entero("AGENTE_HISTORIAL", 8, minimo=0)
# Segundos máximos que se espera a cada respuesta del modelo del agente
AGENTE_TIMEOUT = _entero("AGENTE_TIMEOUT", 180, minimo=30)

# --- Gemini (Vertex AI) ---
# Segundos máximos de espera a Gemini (búsqueda web e ideas); pasado ese tiempo se sigue sin él
GEMINI_TIMEOUT = _entero("GEMINI_TIMEOUT", 25, minimo=5)
# La receta que encuentra Gemini se muestra tal cual (si), o la vuelve a redactar el modelo local (no)
WEB_RECETA_DIRECTA = _opcion("WEB_RECETA_DIRECTA", "si", {"si", "no"}) == "si"

# --- Arranque ---
# Cargar los modelos de Ollama al arrancar el backend (la primera petición no paga la carga)
PRECALENTAR_MODELOS = _opcion("PRECALENTAR_MODELOS", "si", {"si", "no"}) == "si"
# Cargar el modelo de embeddings (bge-m3) al arrancar en vez de en la primera búsqueda
PRECARGAR_EMBEDDINGS = _opcion("PRECARGAR_EMBEDDINGS", "si", {"si", "no"}) == "si"

# --- vLLM ---
# Debe coincidir con el modelo que sirve el contenedor vllm-server (--model)
VLLM_MODELO = os.getenv("VLLM_MODELO", "Qwen/Qwen2.5-7B-Instruct").strip()