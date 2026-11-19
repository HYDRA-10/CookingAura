from langchain_openai import ChatOpenAI
import os

from src.llms.prompts import construir_prompt_generador
from src.config import VLLM_MODELO

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1")
MODELO_GENERADOR = VLLM_MODELO   # configurable en el .env


def get_vllm_generator():
    print(f"Conectando a vLLM en {VLLM_URL}...")
    return ChatOpenAI(
        model=MODELO_GENERADOR,
        base_url=VLLM_URL,
        api_key="vllm-local",
        temperature=0.7,
        max_tokens=450,
        timeout=300,
        max_retries=1,
    )


def generar_respuesta_final_vllm(ingredientes: str, receta_elegida: str,
                                 tipo_comida: str | None = None,
                                 peticion: str | None = None,
                                 descartadas: list[str] | None = None,
                                 restricciones: str = "") -> str:
    llm = get_vllm_generator()
    prompt = construir_prompt_generador(ingredientes, receta_elegida, tipo_comida, peticion, descartadas, restricciones)
    respuesta = llm.invoke(prompt)
    return respuesta.content


async def stream_respuesta_vllm(prompt: str):
    """Genera la respuesta token a token (para el endpoint /chat/stream)."""
    llm = get_vllm_generator()
    async for trozo in llm.astream(prompt):
        if trozo.content:
            yield trozo.content