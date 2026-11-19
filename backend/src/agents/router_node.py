import os
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def clasificar_intencion(estado: dict) -> dict:
    """
    Analiza el mensaje del usuario y decide el flujo a seguir.
    """
    mensaje = estado.get("mensaje", "")
    
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    llm = ChatOllama(base_url=ollama_url, model="llama3", temperature=0)
    
    prompt = PromptTemplate.from_template(
        """Eres un clasificador de intenciones estricto para un asistente de cocina.
        Analiza el siguiente mensaje del usuario y responde ÚNICAMENTE con una de estas tres palabras:
        - SALUDO: Si el usuario solo dice hola, buenos días, gracias, etc.
        - RECETA: Si el usuario menciona comida, ingredientes, hambre, cocinar, recetas, o qué comer.
        - OTRO: Si el usuario habla de política, matemáticas, código, u otros temas no relacionados con comida.
        
        Mensaje: {mensaje}
        
        Clasificación:"""
    )
    
    cadena = prompt | llm | StrOutputParser()
    clasificacion = cadena.invoke({"mensaje": mensaje}).strip().upper()
    
    # Limpieza de seguridad por si el LLM añade texto extra
    if "RECETA" in clasificacion:
        intencion = "RECETA"
    elif "SALUDO" in clasificacion:
        intencion = "SALUDO"
    else:
        intencion = "OTRO"
        
    return {"intencion": intencion}