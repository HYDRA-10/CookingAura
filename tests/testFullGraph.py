import asyncio
import sys
import os

# 1. Calculamos la ruta absoluta hacia la carpeta "backend" y la añadimos al path
ruta_backend = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.append(ruta_backend)

from src.agents.graph import ejecutar_asistente

# BLOQUE DE PRUEBA LOCAL
if __name__ == "__main__":
    print("🚀 Iniciando prueba del flujo multiagente completo...")
    pregunta_prueba = "¿Qué como hoy?"
    respuesta = asyncio.run(ejecutar_asistente(pregunta_prueba))
    
    print("\n" + "="*50)
    print("👨‍🍳 RESPUESTA FINAL DEL ASISTENTE:")
    print("="*50)
    print(respuesta)