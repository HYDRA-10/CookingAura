import sys
import os

# 1. Calculamos la ruta absoluta hacia la carpeta "backend" y la añadimos al path
ruta_backend = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.append(ruta_backend)

from src.llms.vllm_client import generar_respuesta_final_vllm

if __name__ == "__main__":
    ingredientes_prueba = "- pollo\n- arroz\n- cebolla"
    receta_prueba = "Pollo al curry con arroz: Sofreír cebolla, añadir pollo en trozos, leche de coco y curry. Servir con arroz blanco."
    
    print("\nIniciando prueba del Generador vLLM...\n")
    respuesta = generar_respuesta_final_vllm(ingredientes_prueba, receta_prueba)
    
    print("\n" + "="*50)
    print("RESPUESTA FINAL DEL AGENTE:")
    print("="*50)
    print(respuesta)