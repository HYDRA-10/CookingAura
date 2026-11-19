import sys
import os

# 1. Calculamos la ruta absoluta hacia la carpeta "backend" y la añadimos al path
ruta_backend = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.append(ruta_backend)

from src.agents.rag_node import buscar_recetas_en_milvus

if __name__ == "__main__":
    # Simulamos lo que el agente MCP extraería de la base de datos SQLite
    ingredientes_prueba = "pollo, arroz, cebolla"
    
    print("Iniciando prueba de búsqueda RAG...")
    resultado_texto = buscar_recetas_en_milvus(ingredientes_prueba)
    
    print("\n" + "="*50)
    print("RESULTADO DE LA BÚSQUEDA VECTORIAL:")
    print("="*50)
    print(resultado_texto)