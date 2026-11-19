import sys
import os

# 1. Calculamos la ruta absoluta hacia la carpeta "backend" y la añadimos al path
ruta_backend = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.append(ruta_backend)

from src.llms.ollama_client import evaluar_recetas

if __name__ == "__main__":
    # 1. Prueba donde DEBERÍA decir SI (Los ingredientes encajan bastante bien)
    ingredientes_ok = "- pollo (500g)\n- arroz (1kg)\n- cebolla (2 unidades)"
    recetas_ok = "Pollo con arroz y cebolla frita. / Sopa de pollo."
    
    print("\n--- PRUEBA 1: Ingredientes compatibles ---")
    decision_1 = evaluar_recetas(ingredientes_ok, recetas_ok)
    print(f"Decisión: {decision_1}")
    
    # 2. Prueba donde DEBERÍA decir NO (Faltan los ingredientes principales)
    ingredientes_mal = "- pimientos (3 unidades)\n- salsa de soja (1 botella)"
    recetas_mal = "Merluza al horno con patatas. / Chuletón de ternera."
    
    print("\n--- PRUEBA 2: Ingredientes incompatibles ---")
    decision_2 = evaluar_recetas(ingredientes_mal, recetas_mal)
    print(f"Decisión: {decision_2}")