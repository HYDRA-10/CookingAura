import os
import json
from google import genai

RUTA_JSON = "config/gcp-key.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = RUTA_JSON

print("Auditando el archivo de credenciales...")

try:
    # 1. Leer el JSON directamente
    with open(RUTA_JSON, 'r') as f:
        datos_json = json.load(f)
        
    proyecto_real = datos_json.get("project_id")
    correo_sa = datos_json.get("client_email")
    
    print("-" * 50)
    print(f"🔍 Proyecto detectado en el JSON: {proyecto_real}")
    print(f"📧 Correo de la cuenta: {correo_sa}")
    print("-" * 50)

    # 2. Comprobar si coincide con el proyecto que estábamos usando
    if proyecto_real != "sturdy-cable-512913-e3":
        print("⚠️ ATENCIÓN: El JSON pertenece a un proyecto diferente al esperado.")

    # 3. Probar la conexión usando los datos reales del JSON
    print(f"\nConectando a {proyecto_real} en global...")
    client = genai.Client(vertexai=True, project=proyecto_real, location="global")
    
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Responde solo con la palabra OK."
    )
    print(f"✅ ÉXITO (Respondió: {response.text.strip()})")

except FileNotFoundError:
    print("❌ ERROR: No se encuentra el archivo JSON en la ruta especificada.")
except Exception as e:
    print(f"\n❌ ERROR DE GOOGLE: {str(e)}")