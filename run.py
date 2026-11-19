import torch
import subprocess

def iniciar_entorno():
    print("🔍 Comprobando capacidades de la GPU local...")
    arrancar_vllm = False

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        capacidad = major + (minor / 10.0)
        print(f"✅ GPU NVIDIA detectada. Compute Capability: {capacidad}")
        
        if capacidad >= 7.0:
            arrancar_vllm = True
        else:
            print("⚠️ La capacidad es inferior a 7.0. vLLM fallaría al arrancar.")
    else:
        print("❌ No se detectó una GPU compatible con CUDA.")

    # Ruta exacta al archivo de orquestación
    archivo_compose = "docker/docker-compose.yml"

    if arrancar_vllm:
        print("🚀 Levantando infraestructura COMPLETA (incluyendo vLLM)...")
        comando = ["docker-compose", "-f", archivo_compose, "--profile", "vllm-activo", "up", "-d", "--build"]
    else:
        print("🐢 Levantando infraestructura LIGERA (solo Ollama, omitiendo vLLM)...")
        comando = ["docker-compose", "-f", archivo_compose, "up", "-d", "--build"]

    print(f"Ejecutando: {' '.join(comando)}")
    subprocess.run(comando)

if __name__ == "__main__":
    iniciar_entorno()