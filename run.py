"""Arranque de CookingAura con un solo comando.

    python run.py                  # comprueba, construye y levanta todo
    python run.py --sin-build      # levanta sin reconstruir las imágenes (más rápido)
    python run.py --sin-vllm       # no levanta vLLM aunque haya GPU compatible
    python run.py --sin-modelos    # no descarga los modelos de Ollama que falten
    python run.py --attu           # levanta también Attu, la interfaz web de Milvus (http://localhost:3001)
    python run.py --parar          # para todos los contenedores

Ollama puede ir en un contenedor (OLLAMA_URL=http://ollama-server:11434) o instalado en el equipo
(OLLAMA_URL=http://host.docker.internal:11434): en ese caso no se levanta el contenedor ollama-server.

Solo usa la biblioteca estándar de Python: no hace falta instalar nada en el equipo (ni PyTorch).
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import json
import urllib.request
from pathlib import Path

# La consola de Windows puede no admitir emojis si la salida se redirige
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAIZ = Path(__file__).resolve().parent
COMPOSE = RAIZ / "docker" / "docker-compose.yml"
ENV = RAIZ / ".env"
ENV_EJEMPLO = RAIZ / ".env.example"

OLLAMA_LOCAL = "http://127.0.0.1:11434"
HOSTS_LOCALES = ("host.docker.internal", "localhost", "127.0.0.1")

CAPACIDAD_MINIMA_VLLM = 7.0      # vLLM necesita compute capability 7.0 o superior
VRAM_MINIMA_VLLM_MB = 12000      # con menos VRAM no compensa compartir la GPU con el resto


# ------------------------------------------------------------------ utilidades
def ejecutar(comando: list[str], capturar: bool = False, comprobar: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(comando, capture_output=capturar, text=True, encoding="utf-8",
                          errors="replace", check=comprobar)


def comando_compose() -> list[str]:
    """'docker compose' (v2, el actual); si no está, el antiguo 'docker-compose'."""
    if ejecutar(["docker", "compose", "version"], capturar=True).returncode == 0:
        return ["docker", "compose"]
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    sys.exit("❌ No se encuentra Docker Compose. Instala o actualiza Docker Desktop.")


def leer_env() -> dict[str, str]:
    """Lee el .env (KEY=VALUE, sin comentarios) sin dependencias externas."""
    valores = {}
    for linea in ENV.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            clave, valor = linea.split("=", 1)
            valores[clave.strip()] = valor.split(" #")[0].strip().strip('"').strip("'")
    return valores


# ------------------------------------------------------------- comprobaciones
def comprobar_docker():
    if not shutil.which("docker"):
        sys.exit("❌ Docker no está instalado o no está en el PATH.")
    if ejecutar(["docker", "info"], capturar=True).returncode != 0:
        sys.exit("❌ Docker no está en marcha. Abre Docker Desktop y vuelve a intentarlo.")
    print("✅ Docker en marcha")


def comprobar_ficheros():
    if not COMPOSE.exists():
        sys.exit(f"❌ No se encuentra {COMPOSE}")
    if not ENV.exists():
        if ENV_EJEMPLO.exists():
            shutil.copy(ENV_EJEMPLO, ENV)
            print("⚠️  No había .env: se ha creado a partir de .env.example. Revisa los valores marcados con ⚠️.")
        else:
            sys.exit("❌ Falta el fichero .env (y no hay .env.example para crearlo).")
    else:
        print("✅ .env encontrado")

    if not (RAIZ / "config" / "gcp-key.json").exists():
        print("ℹ️  No hay config/gcp-key.json: la búsqueda web con Gemini no funcionará "
              "(el resto de la app sí; las recetas se generarán sin internet).")

    cache_hf = Path(os.environ.get("USERPROFILE") or Path.home()) / ".cache" / "huggingface" / "hub"
    if not (cache_hf / "models--BAAI--bge-m3").exists():
        print("⚠️  No se encuentra el modelo de embeddings bge-m3 en la caché de HuggingFace.\n"
              "    Descárgalo una vez con:  pip install -U huggingface_hub  y  huggingface-cli download BAAI/bge-m3")


def detectar_gpu_para_vllm() -> bool:
    """True si hay una GPU NVIDIA con capacidad y memoria suficientes y Docker puede usarla."""
    print("🔍 Comprobando la GPU...")
    if not shutil.which("nvidia-smi"):
        print("   Sin GPU NVIDIA (no hay nvidia-smi): se usará Ollama en CPU.")
        return False
    r = ejecutar(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader,nounits"],
                 capturar=True)
    if r.returncode != 0 or not r.stdout.strip():
        print("   nvidia-smi no responde: se usará Ollama.")
        return False
    nombre, capacidad, memoria = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
    print(f"   {nombre} · compute capability {capacidad} · {memoria} MB")
    if float(capacidad) < CAPACIDAD_MINIMA_VLLM:
        print(f"   Capacidad inferior a {CAPACIDAD_MINIMA_VLLM}: vLLM no funcionaría. Se usará Ollama.")
        return False
    if int(float(memoria)) < VRAM_MINIMA_VLLM_MB:
        print(f"   Menos de {VRAM_MINIMA_VLLM_MB // 1000} GB de VRAM: no compensa vLLM. Se usará Ollama.")
        return False
    # La GPU existe, pero Docker también tiene que poder usarla (NVIDIA Container Toolkit / WSL2)
    if ejecutar(["docker", "run", "--rm", "--gpus", "all", "busybox", "true"], capturar=True).returncode != 0:
        print("   Docker no tiene acceso a la GPU (falta NVIDIA Container Toolkit): se usará Ollama.")
        return False
    print("   ✅ GPU válida para vLLM")
    return True


# ------------------------------------------------------------------ arranque
def ollama_en_equipo(env: dict[str, str]) -> bool:
    """True si el .env apunta a un Ollama instalado en el equipo (fuera de Docker)."""
    return any(h in env.get("OLLAMA_URL", "") for h in HOSTS_LOCALES)


def _api_ollama(ruta: str, datos: dict | None = None, timeout: float = 10):
    peticion = urllib.request.Request(f"{OLLAMA_LOCAL}{ruta}", method="POST" if datos else "GET",
                                      data=json.dumps(datos).encode() if datos else None,
                                      headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(peticion, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def comprobar_ollama_equipo():
    try:
        _api_ollama("/api/tags")
        print("✅ Ollama del equipo en marcha (fuera de Docker)")
    except Exception:
        sys.exit("❌ El .env usa el Ollama del equipo (host.docker.internal), pero no responde en el puerto 11434.\n"
                 "   Ábrelo desde el menú Inicio (o ejecuta 'ollama serve') y vuelve a lanzar run.py.")


def _modelos_env(env: dict[str, str]) -> set[str]:
    return {env.get(v) for v in ("OLLAMA_MODELO_EVALUADOR", "OLLAMA_MODELO_GENERADOR", "AGENTE_MODELO")} - {None, ""}


def _falta(modelo: str, instalados: set[str]) -> bool:
    return modelo not in instalados and (modelo if ":" in modelo else f"{modelo}:latest") not in instalados


def descargar_modelos_equipo(env: dict[str, str]):
    """Descarga en el Ollama del equipo los modelos del .env que no tenga."""
    instalados = {m["name"] for m in _api_ollama("/api/tags").get("models", [])}
    for modelo in sorted(_modelos_env(env)):
        if not _falta(modelo, instalados):
            print(f"✅ Modelo de Ollama disponible: {modelo}")
            continue
        print(f"⬇️  Descargando el modelo {modelo} en el Ollama del equipo (solo la primera vez)...")
        try:
            _api_ollama("/api/pull", {"model": modelo, "stream": False}, timeout=3600)
        except Exception as e:
            print(f"⚠️  No se pudo descargar {modelo}: {e}. Prueba con:  ollama pull {modelo}")


def descargar_modelos_ollama(env: dict[str, str]):
    """Descarga en el contenedor ollama-server los modelos del .env que todavía no estén."""
    modelos = _modelos_env(env)
    if not modelos:
        return
    lista = ""
    for _ in range(30):                                   # Ollama puede tardar unos segundos en responder
        r = ejecutar(["docker", "exec", "ollama-server", "ollama", "list"], capturar=True)
        if r.returncode == 0:
            lista = r.stdout
            break
        time.sleep(2)
    instalados = {l.split()[0] for l in lista.splitlines()[1:] if l.strip()}
    for modelo in sorted(modelos):
        if not _falta(modelo, instalados):
            print(f"✅ Modelo de Ollama disponible: {modelo}")
            continue
        print(f"⬇️  Descargando el modelo de Ollama {modelo} (solo la primera vez)...")
        if ejecutar(["docker", "exec", "ollama-server", "ollama", "pull", modelo]).returncode != 0:
            print(f"⚠️  No se pudo descargar {modelo}: revisa el nombre en el .env.")


def esperar_backend(espera: int = 240) -> bool:
    print("⏳ Esperando a que el backend responda...")
    limite = time.time() + espera
    while time.time() < limite:
        try:
            with urllib.request.urlopen("http://localhost:8080/docs", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def main():
    parser = argparse.ArgumentParser(description="Arranca CookingAura con Docker Compose.")
    parser.add_argument("--sin-build", action="store_true", help="No reconstruir las imágenes")
    parser.add_argument("--sin-vllm", action="store_true", help="No levantar vLLM aunque haya GPU")
    parser.add_argument("--sin-modelos", action="store_true", help="No descargar los modelos de Ollama que falten")
    parser.add_argument("--attu", action="store_true", help="Levantar también Attu, la interfaz web de Milvus")
    parser.add_argument("--parar", action="store_true", help="Parar todos los contenedores")
    args = parser.parse_args()

    comprobar_docker()
    compose = comando_compose() + ["-f", str(COMPOSE)]

    if args.parar:
        print("🛑 Parando CookingAura...")
        sys.exit(ejecutar(compose + ["--profile", "vllm-activo", "--profile", "ollama-docker", "--profile", "attu",
                                     "stop"]).returncode)

    comprobar_ficheros()
    env = leer_env()
    nativo = ollama_en_equipo(env)
    if nativo:
        comprobar_ollama_equipo()
    con_vllm = not args.sin_vllm and detectar_gpu_para_vllm()

    perfiles = ((["--profile", "vllm-activo"] if con_vllm else []) + ([] if nativo else ["--profile", "ollama-docker"])
                + (["--profile", "attu"] if args.attu else []))
    comando = compose + perfiles + ["up", "-d"]
    if not args.sin_build:
        comando.append("--build")
    print(("🚀 Levantando todo, con vLLM" if con_vllm else "🐢 Levantando todo sin vLLM")
          + (" · Ollama del equipo" if nativo else " · Ollama en Docker") + f"\n   {' '.join(comando)}")
    if ejecutar(comando).returncode != 0:
        sys.exit("❌ Docker Compose ha fallado. Revisa el mensaje de arriba.")

    if not args.sin_modelos:
        descargar_modelos_equipo(env) if nativo else descargar_modelos_ollama(env)

    if esperar_backend():
        print("\n✅ CookingAura está en marcha:\n"
              "   App ............ http://localhost:8501\n"
              "   API (docs) ..... http://localhost:8080/docs\n"
              "   MinIO .......... http://localhost:9001\n"
              + ("   Attu (Milvus) .. http://localhost:3001\n" if args.attu else "") +
              "\n   Los modelos se siguen cargando en segundo plano: la primera respuesta puede tardar más.\n"
              "   Logs del backend:  docker logs -f cookingaura-app-backend\n"
              "   Para pararla:      python run.py --parar")
    else:
        print("\n⚠️  El backend no responde todavía. Revisa:  docker logs cookingaura-app-backend")


if __name__ == "__main__":
    main()