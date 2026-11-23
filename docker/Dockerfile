# 1. Usa una imagen oficial de Python ligera
FROM python:3.12-slim

# 2. Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Instala dependencias del sistema:
#    - build-essential: compilar algunas librerías de Python
#    - curl: healthchecks y descargas
#    - espeak-ng: voz sintética para las métricas de texto frente a voz (metricas.evaluar preparar-voz)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# 4. PyTorch solo para CPU: sin esta línea, pip instala la versión con CUDA y sus librerías
#    de NVIDIA (varios GB que sin GPU no se usan). Misma versión que en requirements.txt.
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

# 5. Copia el archivo de dependencias primero para aprovechar la caché de Docker
COPY requirements.txt .

# 6. Instala las dependencias de Python (LangGraph, LangChain, PyMilvus, PyTorch, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copia la carpeta del código fuente
COPY backend/src/ /app/src/

# 8. Comando por defecto al levantar el contenedor
CMD ["python", "-m", "src.main"]