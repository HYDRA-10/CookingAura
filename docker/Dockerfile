# 1. Usa una imagen oficial de Python ligera
FROM python:3.12-slim

# 2. Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Instala dependencias del sistema necesarias para compilar algunas librerías
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copia el archivo de dependencias primero para aprovechar la caché de Docker
COPY requirements.txt .

# 5. Instala las dependencias de Python (LangGraph, LangChain, PyMilvus, PyTorch, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copia la carpeta del código fuente
COPY backend/src/ /app/src/

# 7. Comando por defecto al levantar el contenedor
CMD ["python", "-m", "src.main"]