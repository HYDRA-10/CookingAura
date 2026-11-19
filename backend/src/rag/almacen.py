"""Almacén de los ficheros originales subidos (PDF, Word, Excel, CSV).

Se guardan en data/raw_recipe/ (en Docker: /app/data/raw_recipes, montado en la carpeta data del host),
con el mismo nombre con el que se subieron. Milvus guarda los fragmentos y vectores; aquí queda
el original para poder descargarlo o volver a indexarlo si se recrea la colección.
"""
from pathlib import Path

CARPETA = Path(__file__).resolve().parents[2] / "data" / "raw_recipes"


def ruta(nombre: str) -> Path:
    """Ruta segura dentro de la carpeta: se ignoran directorios y '..' que vengan en el nombre."""
    limpio = Path(nombre.replace("\\", "/")).name.strip()
    if not limpio or limpio in (".", ".."):
        raise ValueError(f"Nombre de archivo no válido: {nombre!r}")
    return CARPETA / limpio


def guardar(nombre: str, contenido: bytes) -> Path:
    """Guarda (o sustituye) el original. Se escribe en un temporal y se renombra: nunca queda a medias."""
    destino = ruta(nombre)
    CARPETA.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_name(destino.name + ".subiendo")
    temporal.write_bytes(contenido)
    temporal.replace(destino)
    return destino


def existe(nombre: str) -> bool:
    try:
        return ruta(nombre).is_file()
    except ValueError:
        return False


def borrar(nombre: str) -> bool:
    try:
        destino = ruta(nombre)
    except ValueError:
        return False
    if destino.is_file():
        destino.unlink()
        return True
    return False


def listar() -> list[Path]:
    if not CARPETA.is_dir():
        return []
    return sorted(p for p in CARPETA.iterdir() if p.is_file() and not p.name.endswith(".subiendo"))