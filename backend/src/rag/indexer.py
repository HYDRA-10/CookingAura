"""Creación de la colección híbrida (densa + BM25) e indexación de documentos troceados."""
import json
import re
import sys
from collections import Counter

from pymilvus import MilvusClient, DataType, Function, FunctionType

from src.rag.embedder import embedder
from src.rag.cronometro import Cronometro
from src.agents.rag_node import get_client, COLLECTION_NAME, CAMPO_DENSO, CAMPO_DISPERSO, _preparar_coleccion

DIMENSION = 1024           # Dimensión densa de BAAI/bge-m3
MAX_TEXTO = 4000           # Límite del VARCHAR (en bytes UTF-8)
MAX_FUENTE = 512
TAM_TROZO = 1500           # Caracteres por fragmento (holgado frente a MAX_TEXTO aunque haya tildes)
SOLAPE = 200               # Solape al cortar líneas muy largas

CAMPOS_REQUERIDOS = {"id", "texto", "fuente", CAMPO_DENSO, CAMPO_DISPERSO}

# Analizador en español: minúsculas, sin stopwords, con stemming ("huevos" == "huevo") y sin tildes
ANALIZADOR_ES = {
    "tokenizer": "standard",
    "filter": [
        "lowercase",
        {"type": "stop", "stop_words": ["_spanish_"]},
        {"type": "stemmer", "language": "spanish"},
        "asciifolding",
    ],
}


def init_milvus(recrear: bool = False) -> MilvusClient:
    """Crea la colección si no existe. Con recrear=True la borra antes (¡se pierden los datos!)."""
    client = get_client()

    if client.has_collection(collection_name=COLLECTION_NAME):
        if not recrear:
            return client
        print(f"🗑️  Borrando colección '{COLLECTION_NAME}'...")
        client.drop_collection(collection_name=COLLECTION_NAME)

    print("Creando colección híbrida (semántica + BM25)...")
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="texto", datatype=DataType.VARCHAR, max_length=MAX_TEXTO,
                     enable_analyzer=True, analyzer_params=ANALIZADOR_ES)
    schema.add_field(field_name="fuente", datatype=DataType.VARCHAR, max_length=MAX_FUENTE)
    schema.add_field(field_name=CAMPO_DENSO, datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    schema.add_field(field_name=CAMPO_DISPERSO, datatype=DataType.SPARSE_FLOAT_VECTOR)

    # Milvus genera el vector disperso BM25 a partir de 'texto' al insertar y al buscar
    schema.add_function(Function(
        name="generador_bm25",
        function_type=FunctionType.BM25,
        input_field_names=["texto"],
        output_field_names=[CAMPO_DISPERSO],
    ))

    index_params = client.prepare_index_params()
    index_params.add_index(field_name=CAMPO_DENSO, metric_type="COSINE", index_type="HNSW",
                           params={"M": 16, "efConstruction": 200})
    index_params.add_index(field_name=CAMPO_DISPERSO, metric_type="BM25",
                           index_type="SPARSE_INVERTED_INDEX")

    client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)
    print(f"✅ Colección híbrida '{COLLECTION_NAME}' creada.")
    return client


def asegurar_coleccion() -> MilvusClient:
    """Crea la colección si falta y avisa si existe con un esquema antiguo."""
    client = init_milvus(recrear=False)
    campos = {f["name"] for f in client.describe_collection(COLLECTION_NAME)["fields"]}
    faltan = CAMPOS_REQUERIDOS - campos
    if faltan:
        raise RuntimeError(
            f"La colección '{COLLECTION_NAME}' tiene un esquema antiguo (faltan: {', '.join(sorted(faltan))}). "
            f"Recréala con: python -m src.rag.indexer --recrear  (borra los documentos indexados)."
        )
    return client


def trocear_texto(texto: str, tam: int = TAM_TROZO, solape: int = SOLAPE) -> list[str]:
    """Agrupa líneas en fragmentos de ~tam caracteres sin partir líneas (salvo las enormes).
    Respeta las secciones ('## Título'): un título nuevo empieza fragmento si el actual ya va por la
    mitad, y si una sección se parte, el siguiente fragmento empieza repitiendo su título."""
    lineas = [l.strip() for l in re.split(r"\n+", texto) if l.strip()]
    trozos, actual, seccion = [], "", None

    def cerrar():
        nonlocal actual
        # Un fragmento que solo contiene títulos no aporta nada por sí mismo
        if actual and any(not l.startswith("## ") for l in actual.splitlines()):
            trozos.append(actual)
        actual = ""

    def empezar() -> str:
        return f"{seccion} (continuación)" if seccion else ""

    for linea in lineas:
        if linea.startswith("## "):
            # Títulos seguidos (DOCUMENTO > SECCIÓN) se mantienen juntos con lo que viene después
            if len(actual) >= tam // 2 and not actual.splitlines()[-1].startswith("## "):
                cerrar()
            seccion = linea
            actual = f"{actual}\n{linea}" if actual else linea
            continue
        while len(linea) > tam:                         # línea gigante: ventanas con solape
            cerrar()
            cabecera = empezar()
            hueco = tam - len(cabecera) - 1
            trozos.append(f"{cabecera}\n{linea[:hueco]}" if cabecera else linea[:tam])
            linea = linea[hueco - solape:] if cabecera else linea[tam - solape:]
        if len(actual) + len(linea) + 1 > tam:
            cerrar()
            actual = empezar()
        actual = f"{actual}\n{linea}" if actual else linea
    cerrar()
    return trozos


def trocear_receta(receta: str, tam: int = TAM_TROZO) -> list[str]:
    """Una receta de un recetario tabular: un solo fragmento si cabe; si no, cada trozo repite el nombre."""
    if len(receta) <= tam:
        return [receta]
    titulo = receta.splitlines()[0][:100]
    trozos = trocear_texto(receta, tam=tam - len(titulo) - 20)
    return [trozos[0]] + [f"{titulo} (continuación)\n{t}" for t in trozos[1:]]


def _filtro_fuente(fuente: str) -> str:
    # json.dumps escapa comillas y barras: el nombre del archivo no puede romper la expresión
    return f"fuente == {json.dumps(fuente[:MAX_FUENTE], ensure_ascii=False)}"


def borrar_documento(fuente: str) -> int:
    """Elimina todos los fragmentos de un documento. Devuelve cuántos había."""
    client = get_client()
    if not client.has_collection(COLLECTION_NAME):
        return 0
    filtro = _filtro_fuente(fuente)
    existentes = client.query(collection_name=COLLECTION_NAME, filter=filtro,
                              output_fields=["id"], limit=16384, consistency_level="Strong")
    if not existentes:
        return 0
    client.delete(collection_name=COLLECTION_NAME, filter=filtro)
    return len(existentes)


def listar_documentos() -> list[dict]:
    """Documentos indexados en Milvus con su número de fragmentos: [{"nombre", "fragmentos"}]."""
    crono = Cronometro("RAG listar documentos")
    client = get_client()
    with crono.fase("colección"):
        if not _preparar_coleccion(client):
            return []

    conteo = Counter()
    with crono.fase("milvus query"):
        try:
            # Se recorre por lotes: funciona con colecciones de cualquier tamaño
            iterador = client.query_iterator(collection_name=COLLECTION_NAME, batch_size=1000, filter="",
                                             output_fields=["fuente"], consistency_level="Strong")
            while True:
                lote = iterador.next()
                if not lote:
                    iterador.close()
                    break
                conteo.update(f.get("fuente") or "(sin nombre)" for f in lote)
        except AttributeError:   # pymilvus sin query_iterator en MilvusClient
            filas = client.query(collection_name=COLLECTION_NAME, filter="", output_fields=["fuente"],
                                 limit=16384, consistency_level="Strong")
            conteo.update(f.get("fuente") or "(sin nombre)" for f in filas)

    crono.imprimir(f"{len(conteo)} documentos · {sum(conteo.values())} fragmentos")
    return [{"nombre": n, "fragmentos": c} for n, c in sorted(conteo.items(), key=lambda x: x[0].lower())]


def indexar_documento(texto: str, fuente: str, recetas: list[str] | None = None,
                      reemplazar: bool = True, crono: Cronometro | None = None) -> int:
    """Trocea, vectoriza (denso) e inserta un documento. El vector BM25 lo genera Milvus.
    'texto' se trocea por líneas; cada elemento de 'recetas' (una fila de un recetario) va por separado.
    Con reemplazar=True, si ya había un documento con el mismo nombre se sustituye (no se duplica)."""
    crono = crono or Cronometro("RAG indexación")

    with crono.fase("troceado"):
        trozos = trocear_texto(texto) if texto and texto.strip() else []
        for receta in recetas or []:
            if receta.strip():
                trozos.extend(trocear_receta(receta.strip()))
    if not trozos:
        crono.imprimir(f"'{fuente}' sin contenido")
        return 0

    with crono.fase("colección"):
        client = asegurar_coleccion()

    if reemplazar:
        with crono.fase("borrar versión anterior"):
            borrados = borrar_documento(fuente)
        if borrados:
            print(f"   ♻️  '{fuente}' ya existía: se sustituyen sus {borrados} fragmentos", flush=True)

    etiqueta = "embeddings" if embedder.model is not None else "embeddings+carga modelo"
    with crono.fase(etiqueta):
        vectores = embedder.get_embeddings(trozos)

    with crono.fase("inserción milvus"):
        datos = [{"texto": t, "fuente": fuente[:MAX_FUENTE], CAMPO_DENSO: v} for t, v in zip(trozos, vectores)]
        client.insert(collection_name=COLLECTION_NAME, data=datos)

    crono.imprimir(f"'{fuente}' · {len(trozos)} fragmentos · {sum(len(t) for t in trozos)} caracteres")
    return len(trozos)


def reindexar_originales() -> None:
    """Vuelve a indexar todos los originales guardados en data/documentos (p. ej. tras --recrear)."""
    from src.rag import almacen
    from src.rag.extractores import extraer_documento

    archivos = almacen.listar()
    if not archivos:
        print(f"No hay originales guardados en {almacen.CARPETA}")
        return
    total = 0
    for archivo in archivos:
        try:
            extraccion = extraer_documento(archivo.name, archivo.read_bytes())
            total += indexar_documento(extraccion.texto, archivo.name, recetas=extraccion.recetas)
        except Exception as e:
            print(f"⚠️  '{archivo.name}' no se pudo reindexar: {e}")
    print(f"✅ {len(archivos)} documentos reindexados ({total} fragmentos)")


if __name__ == "__main__":
    # python -m src.rag.indexer --recrear              -> colección nueva y vacía
    # python -m src.rag.indexer --recrear --reindexar  -> colección nueva con todos los originales guardados
    init_milvus(recrear="--recrear" in sys.argv)
    if "--reindexar" in sys.argv:
        reindexar_originales()