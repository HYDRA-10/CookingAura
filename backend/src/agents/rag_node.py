"""Búsqueda de recetas en Milvus: semántica (denso), léxica (BM25 disperso) o híbrida."""
import os
import re
import time

from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, WeightedRanker

from src.rag.embedder import embedder
from src.rag.cronometro import Cronometro
from src.llms.prompts import tipo_comida_con_articulo
from src.agents.coincidencias import separar_ingredientes, contar_coincidencias
from src.rag import metadatos

milvus_host = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_URI = f"http://{milvus_host}:19530"
COLLECTION_NAME = "recetas"

CAMPO_DENSO = "vector_denso"
CAMPO_DISPERSO = "vector_disperso"
CAMPOS_SALIDA = ["texto", "fuente"]
# Metadatos que se devuelven con cada resultado (solo si la colección los tiene)
CAMPOS_META_SALIDA = ["nombre", "tipo_comida", "categoria", "tiempo_min", "dificultad", "dieta", "alergenos",
                      "origen_meta"]

MODOS = {"hibrida", "semantica", "lexica"}
RANKERS = {"rrf", "ponderado"}

# Valores por defecto para el nodo busqueda_rag (ajustables desde el docker-compose / .env)
RAG_MODO = os.getenv("RAG_MODO", "hibrida")
RAG_RANKER = os.getenv("RAG_RANKER", "rrf")
RAG_PESO_DENSO = float(os.getenv("RAG_PESO_DENSO", "0.7"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_RRF_K = int(os.getenv("RAG_RRF_K", "60"))
# Cada sub-búsqueda trae más candidatos de los que se devuelven, para que la fusión tenga margen
FACTOR_CANDIDATOS = 3

_client: MilvusClient | None = None
_coleccion_cargada = False


def get_client() -> MilvusClient:
    """Cliente de Milvus reutilizado entre peticiones."""
    global _client
    if _client is None:
        _client = MilvusClient(uri=MILVUS_URI)
    return _client


def _preparar_coleccion(client: MilvusClient) -> bool:
    """Comprueba que la colección existe y la carga en memoria (solo la primera vez)."""
    global _coleccion_cargada
    if not client.has_collection(COLLECTION_NAME):
        _coleccion_cargada = False
        return False
    if not _coleccion_cargada:
        client.load_collection(COLLECTION_NAME)
        _coleccion_cargada = True
    return True


_metadatos_cache = {"valor": None, "hasta": 0.0}


def tiene_metadatos(client: MilvusClient) -> bool:
    """True si la colección tiene los campos de metadatos (esquema nuevo). Se revisa cada minuto,
    por si se recrea la colección con el backend en marcha."""
    if time.time() < _metadatos_cache["hasta"]:
        return _metadatos_cache["valor"]
    try:
        campos = {f["name"] for f in client.describe_collection(COLLECTION_NAME)["fields"]}
        valor = "tipo_comida" in campos
    except Exception:
        valor = False
    _metadatos_cache.update(valor=valor, hasta=time.time() + 60)
    return valor


def _param_denso() -> dict:
    return {"metric_type": "COSINE", "params": {"ef": 64}}


def _param_disperso() -> dict:
    # drop_ratio_search descarta los términos de menor peso de la consulta: más rápido, casi sin pérdida
    return {"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}}


def _crear_ranker(ranker: str, peso_denso: float, rrf_k: int):
    if ranker == "ponderado":
        return WeightedRanker(peso_denso, 1.0 - peso_denso)
    return RRFRanker(rrf_k)


def buscar_recetas(consulta_densa: str,
                   consulta_dispersa: str | None = None,
                   modo: str = RAG_MODO,
                   top_k: int = RAG_TOP_K,
                   ranker: str = RAG_RANKER,
                   peso_denso: float = RAG_PESO_DENSO,
                   rrf_k: int = RAG_RRF_K,
                   excluir_ids: list[int] | None = None,
                   filtro: str = "") -> list[dict]:
    """
    Busca recetas en Milvus y devuelve [{"texto", "fuente", "score"}, ...].

    - consulta_densa: frase natural para el embedding (bge-m3).
    - consulta_dispersa: palabras clave para BM25 (por defecto, la misma consulta).
    - modo: "hibrida" | "semantica" | "lexica".
    - ranker (solo híbrida): "rrf" fusiona por posición; "ponderado" mezcla puntuaciones con peso_denso.
    - excluir_ids: fragmentos ya propuestos al usuario que no deben volver a salir.
    - filtro: expresión de Milvus sobre los metadatos (ver buscar_con_criterios).
    """
    if modo not in MODOS:
        raise ValueError(f"Modo de búsqueda no válido: {modo}")
    if ranker not in RANKERS:
        raise ValueError(f"Ranker no válido: {ranker}")

    consulta_densa = (consulta_densa or "").strip()
    consulta_dispersa = (consulta_dispersa or consulta_densa).strip()
    if not consulta_densa:
        return []

    peso_denso = min(max(float(peso_denso), 0.0), 1.0)
    top_k = max(1, int(top_k))

    # Con un peso extremo, la híbrida ponderada equivale a una sola búsqueda: nos ahorramos la otra
    if modo == "hibrida" and ranker == "ponderado":
        if peso_denso >= 1.0:
            modo = "semantica"
        elif peso_denso <= 0.0:
            modo = "lexica"

    crono = Cronometro("RAG búsqueda")
    client = get_client()
    with crono.fase("colección"):
        disponible = _preparar_coleccion(client)
    if not disponible:
        crono.imprimir("colección inexistente")
        return []

    partes = [f"({filtro})"] if filtro else []
    if excluir_ids:
        partes.append(f"id not in {[int(i) for i in excluir_ids]}")
    filtro = " and ".join(partes)
    campos_salida = CAMPOS_SALIDA + (CAMPOS_META_SALIDA if tiene_metadatos(client) else [])

    # El embedding se calcula aparte para medirlo (la 1ª vez incluye la carga de bge-m3)
    vector = None
    if modo in ("semantica", "hibrida"):
        etiqueta = "embedding" if embedder.model is not None else "embedding+carga modelo"
        with crono.fase(etiqueta):
            vector = embedder.get_embedding(consulta_densa)

    with crono.fase(f"milvus {modo}"):
        if modo == "semantica":
            resultados = client.search(
                collection_name=COLLECTION_NAME,
                data=[vector],
                anns_field=CAMPO_DENSO,
                search_params=_param_denso(),
                limit=top_k,
                filter=filtro,
                output_fields=campos_salida,
            )
        elif modo == "lexica":
            # Para BM25 se envía el texto tal cual: Milvus lo tokeniza con el analizador del campo
            resultados = client.search(
                collection_name=COLLECTION_NAME,
                data=[consulta_dispersa],
                anns_field=CAMPO_DISPERSO,
                search_params=_param_disperso(),
                limit=top_k,
                filter=filtro,
                output_fields=campos_salida,
            )
        else:
            candidatos = top_k * FACTOR_CANDIDATOS
            extra = {"expr": filtro} if filtro else {}
            peticiones = [
                AnnSearchRequest(data=[vector], anns_field=CAMPO_DENSO,
                                 param=_param_denso(), limit=candidatos, **extra),
                AnnSearchRequest(data=[consulta_dispersa], anns_field=CAMPO_DISPERSO,
                                 param=_param_disperso(), limit=candidatos, **extra),
            ]
            resultados = client.hybrid_search(
                collection_name=COLLECTION_NAME,
                reqs=peticiones,
                ranker=_crear_ranker(ranker, peso_denso, rrf_k),
                limit=top_k,
                output_fields=campos_salida,
            )

    hits = resultados[0] if resultados else []
    salida = [
        {
            "id": hit["id"],
            "texto": hit["entity"].get("texto", ""),
            "fuente": hit["entity"].get("fuente", ""),
            "score": round(float(hit["distance"]), 4),
            "meta": {c: hit["entity"].get(c) for c in CAMPOS_META_SALIDA if c in hit["entity"]},
        }
        for hit in hits
    ]
    detalle_ranker = f"/{ranker}" if modo == "hibrida" else ""
    crono.imprimir(f"{modo}{detalle_ranker} · top {top_k} · {len(salida)} resultados"
                   + (f" · {len(excluir_ids)} excluidos" if excluir_ids else "")
                   + (" · con filtro de metadatos" if partes and (len(partes) > 1 or not excluir_ids) else ""))
    return salida


def buscar_con_criterios(consulta_densa: str, consulta_dispersa: str | None = None,
                         top_k: int = RAG_TOP_K, excluir_ids: list[int] | None = None,
                         tipo_comida: str | None = None, alergias: list[str] | None = None,
                         dieta: str | None = None, tiempo_max: int | None = None,
                         fuente: str | None = None, **opciones) -> list[dict]:
    """Búsqueda con filtros por metadatos, en dos pasos:
    1. Las recetas que CUMPLEN todos los criterios (un desayuno nunca devuelve una comida).
    2. Si faltan resultados, las que NO TIENEN el dato (documentos sin metadatos), nunca las que lo incumplen.
    Cada resultado lleva "cumple": True (paso 1) o False (paso 2).

    Con una colección antigua, sin metadatos, se filtra el tipo de comida por las líneas
    "Tipo de comida: ..." del texto (recetarios) y se ignoran los demás criterios."""
    lista = metadatos.criterios(tipo_comida, alergias, dieta, tiempo_max, fuente)
    if not lista:
        resultados = buscar_recetas(consulta_densa, consulta_dispersa, top_k=top_k, excluir_ids=excluir_ids,
                                    **opciones)
        for r in resultados:
            r["cumple"] = True
        return resultados

    client = get_client()
    if _preparar_coleccion(client) and tiene_metadatos(client):
        estricto, flexible = metadatos.filtros_en_dos_pasos(lista)
        resultados = buscar_recetas(consulta_densa, consulta_dispersa, top_k=top_k, excluir_ids=excluir_ids,
                                    filtro=estricto, **opciones)
        for r in resultados:
            r["cumple"] = True
        extra = []
        if flexible and len(resultados) < top_k:
            vistos = list(excluir_ids or []) + [r["id"] for r in resultados]
            extra = buscar_recetas(consulta_densa, consulta_dispersa, top_k=top_k - len(resultados),
                                   excluir_ids=vistos, filtro=flexible, **opciones)
            for r in extra:
                r["cumple"] = False
        print(f"   Filtro de metadatos: {len(resultados)} cumplen los criterios, {len(extra)} sin ese dato "
              f"({estricto})", flush=True)
        return resultados + extra

    # Colección sin metadatos: se buscan más candidatos y se filtra por el texto
    candidatos = buscar_recetas(consulta_densa, consulta_dispersa, top_k=top_k * 2, excluir_ids=excluir_ids,
                                **opciones)
    filtrados = [r for r in candidatos
                 if metadatos.cumple_tipo_en_texto(r["texto"], tipo_comida)
                 and (not fuente or r.get("fuente") == fuente)]
    for r in filtrados:
        r["cumple"] = True
    if len(filtrados) < len(candidatos):
        print(f"   Sin metadatos en la colección: {len(candidatos) - len(filtrados)} descartados por el texto "
              f"(recrea la colección para filtrar en Milvus)", flush=True)
    return filtrados[:top_k]


def limpiar_ingredientes(texto: str) -> str:
    """'Ingredientes disponibles...\\n- Huevos (6 unidades)' -> 'Huevos, ...'.
    Las cantidades y cabeceras solo meten ruido en el embedding y en BM25."""
    if not texto:
        return ""
    texto = re.sub(r"Ingredientes disponibles en la despensa:\s*", "", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\([^)]*\)", "", texto)
    partes = [p.strip(" -•*\t") for p in re.split(r"[,;\n]+", texto)]
    return ", ".join(p for p in partes if p)


def buscar_recetas_en_milvus(ingredientes_disponibles: str, tipo_comida: str | None = None,
                             top_k: int | None = None,
                             excluir_ids: list[int] | None = None,
                             alergias: list[str] | None = None,
                             dieta: str | None = None) -> list[dict]:
    """Búsqueda para el nodo busqueda_rag.
    Devuelve los fragmentos con su número de ingredientes en común, el mejor primero; [] si no hay nada."""
    ingredientes = limpiar_ingredientes(ingredientes_disponibles)
    if not ingredientes:
        return []

    momento = tipo_comida_con_articulo(tipo_comida)
    # Densa: frase natural (el embedding entiende el contexto "para la cena")
    consulta_densa = f"Receta para {momento} con {ingredientes}" if momento else f"Receta con {ingredientes}"
    # Dispersa: solo palabras con contenido; "receta", "para", "con" no aportan nada a BM25
    consulta_dispersa = " ".join(filter(None, [tipo_comida, ingredientes.replace(",", " ")]))

    print(f"Buscando recetas en Milvus ({RAG_MODO}/{RAG_RANKER}) | densa: '{consulta_densa}' | BM25: '{consulta_dispersa}'")
    resultados = buscar_con_criterios(consulta_densa, consulta_dispersa, top_k=top_k or RAG_TOP_K,
                                      excluir_ids=excluir_ids, tipo_comida=tipo_comida, alergias=alergias,
                                      dieta=dieta)

    crono = Cronometro("RAG coincidencias")
    with crono.fase("contar ingredientes"):
        lista = separar_ingredientes(ingredientes)
        for r in resultados:
            r["coincidencias"] = contar_coincidencias(lista, r["texto"])
        # Primero las que cumplen los filtros; dentro, más ingredientes en común primero
        # (a igualdad se respeta el orden de Milvus: sorted es estable)
        ordenados = sorted(resultados, key=lambda r: (r.get("cumple", True), r["coincidencias"]), reverse=True)
    crono.imprimir(f"{len(lista)} ingredientes × {len(resultados)} fragmentos")
    return ordenados


def obtener_receta_por_id(id_fragmento: int) -> dict | None:
    """Recupera un fragmento concreto (la idea local elegida por el usuario)."""
    crono = Cronometro("RAG receta por id")
    client = get_client()
    with crono.fase("colección"):
        disponible = _preparar_coleccion(client)
    if not disponible:
        return None
    with crono.fase("milvus get"):
        filas = client.get(collection_name=COLLECTION_NAME, ids=[int(id_fragmento)], output_fields=CAMPOS_SALIDA)
    crono.imprimir(f"id {id_fragmento} · {'encontrado' if filas else 'no encontrado'}")
    return filas[0] if filas else None