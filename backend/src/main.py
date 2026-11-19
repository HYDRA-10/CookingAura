import src.config  # noqa: F401  (carga el .env antes que el resto de módulos)
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
import json
import time
from fastapi.concurrency import run_in_threadpool
import traceback
from pydantic import BaseModel
from typing import Optional, Literal
from .llms.transcriptor import transcribir_audio
import uvicorn
import tempfile
import os
import sqlite3
import io
from pymilvus import MilvusClient
from src.agents.rag_node import buscar_recetas, MODOS, RANKERS, COLLECTION_NAME, MILVUS_URI
from src.rag.indexer import indexar_documento, listar_documentos, borrar_documento
from src.rag.cronometro import Cronometro
from src.rag import almacen
from src.data import sqlitebd
from fastapi.responses import FileResponse
from datetime import datetime
from src.rag.extractores import extraer_documento


from .agents.graph import ejecutar_asistente, ejecutar_asistente_stream
from .agents.mcp_client import gestor_mcp
from .llms.ollama_client import precalentar_modelos, MODELO_EVALUADOR as OLLAMA_MODELO_EVALUADOR, MODELO_GENERADOR as OLLAMA_MODELO_GENERADOR
from src.config import AGENTE_MODELO, PRECALENTAR_MODELOS, PRECARGAR_EMBEDDINGS
from src.rag.embedder import embedder
import asyncio

# Ruta a la base de datos
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ingredients.db")

async def _precargar_modelos():
    """Embeddings y modelos de Ollama, uno detrás de otro (a la vez competirían por la RAM y la CPU)."""
    if PRECARGAR_EMBEDDINGS:
        try:
            # En un hilo: la carga de bge-m3 es bloqueante y no debe frenar las peticiones
            await asyncio.to_thread(embedder.precargar)
        except Exception as e:
            print(f"⚠️  No se pudo precargar el modelo de embeddings: {e}", flush=True)
    if PRECALENTAR_MODELOS:
        await precalentar_modelos([OLLAMA_MODELO_EVALUADOR, OLLAMA_MODELO_GENERADOR, AGENTE_MODELO])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Una sola sesión MCP para todo el backend: el servidor de herramientas arranca una vez
    await gestor_mcp.iniciar()
    # Modelos cargados en segundo plano: el backend ya responde mientras tanto
    precarga = asyncio.create_task(_precargar_modelos()) if (PRECALENTAR_MODELOS or PRECARGAR_EMBEDDINGS) else None
    yield
    if precarga and not precarga.done():
        precarga.cancel()
    await gestor_mcp.detener()


app = FastAPI(title="CookingAura API", lifespan=lifespan)

class Ingrediente(BaseModel):
    nombre: str
    cantidad: str
    categoria: str
    unidad: str = None  # Unidad opcional, por ejemplo: "gramos", "litros", etc.
    id: Optional[int] = None  # identificador en SQLite: distingue ingredientes con el mismo nombre

class IdeaPlato(BaseModel):
    id: str
    nombre: str
    descripcion: str = ""
    origen: Literal["local", "creativa"] = "creativa"
    ref_id: Optional[int] = None     # id del fragmento en Milvus (ideas de tus recetas)
    fuente: Optional[str] = None

class MensajeChat(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    mensaje: str
    intencion: Optional[str] = None
    ingredientes_db: Optional[str] = None
    ingredientes_manual: Optional[str] = None
    tipo_comida: Optional[str] = None  # "desayuno", "comida", "cena"... (o "Desayunar", "Cenar"...)
    # Fase 1 ("ideas", por defecto): propone platos · Fase 2 ("receta"): receta de la idea elegida
    fase: Optional[Literal["ideas", "receta"]] = None
    idea: Optional[IdeaPlato] = None
    # Para pedir más ideas: de dónde sacarlas y qué no repetir
    fuente_busqueda: Optional[Literal["auto", "local", "web"]] = None
    ids_excluidos: list[int] = []          # recetas propias ya propuestas
    recetas_descartadas: list[str] = []    # nombres de platos ya propuestos
    # Chat libre: mensajes anteriores para que el agente entienda el contexto ("sí", "para 6 personas")
    historial: list[MensajeChat] = []


# Endpoint para recibir mensajes de texto
def _argumentos(request: ChatRequest) -> dict:
    return dict(
        pregunta=request.mensaje,
        intencion=request.intencion,
        ingredientes_db=request.ingredientes_db,
        ingredientes_manual=request.ingredientes_manual,
        tipo_comida=request.tipo_comida,
        fuente_busqueda=request.fuente_busqueda,
        ids_excluidos=request.ids_excluidos,
        recetas_descartadas=request.recetas_descartadas,
        fase=request.fase,
        idea=request.idea.model_dump() if request.idea else None,
        historial=[m.model_dump() for m in request.historial],
    )


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Respuesta completa de una vez: {"respuesta": str, "contexto": {...}}."""
    return await ejecutar_asistente(**_argumentos(request))


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Misma lógica que /chat, pero emite NDJSON (un evento JSON por línea) mientras trabaja."""
    print(f"📥 /chat/stream: {request.mensaje!r}", flush=True)

    async def eventos():
        try:
            async for evento in ejecutar_asistente_stream(**_argumentos(request)):
                yield json.dumps(evento, ensure_ascii=False) + "\n"
        except Exception as e:
            print(f"❌ Error en /chat/stream: {e}", flush=True)
            traceback.print_exc()   # fichero y línea exactos en docker logs
            yield json.dumps({"tipo": "error", "mensaje": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(eventos(), media_type="application/x-ndjson")

#Endpoint para transcribir la voz del usuario (el frontend envía luego el texto como un mensaje normal)
@app.post("/transcribir")
async def transcribir_endpoint(audio: UploadFile = File(...)):
    contenido = await audio.read()
    if not contenido:
        raise HTTPException(status_code=400, detail="El audio está vacío.")
    sufijo = os.path.splitext(audio.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=sufijo) as temporal:
        temporal.write(contenido)
        ruta = temporal.name
    try:
        t0 = time.perf_counter()
        # Whisper es costoso: en un hilo aparte para no bloquear al resto de peticiones
        texto = await run_in_threadpool(transcribir_audio, ruta)
        print(f"   🎙️ Transcripción en {time.perf_counter() - t0:.1f} s: {texto!r}", flush=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo transcribir el audio: {e}")
    finally:
        os.remove(ruta)
    return {"texto": texto}


#Endpoint para recibir mensajes de audio
@app.post("/chat/audio")
async def chat_audio_endpoint(audio: UploadFile = File(...)):
    # Guardar el audio temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(await audio.read())
        temp_path = temp_audio.name
        
    # Transcribir a texto
    texto_transcrito = transcribir_audio(temp_path)
    os.remove(temp_path)
    
    # Inyectar el texto en tu orquestador actual
    resultado = await ejecutar_asistente(pregunta=texto_transcrito)
    return {"transcripcion": texto_transcrito, **resultado}

#OPERACIONES BD SQLITE
#Endpoint para obtener todos los ingredientes de la despensa desde SQLite
@app.get("/ingredientes")
async def obtener_todos_los_ingredientes():
    """Conecta a SQLite y devuelve la lista completa de ingredientes de la despensa."""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
        
            cursor.execute("SELECT id, name, quantity, category, unit FROM ingredients")
            resultados = cursor.fetchall()
        
        # Mapeamos cada fila al modelo Ingrediente de Pydantic
        lista_ingredientes = [
            Ingrediente(
                id=fila[0],
                nombre=fila[1].capitalize(), 
                cantidad=str(fila[2]), 
                categoria=str(fila[3]),
                unidad=str(fila[4]) if len(fila) > 4 else None
            ) 
            for fila in resultados
        ]
        
        # FastAPI serializará automáticamente la lista de modelos a JSON
        return {"ingredientes": lista_ingredientes}
    
    except Exception as e:
        return {"error": str(e), "ingredientes": []}

#Endpoint para añadir un nuevo ingrediente a la despensa en SQLite
@app.post("/nuevo_ingrediente")    
def agregar_ingrediente(ingrediente: Ingrediente) -> bool:
    """Inserta un nuevo ingrediente en la tabla ingredients."""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingredients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    quantity TEXT NOT NULL,
                    category TEXT,
                    unit TEXT
                )
            """)
            
            conn.commit()
        # Si ya existe (aunque cambie el plural, las mayúsculas o las tildes) se suma la cantidad: no se duplica
        cantidad = sqlitebd.a_numero(ingrediente.cantidad)
        resumen = sqlitebd.anadir_ingrediente(ingrediente.nombre, cantidad if cantidad is not None else 0,
                                        ingrediente.unidad or "unidades", ingrediente.categoria or "Otros")
        print(f"✅ {resumen}")
        return True
    except Exception as e:
        print(f"❌ Error al guardar el ingrediente: {e}")
        return False

#Endpoint para añadir varios nuevos ingredientes a la despensa en SQLite
@app.post("/nuevos_ingredientes")  
def agregar_multiples_ingredientes(ingredientes: list[Ingrediente]) -> bool:
    """
    Inserta una lista de ingredientes en la tabla ingredients mediante una sola operación masiva.
    Formato esperado de la lista: [("nombre", "cantidad", "categoría"), ...]
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            datos_limpios = [
                (nombre.lower().strip(), cantidad, categoria) 
                for nombre, cantidad, categoria in ingredientes
            ]
            
            cursor.executemany(
                "INSERT INTO ingredients (name, quantity, category, unit) VALUES (?, ?, ?, ?)", 
                datos_limpios
            )
            
            conn.commit()
            print(f"✅ Se han guardado {cursor.rowcount} ingredientes nuevos en la despensa.")
            return True
            
    except sqlite3.Error as e:
        print(f"❌ Error de base de datos al guardar los ingredientes: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

@app.put("/actualizar_ingrediente")
async def actualizar_cantidad_ingrediente(ingrediente: Ingrediente):
    """Actualiza la cantidad de un ingrediente existente en la despensa."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
        
            if ingrediente.id is not None:
                cursor.execute("UPDATE ingredients SET quantity = ?, unit = ? WHERE id = ?",
                               (ingrediente.cantidad, ingrediente.unidad, ingrediente.id))
            else:
                cursor.execute(
                    "UPDATE ingredients SET quantity = ?, unit = ? WHERE name = ?",
                    (ingrediente.cantidad, ingrediente.unidad, ingrediente.nombre.lower().strip())
                )
            conn.commit()
            conn.close()
        
            return {"mensaje": f"Cantidad de '{ingrediente.nombre}' actualizada a {ingrediente.cantidad}+" f" {ingrediente.unidad}."}
    except Exception as e:
        return {"error": str(e)}

#Endpoint con los ingredientes principales de la despensa para un tipo de comida (selección rápida del chat)
@app.get("/ingredientes/principales")
def ingredientes_principales_endpoint(tipo_comida: Optional[str] = None, cantidad: int = 20):
    """Hasta `cantidad` ingredientes principales (sin condimentos), adecuados al tipo de comida y al azar."""
    from src.data import seleccion_ingredientes
    from src.llms.prompts import normalizar_tipo_comida
    preferencias = sqlitebd.obtener_preferencias()
    evitar = (preferencias.get("alergia") or []) + (preferencias.get("no_me_gusta") or [])
    elegidos = seleccion_ingredientes.seleccionar(sqlitebd.listar_despensa(), normalizar_tipo_comida(tipo_comida),
                                                  max(1, min(cantidad, 100)), evitar)
    return {"ingredientes": [i["nombre"].capitalize() for i in elegidos]}


#Endpoint para eliminar un ingrediente de la despensa
@app.delete("/ingrediente")
def eliminar_ingrediente(nombre: str, id: Optional[int] = None):
    """Elimina un ingrediente de la despensa: por id si se indica (solo ese), si no, por nombre."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            if id is not None:
                cursor = conn.execute("DELETE FROM ingredients WHERE id = ?", (id,))
            else:
                cursor = conn.execute("DELETE FROM ingredients WHERE name = ?", (nombre.lower().strip(),))
            conn.commit()
            borrados = cursor.rowcount
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Error de base de datos: {e}")
    if not borrados:
        raise HTTPException(status_code=404, detail=f"No hay ningún ingrediente llamado '{nombre}'.")
    print(f"🗑️ Ingrediente '{nombre}' eliminado de la despensa.", flush=True)
    return {"mensaje": f"'{nombre}' eliminado de la despensa."}

#OPERACIONES RAG
#Endpoint para verificar si hay documentos indexados en Milvus
@app.get("/estado_milvus")
def verificar_estado_milvus():
    """Comprueba si la colección existe y tiene datos para habilitar la UI."""
    try:
        client = MilvusClient(uri=MILVUS_URI)
        if not client.has_collection(COLLECTION_NAME):
            return {"hay_documentos": False}

        client.load_collection(COLLECTION_NAME)
        resultados = client.query(collection_name=COLLECTION_NAME, filter="", output_fields=["count(*)"])
        cantidad = resultados[0].get("count(*)", 0) if resultados else 0
        return {"hay_documentos": cantidad > 0, "fragmentos": cantidad}

    except Exception as e:
        print(f"Error conectando a Milvus: {e}")
        return {"hay_documentos": False}

#Endpoint para buscar recetas en Milvus: híbrida (por defecto), semántica o léxica (BM25)
@app.get("/recetas/buscar")
def buscar_recetas_endpoint(query: str, modo: str = "hibrida", top_k: int = 5,
                            ranker: str = "rrf", peso_denso: float = 0.7):
    """
    - modo: hibrida | semantica | lexica
    - ranker (solo híbrida): rrf | ponderado
    - peso_denso (solo ponderado): 0 = solo BM25, 1 = solo semántica
    """
    if modo not in MODOS:
        raise HTTPException(status_code=400, detail=f"Modo no válido. Usa uno de: {', '.join(sorted(MODOS))}")
    if ranker not in RANKERS:
        raise HTTPException(status_code=400, detail=f"Ranker no válido. Usa uno de: {', '.join(sorted(RANKERS))}")
    if not query.strip():
        return {"resultados": []}

    try:
        resultados = buscar_recetas(
            query, modo=modo, top_k=max(1, min(top_k, 20)), ranker=ranker, peso_denso=peso_denso
        )
        return {"resultados": resultados}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Metadatos de los documentos subidos (fecha, tamaño...) en SQLite; los fragmentos viven en Milvus ---
def _tabla_documentos(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documentos (
            nombre TEXT PRIMARY KEY,
            fecha_subida TEXT,
            tamano_bytes INTEGER,
            fragmentos INTEGER,
            recetas_tabla INTEGER
        )
    """)


def _guardar_metadatos(nombre: str, tamano: int, fragmentos: int, recetas_tabla: int):
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with sqlite3.connect(DB_PATH) as conn:
            _tabla_documentos(conn)
            conn.execute("INSERT OR REPLACE INTO documentos VALUES (?, ?, ?, ?, ?)",
                         (nombre, datetime.now().isoformat(timespec="seconds"), tamano, fragmentos, recetas_tabla))
    except sqlite3.Error as e:
        print(f"⚠️  No se pudieron guardar los metadatos de '{nombre}': {e}", flush=True)


#Endpoint para subir recetas (PDF, Word, Excel o CSV), trocearlas, vectorizarlas e indexarlas en Milvus
@app.post("/recetas/upload")
def subir_e_indexar_receta(archivo: UploadFile = File(...)):
    """Extrae el contenido (tablas incluidas), lo trocea y lo indexa (vector denso + BM25 de Milvus).
    Si ya existía un documento con el mismo nombre, se sustituye."""
    try:
        # Solo el nombre del archivo (sin rutas): es la clave del documento en Milvus y en disco
        nombre = almacen.ruta(archivo.filename or "documento").name
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    crono = Cronometro("RAG indexación")
    with crono.fase("lectura"):
        contenido = archivo.file.read()
    try:
        with crono.fase("extracción"):
            extraccion = extraer_documento(nombre, contenido)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el documento: {e}")

    for aviso in extraccion.avisos:
        print(f"   ⚠️  {nombre}: {aviso}", flush=True)
    if extraccion.vacia:
        motivo = " ".join(extraccion.avisos) or "El documento está vacío o no tiene texto extraíble."
        raise HTTPException(status_code=400, detail=motivo)

    try:
        fragmentos = indexar_documento(extraccion.texto, nombre, recetas=extraccion.recetas, crono=crono)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # El original se guarda solo si la indexación ha ido bien (si ya existía, se sustituye)
    try:
        with crono.fase("guardar original"):
            destino = almacen.guardar(nombre, contenido)
        print(f"   💾 Original guardado en {destino}", flush=True)
    except OSError as e:
        print(f"⚠️  Indexado, pero no se pudo guardar el original de '{nombre}': {e}", flush=True)

    _guardar_metadatos(nombre, len(contenido), fragmentos, len(extraccion.recetas))
    detalle = f", {len(extraccion.recetas)} recetas de tabla" if extraccion.recetas else ""
    return {"mensaje": f"'{nombre}' indexado correctamente ({fragmentos} fragmentos{detalle}) "
                       f"en {crono.total:.1f} s.",
            "fragmentos": fragmentos, "recetas_tabla": len(extraccion.recetas), "tiempos": crono.resumen(),
            "avisos": extraccion.avisos}


#Endpoint para listar los documentos guardados (Milvus manda; SQLite aporta fecha y tamaño)
@app.get("/recetas/documentos")
def listar_documentos_endpoint():
    try:
        documentos = listar_documentos()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo consultar Milvus: {e}")

    metadatos = {}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            _tabla_documentos(conn)
            for nombre, fecha, tamano, _frag, recetas_tabla in conn.execute("SELECT * FROM documentos"):
                metadatos[nombre] = {"fecha_subida": fecha, "tamano_bytes": tamano, "recetas_tabla": recetas_tabla}
    except sqlite3.Error as e:
        print(f"⚠️  No se pudieron leer los metadatos de documentos: {e}", flush=True)

    for doc in documentos:
        doc.update(metadatos.get(doc["nombre"], {"fecha_subida": None, "tamano_bytes": None, "recetas_tabla": None}))
        doc["archivo_guardado"] = almacen.existe(doc["nombre"])
    return {"documentos": documentos, "total_fragmentos": sum(d["fragmentos"] for d in documentos)}


#Endpoint para borrar un documento (todos sus fragmentos)
@app.delete("/recetas/documentos")
def borrar_documento_endpoint(nombre: str):
    try:
        borrados = borrar_documento(nombre)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo borrar en Milvus: {e}")
    original_borrado = almacen.borrar(nombre)
    if not borrados and not original_borrado:
        raise HTTPException(status_code=404, detail=f"No hay ningún documento llamado '{nombre}'.")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            _tabla_documentos(conn)
            conn.execute("DELETE FROM documentos WHERE nombre = ?", (nombre,))
    except sqlite3.Error:
        pass
    return {"mensaje": f"'{nombre}' eliminado ({borrados} fragmentos"
                       f"{' y el archivo original' if original_borrado else ''}).",
            "fragmentos": borrados, "original_borrado": original_borrado}


#Endpoint para descargar el archivo original de un documento
@app.get("/recetas/documentos/archivo")
def descargar_documento(nombre: str):
    try:
        ruta_archivo = almacen.ruta(nombre)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ruta_archivo.is_file():
        raise HTTPException(status_code=404, detail=f"No se guardó el original de '{nombre}'. Vuelve a subirlo.")
    return FileResponse(ruta_archivo, filename=ruta_archivo.name)


# Arranque del servidor web
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)