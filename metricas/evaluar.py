"""Evaluación de CookingAura: calidad de la búsqueda (recall) y tiempos de respuesta.

Ejecutar dentro del contenedor del backend (los resultados quedan en data/eval del host):

  1) Crear el conjunto de prueba a partir de tus recetas indexadas (una consulta por fragmento):
       python -m metricas.evaluar generar --n 40
  2) Recall@k, MRR y latencia de cada modo de búsqueda (semántica, léxica, híbridas):
       python -m metricas.evaluar recall --k 1 3 5 10
  3) Tiempos de extremo a extremo del asistente (con el backend levantado):
       python -m metricas.evaluar latencia --repeticiones 3
  Casos propios: python -m metricas.evaluar fragmentos --buscar "gazpacho"  -> ids para consultas.jsonl
  Calidad de la generación (ideas, recetas, agente): python -m metricas.evaluar generacion [--juez]
  4) Flujo completo por TEXTO frente a flujo completo por VOZ (transcripción + asistente):
       python -m metricas.evaluar preparar-voz        # casos de prueba y audios (sintetizados o tuyos)
       python -m metricas.evaluar flujos --repeticiones 3 [--receta]
"""
import argparse
import csv
import json
import random
import re
import shutil
import statistics
import subprocess
import time
import unicodedata
import wave
from datetime import datetime
from pathlib import Path

CARPETA = Path(__file__).resolve().parents[1] / "data" / "eval"
CONSULTAS = CARPETA / "consultas.jsonl"
CARPETA_VOZ = CARPETA / "voz"
CASOS_VOZ = CARPETA_VOZ / "casos.json"

# Casos del flujo texto/voz. Solo lecturas: la evaluación no cambia la despensa ni la lista de la compra.
# intencion: "ideas" (flujo de ideas con botones), "asistente" (agente con herramientas) o "directa" (saludo/fuera de tema)
CASOS_POR_DEFECTO = [
    {"id": "saludo_comer", "texto": "Hola, ¿qué puedo comer hoy?", "intencion": "ideas"},
    {"id": "cena_ingredientes", "texto": "¿Qué ceno con huevos y patatas?", "intencion": "ideas"},
    {"id": "consulta_despensa", "texto": "¿Me quedan huevos en la despensa?", "intencion": "asistente"},
    {"id": "consulta_compra", "texto": "¿Qué tengo en la lista de la compra?", "intencion": "asistente"},
    {"id": "duda_cocina", "texto": "¿Cómo se pocha una cebolla?", "intencion": "asistente"},
    {"id": "saludo", "texto": "Buenos días", "intencion": "directa"},
]

# Configuraciones de búsqueda a comparar: (nombre, parámetros de buscar_recetas)
CONFIGURACIONES = [
    ("semantica", {"modo": "semantica"}),
    ("lexica_bm25", {"modo": "lexica"}),
    ("hibrida_rrf", {"modo": "hibrida", "ranker": "rrf"}),
    ("hibrida_pond_0.3", {"modo": "hibrida", "ranker": "ponderado", "peso_denso": 0.3}),
    ("hibrida_pond_0.5", {"modo": "hibrida", "ranker": "ponderado", "peso_denso": 0.5}),
    ("hibrida_pond_0.7", {"modo": "hibrida", "ranker": "ponderado", "peso_denso": 0.7}),
]

# Escenarios de latencia (fase ideas; si "receta" es True, después se pide la receta de una idea)
ESCENARIOS = [
    {"nombre": "ideas_con_ingredientes", "receta": True,
     "payload": {"mensaje": "¿Qué ceno con huevos, patatas y cebolla?", "fase": "ideas",
                 "ingredientes_db": "huevos, patatas, cebolla", "tipo_comida": "cena"}},
    {"nombre": "ideas_despensa", "receta": True,
     "payload": {"mensaje": "¿Qué como hoy con lo que tengo?", "fase": "ideas", "intencion": "RECETA"}},
    {"nombre": "chat_libre", "receta": False,
     "payload": {"mensaje": "Quiero algo rápido para desayunar con avena", "fase": "ideas"}},
]


# ----------------------------------------------------------------- utilidades
def _percentil(valores: list[float], p: float) -> float:
    if not valores:
        return float("nan")
    if len(valores) == 1:
        return valores[0]
    return statistics.quantiles(valores, n=100, method="inclusive")[int(p) - 1]


def _resumen_tiempos(valores: list[float]) -> str:
    if not valores:
        return "-"
    return f"p50 {statistics.median(valores):6.2f}s · p95 {_percentil(valores, 95):6.2f}s"


def _guardar_csv(ruta: Path, filas: list[dict]):
    if not filas:
        return
    with ruta.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)


def _marca() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ------------------------------------------------------------ 1. generar
PROMPT_CONSULTA = """Esta es una receta guardada por un usuario:

---
{texto}
---

Escribe UNA consulta corta, en español, que haría alguien que busca exactamente esta receta.
No uses el nombre exacto del plato: describe ingredientes principales, tipo de plato o momento del día
(por ejemplo: "algo con garbanzos y espinacas para comer").
Responde SOLO con JSON: {{"consulta": "..."}}"""


def generar(n: int, motor: str, semilla: int):
    """Crea consultas sintéticas: cada una tiene como respuesta correcta el fragmento del que sale."""
    from src.agents.rag_node import get_client, COLLECTION_NAME, _preparar_coleccion
    from src.agents.ideas_node import titulo_y_resumen

    cliente = get_client()
    if not _preparar_coleccion(cliente):
        raise SystemExit("❌ La colección 'recetas' no existe o está vacía.")

    filas = cliente.query(collection_name=COLLECTION_NAME, filter="",
                          output_fields=["id", "texto", "fuente"], limit=min(max(n * 5, 100), 16384))
    candidatas = [f for f in filas if len((f.get("texto") or "").strip()) >= 200]
    random.Random(semilla).shuffle(candidatas)
    if not candidatas:
        raise SystemExit("❌ No hay fragmentos con texto suficiente.")

    if motor == "gemini":
        from src.agents.web_search_node import generar_json_gemini as generar_json
    else:
        from src.llms.ollama_client import generar_json_ollama as generar_json

    CARPETA.mkdir(parents=True, exist_ok=True)
    creadas = 0
    with CONSULTAS.open("w", encoding="utf-8") as salida:
        for fila in candidatas:
            if creadas >= n:
                break
            try:
                respuesta = generar_json(PROMPT_CONSULTA.format(texto=fila["texto"][:1500]))
                consulta = str(json.loads(respuesta).get("consulta", "")).strip()
            except Exception as e:
                print(f"⚠️  Fragmento {fila['id']} omitido: {e}")
                continue
            if not consulta:
                continue
            titulo, _ = titulo_y_resumen(fila["texto"])
            registro = {"consulta": consulta, "ids_relevantes": [fila["id"]],
                        "fuente": fila.get("fuente", ""), "titulo": titulo}
            salida.write(json.dumps(registro, ensure_ascii=False) + "\n")
            creadas += 1
            print(f"[{creadas}/{n}] {titulo[:40]:40} ← «{consulta}»")

    print(f"\n✅ {creadas} consultas en {CONSULTAS}")
    print("   Revísalas y corrige o añade las tuyas: cada línea es un JSON con 'consulta' e 'ids_relevantes'.")


# -------------------------------------------------------------- 2. recall
def recall(ks: list[int], configs: list[str] | None):
    from src.agents.rag_node import buscar_recetas

    if not CONSULTAS.exists():
        raise SystemExit(f"❌ No existe {CONSULTAS}. Ejecuta antes: python -m metricas.evaluar generar")
    casos = [json.loads(l) for l in CONSULTAS.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    sin_respuesta = [c["consulta"] for c in casos if not c.get("ids_relevantes") and not c.get("texto_relevante")]
    if sin_respuesta:
        raise SystemExit(f"❌ Estos casos no indican la respuesta correcta (ids_relevantes o texto_relevante): {sin_respuesta}")
    k_max = max(ks)
    seleccion = [c for c in CONFIGURACIONES if not configs or c[0] in configs]

    print(f"Evaluando {len(casos)} consultas · k = {ks}\n")
    buscar_recetas(casos[0]["consulta"], top_k=1)   # calentamiento: carga del modelo de embeddings

    resumen, detalle = [], []
    for nombre, params in seleccion:
        aciertos = {k: 0 for k in ks}
        aciertos_doc = {k: 0 for k in ks}
        rr, tiempos = [], []
        for caso in casos:
            relevantes = set(caso.get("ids_relevantes") or [])
            texto_rel = _normalizar_texto(caso.get("texto_relevante", ""))
            t = time.perf_counter()
            resultados = buscar_recetas(caso["consulta"], top_k=k_max, **params)
            tiempos.append(time.perf_counter() - t)

            ids = [r["id"] for r in resultados]
            fuentes = [r.get("fuente") for r in resultados]
            # Relevante: su id está en ids_relevantes o su texto contiene texto_relevante
            rango = next((i for i, r in enumerate(resultados, start=1)
                          if r["id"] in relevantes
                          or (texto_rel and texto_rel in _normalizar_texto(r.get("texto", "")))), None)
            rr.append(1 / rango if rango else 0.0)
            for k in ks:
                aciertos[k] += bool(rango and rango <= k)
                aciertos_doc[k] += caso.get("fuente", "") in fuentes[:k]
            detalle.append({"config": nombre, "consulta": caso["consulta"], "titulo": caso.get("titulo", ""),
                            "rango": rango or "", "segundos": round(tiempos[-1], 3)})

        n = len(casos)
        fila = {"config": nombre}
        fila.update({f"recall@{k}": round(aciertos[k] / n, 3) for k in ks})
        fila["mrr"] = round(sum(rr) / n, 3)
        fila.update({f"doc_hit@{k}": round(aciertos_doc[k] / n, 3) for k in ks})
        fila["latencia_p50_ms"] = round(statistics.median(tiempos) * 1000, 1)
        fila["latencia_p95_ms"] = round(_percentil(tiempos, 95) * 1000, 1)
        resumen.append(fila)

    # Tabla por consola
    cabecera = ["config"] + [f"recall@{k}" for k in ks] + ["mrr", f"doc_hit@{k_max}", "latencia_p50_ms", "latencia_p95_ms"]
    print("  ".join(f"{c:>17}" for c in cabecera))
    for fila in resumen:
        print("  ".join(f"{str(fila[c]):>17}" for c in cabecera))

    marca = _marca()
    _guardar_csv(CARPETA / f"recall_{marca}.csv", resumen)
    _guardar_csv(CARPETA / f"recall_detalle_{marca}.csv", detalle)
    mejor = max(resumen, key=lambda f: (f[f"recall@{ks[min(1, len(ks) - 1)]}"], f["mrr"]))
    print(f"\n🏆 Mejor configuración: {mejor['config']}")
    print(f"✅ Resultados en {CARPETA}/recall_{marca}.csv (y el detalle por consulta)")


# ------------------------------------------------------------ 3. latencia
def _ejecutar(url: str, payload: dict, timeout: float) -> dict:
    """Llama a /chat/stream y mide: tiempo total, primer token, duración de cada nodo."""
    import httpx

    t0 = time.perf_counter()
    medida = {"nodos": {}, "primer_token_s": None, "contexto": {}, "metricas": {}, "error": None}
    with httpx.stream("POST", f"{url}/chat/stream", json=payload, timeout=timeout) as r:
        r.raise_for_status()
        for linea in r.iter_lines():
            if not linea:
                continue
            evento = json.loads(linea)
            if evento["tipo"] == "estado":
                nodo = evento.get("nodo") or evento.get("texto", "?")
                medida["nodos"][nodo] = round(medida["nodos"].get(nodo, 0) + (evento.get("segundos") or 0), 2)
            elif evento["tipo"] == "token" and medida["primer_token_s"] is None:
                medida["primer_token_s"] = time.perf_counter() - t0
            elif evento["tipo"] == "fin":
                medida["contexto"] = evento.get("contexto") or {}
                medida["metricas"] = evento.get("metricas") or {}
                medida["respuesta"] = evento.get("respuesta", "")
            elif evento["tipo"] == "error":
                medida["error"] = evento.get("mensaje")
    medida["total_s"] = time.perf_counter() - t0
    return medida


def latencia(url: str, repeticiones: int, timeout: float):
    filas = []

    def registrar(escenario: str, fase: str, rep: int, m: dict):
        filas.append({
            "escenario": escenario, "fase": fase, "repeticion": rep,
            "total_s": round(m["total_s"], 2),
            "primer_token_s": round(m["primer_token_s"], 2) if m["primer_token_s"] else "",
            "generacion_s": m["metricas"].get("generacion_s", ""),
            "motor": m["metricas"].get("motor", ""),
            "n_ideas": len(m["contexto"].get("ideas") or []),
            "nodos": json.dumps(m["nodos"], ensure_ascii=False),
            "error": m["error"] or "",
        })
        print(f"  {escenario:24} {fase:7} #{rep}  total {m['total_s']:6.1f}s"
              + (f" · 1er token {m['primer_token_s']:5.1f}s" if fase == "receta" and m["primer_token_s"] else "")
              + (f" · {filas[-1]['n_ideas']} ideas" if fase == "ideas" else "")
              + (f"  ❌ {m['error']}" if m["error"] else ""))

    print(f"Midiendo contra {url} · {repeticiones} repeticiones por escenario\n")
    for rep in range(1, repeticiones + 1):
        for esc in ESCENARIOS:
            try:
                m = _ejecutar(url, esc["payload"], timeout)
            except Exception as e:
                print(f"  {esc['nombre']:24} ideas   #{rep}  ❌ {e}")
                continue
            registrar(esc["nombre"], "ideas", rep, m)

            ideas = m["contexto"].get("ideas") or []
            if not esc["receta"] or not ideas:
                continue
            # Una receta de idea propia y otra de idea creativa, si las hay
            elegidas = [next((i for i in ideas if i["origen"] == o), None) for o in ("local", "creativa")]
            for idea in filter(None, elegidas):
                payload = {"mensaje": f"Quiero preparar: {idea['nombre']}", "fase": "receta", "idea": idea,
                           "ingredientes_db": m["contexto"].get("ingredientes"),
                           "tipo_comida": m["contexto"].get("tipo_comida")}
                try:
                    registrar(f"{esc['nombre']}/{idea['origen']}", "receta", rep, _ejecutar(url, payload, timeout))
                except Exception as e:
                    print(f"  {esc['nombre']:24} receta  #{rep}  ❌ {e}")

    # Resumen: tiempos por escenario/fase y media por nodo
    print("\nResumen")
    grupos = {}
    for f in filas:
        if not f["error"]:
            grupos.setdefault((f["escenario"], f["fase"]), []).append(f)
    for (escenario, fase), fs in grupos.items():
        totales = [f["total_s"] for f in fs]
        linea = f"  {escenario:32} {fase:7} total {_resumen_tiempos(totales)}"
        primeros = [f["primer_token_s"] for f in fs if f["primer_token_s"] != ""]
        if fase == "receta" and primeros:
            linea += f" | 1er token {_resumen_tiempos(primeros)}"
        print(linea)

    por_nodo = {}
    for f in filas:
        for nodo, seg in json.loads(f["nodos"]).items():
            if seg is not None:
                por_nodo.setdefault(nodo, []).append(seg)
    print("\nTiempo por nodo (todas las ejecuciones)")
    for nodo, segs in sorted(por_nodo.items(), key=lambda x: -statistics.median(x[1])):
        print(f"  {nodo:22} {_resumen_tiempos(segs)}  (n={len(segs)})")

    CARPETA.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA / f"latencia_{_marca()}.csv"
    _guardar_csv(ruta, filas)
    print(f"\n✅ Detalle en {ruta}")


# ---------------------------------------------------- 4. flujos texto / voz
def _normalizar_palabras(texto: str) -> list[str]:
    texto = unicodedata.normalize("NFD", (texto or "").lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9ñ]+", texto)


def wer(referencia: str, hipotesis: str) -> float:
    """Word Error Rate: (sustituciones + borrados + inserciones) / palabras de la referencia.
    Sin tildes, mayúsculas ni puntuación. 0 = transcripción perfecta."""
    ref, hip = _normalizar_palabras(referencia), _normalizar_palabras(hipotesis)
    if not ref:
        return 0.0 if not hip else 1.0
    anterior = list(range(len(hip) + 1))
    for i, r in enumerate(ref, start=1):
        actual = [i] + [0] * len(hip)
        for j, h in enumerate(hip, start=1):
            actual[j] = min(anterior[j] + 1, actual[j - 1] + 1, anterior[j - 1] + (r != h))
        anterior = actual
    return anterior[-1] / len(ref)


def _duracion_wav(ruta: Path) -> float | None:
    try:
        with wave.open(str(ruta), "rb") as w:
            return w.getnframes() / w.getframerate()
    except (wave.Error, EOFError, OSError):
        return None


def _sintetizar(texto: str, ruta: Path) -> bool:
    """Genera un WAV con espeak-ng (voz sintética en español). False si no está instalado."""
    programa = shutil.which("espeak-ng") or shutil.which("espeak")
    if not programa:
        return False
    resultado = subprocess.run([programa, "-v", "es", "-s", "150", "-w", str(ruta), texto],
                               capture_output=True, timeout=60)
    return resultado.returncode == 0 and ruta.exists()


def _cargar_casos() -> list[dict]:
    if CASOS_VOZ.exists():
        return json.loads(CASOS_VOZ.read_text(encoding="utf-8"))
    return [dict(c, audio=f"{c['id']}.wav") for c in CASOS_POR_DEFECTO]


def preparar_voz():
    """Crea data/eval/voz/casos.json y, si hay espeak-ng, sintetiza un WAV por caso.
    Para medir con voz real, sustituye los WAV por grabaciones tuyas con el mismo nombre."""
    CARPETA_VOZ.mkdir(parents=True, exist_ok=True)
    casos = _cargar_casos()
    if not CASOS_VOZ.exists():
        CASOS_VOZ.write_text(json.dumps(casos, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Casos de prueba en {CASOS_VOZ} (edítalos o añade los tuyos)")
    for caso in casos:
        ruta = CARPETA_VOZ / caso["audio"]
        if ruta.exists():
            print(f"   {caso['audio']:28} ya existe ({_duracion_wav(ruta) or 0:.1f} s): se conserva")
        elif _sintetizar(caso["texto"], ruta):
            print(f"   {caso['audio']:28} sintetizado ({_duracion_wav(ruta) or 0:.1f} s) ← «{caso['texto']}»")
        else:
            print(f"   {caso['audio']:28} FALTA: grábalo tú (WAV) o instala espeak-ng en el contenedor")
    print("\nConsejo: una voz sintética mide el sistema, pero tus propias grabaciones (con el micro que usarás)"
          "\ndan un WER y unos tiempos más realistas. Basta con guardarlas con el mismo nombre de archivo.")


def _transcribir_http(url: str, ruta: Path, timeout: float) -> tuple[str, float]:
    import httpx
    t0 = time.perf_counter()
    with ruta.open("rb") as f:
        r = httpx.post(f"{url}/transcribir", files={"audio": (ruta.name, f, "audio/wav")}, timeout=timeout)
    r.raise_for_status()
    return (r.json().get("texto") or "").strip(), time.perf_counter() - t0


def _ruta(medida: dict) -> str:
    """Camino que ha seguido el grafo, deducido de los pasos emitidos."""
    nodos = medida.get("nodos") or {}
    if any(n == "agente" or n.startswith("herramienta:") for n in nodos):
        return "asistente"
    if any(n in nodos for n in ("ideas_locales", "ideas_creativas", "presentar_ideas", "sin_recetas_locales")):
        return "ideas"
    return "directa"


def flujos(url: str, repeticiones: int, timeout: float, con_receta: bool):
    """Mide el flujo completo por texto y por voz con los mismos casos."""
    casos = _cargar_casos()
    filas = []

    def registrar(caso, modo, rep, m, fase="chat", transcripcion="", t_trans=None, duracion=None, error=None):
        total_chat = m.get("total_s") if m else None
        total = (total_chat or 0) + (t_trans or 0) if total_chat is not None else None
        primer = m.get("primer_token_s") if m else None
        fila = {
            "caso": caso["id"], "modo": modo, "fase": fase, "repeticion": rep,
            "intencion_esperada": caso.get("intencion", "") if fase == "chat" else "",
            "ruta": _ruta(m) if m and fase == "chat" else "",
            "acierto_ruta": (int(_ruta(m) == caso.get("intencion")) if m and fase == "chat" and caso.get("intencion")
                             else ""),
            "total_s": round(total, 2) if total is not None else "",
            "primer_token_s": round(primer + (t_trans or 0), 2) if primer is not None else "",
            "transcripcion_s": round(t_trans, 2) if t_trans is not None else "",
            "audio_s": round(duracion, 2) if duracion else "",
            "rtf": round(t_trans / duracion, 2) if t_trans is not None and duracion else "",
            "wer": round(wer(caso["texto"], transcripcion), 3) if modo == "voz" and transcripcion else "",
            "transcripcion": transcripcion,
            "nodos": json.dumps((m or {}).get("nodos", {}), ensure_ascii=False),
            "error": error or (m or {}).get("error") or "",
        }
        filas.append(fila)
        detalle = (f"total {fila['total_s']:>6}s" if fila["total_s"] != "" else "total      -")
        if modo == "voz" and fila["transcripcion_s"] != "":
            detalle += f" · transcripción {fila['transcripcion_s']}s · WER {fila['wer']}"
        if fila["ruta"]:
            detalle += f" · ruta {fila['ruta']}{' ✅' if fila['acierto_ruta'] == 1 else ' ❌' if fila['acierto_ruta'] == 0 else ''}"
        print(f"  {caso['id']:20} {modo:5} {fase:6} #{rep}  {detalle}" + (f"  ❌ {fila['error']}" if fila["error"] else ""))
        return fila

    def receta(caso, modo, rep, m):
        ideas = (m.get("contexto") or {}).get("ideas") or []
        if not con_receta or not ideas:
            return
        idea = ideas[0]
        payload = {"mensaje": f"Quiero preparar: {idea['nombre']}", "fase": "receta", "idea": idea,
                   "ingredientes_db": m["contexto"].get("ingredientes"), "tipo_comida": m["contexto"].get("tipo_comida")}
        try:
            registrar(caso, modo, rep, _ejecutar(url, payload, timeout), fase="receta")
        except Exception as e:
            registrar(caso, modo, rep, None, fase="receta", error=str(e))

    hay_voz = any((CARPETA_VOZ / c.get("audio", "")).exists() for c in casos if c.get("audio"))
    if not hay_voz:
        print("⚠️  No hay audios en data/eval/voz: ejecuta antes 'preparar-voz'. Se mide solo el flujo por texto.\n")

    print(f"Midiendo {len(casos)} casos contra {url} · {repeticiones} repeticiones · texto y voz\n")
    for rep in range(1, repeticiones + 1):
        for caso in casos:
            base = {"fase": "ideas", "historial": []}
            # --- Texto: el mensaje tal cual
            try:
                m = _ejecutar(url, {**base, "mensaje": caso["texto"]}, timeout)
                registrar(caso, "texto", rep, m)
                receta(caso, "texto", rep, m)
            except Exception as e:
                registrar(caso, "texto", rep, None, error=str(e))

            # --- Voz: audio -> /transcribir -> el texto transcrito al mismo chat
            ruta_audio = CARPETA_VOZ / caso.get("audio", "")
            if not caso.get("audio") or not ruta_audio.exists():
                continue
            try:
                transcrito, t_trans = _transcribir_http(url, ruta_audio, timeout)
            except Exception as e:
                registrar(caso, "voz", rep, None, error=f"transcripción: {e}")
                continue
            if not transcrito:
                registrar(caso, "voz", rep, None, t_trans=t_trans, duracion=_duracion_wav(ruta_audio),
                          error="transcripción vacía")
                continue
            try:
                m = _ejecutar(url, {**base, "mensaje": transcrito}, timeout)
                registrar(caso, "voz", rep, m, transcripcion=transcrito, t_trans=t_trans,
                          duracion=_duracion_wav(ruta_audio))
                receta(caso, "voz", rep, m)
            except Exception as e:
                registrar(caso, "voz", rep, None, transcripcion=transcrito, t_trans=t_trans, error=str(e))

    # ------------------------------------------------------------ resumen
    def valores(clave, **filtro):
        return [f[clave] for f in filas
                if f[clave] != "" and not f["error"] and all(f[k] == v for k, v in filtro.items())]

    print("\n================ RESUMEN: TEXTO frente a VOZ ================")
    for modo in ("texto", "voz"):
        totales = valores("total_s", modo=modo, fase="chat")
        if not totales:
            continue
        aciertos = valores("acierto_ruta", modo=modo, fase="chat")
        linea = f"\n{modo.upper():6} total        {_resumen_tiempos(totales)}"
        primeros = valores("primer_token_s", modo=modo, fase="chat")
        if primeros:
            linea += f"\n       1ª respuesta {_resumen_tiempos(primeros)}"
        if aciertos:
            linea += f"\n       ruta correcta {sum(aciertos)}/{len(aciertos)} ({sum(aciertos) / len(aciertos):.0%})"
        print(linea)
        if modo == "voz":
            trans, rtfs, wers = valores("transcripcion_s", modo="voz"), valores("rtf", modo="voz"), valores("wer", modo="voz")
            if trans:
                print(f"       transcripción {_resumen_tiempos(trans)}"
                      + (f" · RTF medio {statistics.mean(rtfs):.2f}" if rtfs else "")
                      + (f" · {statistics.mean(trans) / statistics.mean(totales):.0%} del tiempo total"
                         if statistics.mean(totales) > 0 else ""))
            if wers:
                print(f"       WER medio {statistics.mean(wers):.1%} (0% = transcripción perfecta)")
        recetas = valores("total_s", modo=modo, fase="receta")
        if recetas:
            print(f"       receta        {_resumen_tiempos(recetas)}")

    print("\nPor caso (p50 del total, texto → voz):")
    for caso in casos:
        t = valores("total_s", caso=caso["id"], modo="texto", fase="chat")
        v = valores("total_s", caso=caso["id"], modo="voz", fase="chat")
        txt = f"{statistics.median(t):6.1f}s" if t else "     -"
        voz = f"{statistics.median(v):6.1f}s" if v else "     -"
        extra = ""
        if t and v:
            extra = f"  (+{statistics.median(v) - statistics.median(t):.1f}s por voz)"
        transcritos = [f["transcripcion"] for f in filas if f["caso"] == caso["id"] and f["modo"] == "voz" and f["transcripcion"]]
        if transcritos and wer(caso["texto"], transcritos[0]) > 0:
            extra += f"  · oído: «{transcritos[0]}»"
        print(f"  {caso['id']:20} {txt} → {voz}{extra}")

    CARPETA.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA / f"flujos_{_marca()}.csv"
    _guardar_csv(ruta, filas)
    print(f"\n✅ Detalle en {ruta}")


# ------------------------------------------------- 5. fragmentos (ayuda para casos)
def _normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFD", (texto or "").lower())
    return " ".join("".join(c for c in texto if unicodedata.category(c) != "Mn").split())


def fragmentos(buscar: str | None, limite: int):
    """Lista los fragmentos indexados con su id, para escribir casos de prueba a mano.
    Con --buscar, solo los que contienen todas esas palabras (sin tildes ni mayúsculas)."""
    from src.agents.rag_node import get_client, COLLECTION_NAME, _preparar_coleccion
    from src.agents.ideas_node import titulo_y_resumen

    cliente = get_client()
    if not _preparar_coleccion(cliente):
        raise SystemExit("❌ La colección 'recetas' no existe o está vacía.")
    filas = cliente.query(collection_name=COLLECTION_NAME, filter="", output_fields=["id", "texto", "fuente"],
                          limit=16384)
    palabras = _normalizar_texto(buscar or "").split()
    elegidas = [f for f in filas if all(p in _normalizar_texto(f.get("texto", "")) for p in palabras)]
    elegidas.sort(key=lambda f: (f.get("fuente") or "", titulo_y_resumen(f.get("texto", ""))[0]))

    salida = []
    for f in elegidas:
        titulo, resumen = titulo_y_resumen(f.get("texto", ""))
        salida.append({"id": f["id"], "fuente": f.get("fuente", ""), "titulo": titulo, "inicio": resumen[:120]})

    print(f"{len(elegidas)} fragmentos" + (f" con «{buscar}»" if buscar else "") + f" (de {len(filas)} en total)\n")
    for f in salida[:limite]:
        print(f"  {f['id']:>20}  {f['fuente'][:28]:28}  {f['titulo'][:60]}")
    if len(salida) > limite:
        print(f"  … y {len(salida) - limite} más (están todos en el CSV)")

    CARPETA.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA / "fragmentos.csv"
    _guardar_csv(ruta, salida)
    print(f"\n✅ Lista completa en {ruta}")
    print("   Copia el id en \"ids_relevantes\" de tu caso, o usa \"texto_relevante\" con un trozo de su título.")


# ------------------------------------------------------------------- CLI
def main():
    import src.config  # noqa: F401  (carga el .env)

    parser = argparse.ArgumentParser(description="Evaluación de CookingAura")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_gen = sub.add_parser("generar", help="Crear consultas de prueba a partir de las recetas indexadas")
    p_gen.add_argument("--n", type=int, default=40, help="Número de consultas")
    p_gen.add_argument("--motor", choices=["ollama", "gemini"], default=None,
                       help="LLM para redactar las consultas (por defecto, IDEAS_MOTOR)")
    p_gen.add_argument("--semilla", type=int, default=42)

    p_rec = sub.add_parser("recall", help="Recall@k, MRR y latencia de cada modo de búsqueda")
    p_rec.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    p_rec.add_argument("--configs", nargs="+", choices=[c[0] for c in CONFIGURACIONES], default=None)

    p_lat = sub.add_parser("latencia", help="Tiempos de extremo a extremo contra /chat/stream")
    p_lat.add_argument("--url", default="http://localhost:8080")
    p_lat.add_argument("--repeticiones", type=int, default=3)
    p_lat.add_argument("--timeout", type=float, default=900)

    p_gen2 = sub.add_parser("generacion", help="Calidad de ideas, recetas y agente (data/eval/generacion.json)")
    p_gen2.add_argument("--url", default="http://localhost:8080")
    p_gen2.add_argument("--repeticiones", type=int, default=1)
    p_gen2.add_argument("--timeout", type=float, default=900)
    p_gen2.add_argument("--juez", action="store_true", help="Añadir la nota de Gemini (1-5) con una rúbrica fija")
    p_gen2.add_argument("--solo", choices=["ideas", "plato", "agente"], default=None, help="Solo un tipo de caso")

    p_frag = sub.add_parser("fragmentos", help="Lista los fragmentos indexados con su id (para escribir casos)")
    p_frag.add_argument("--buscar", default=None, help="Solo los que contienen estas palabras")
    p_frag.add_argument("--limite", type=int, default=40, help="Cuántos mostrar por consola (el CSV tiene todos)")

    sub.add_parser("preparar-voz", help="Casos de prueba de voz y sus audios (sintetizados con espeak-ng)")

    p_flu = sub.add_parser("flujos", help="Flujo completo por texto frente a flujo completo por voz")
    p_flu.add_argument("--url", default="http://localhost:8080")
    p_flu.add_argument("--repeticiones", type=int, default=3)
    p_flu.add_argument("--timeout", type=float, default=900)
    p_flu.add_argument("--receta", action="store_true", help="Pedir también la receta de la primera idea")

    args = parser.parse_args()
    if args.comando == "generar":
        from src.config import IDEAS_MOTOR
        generar(args.n, args.motor or IDEAS_MOTOR, args.semilla)
    elif args.comando == "recall":
        recall(sorted(set(args.k)), args.configs)
    elif args.comando == "latencia":
        latencia(args.url.rstrip("/"), args.repeticiones, args.timeout)
    elif args.comando == "generacion":
        from metricas.generacion import evaluar_generacion
        evaluar_generacion(args.url.rstrip("/"), args.repeticiones, args.timeout, args.juez, args.solo)
    elif args.comando == "fragmentos":
        fragmentos(args.buscar, args.limite)
    elif args.comando == "preparar-voz":
        preparar_voz()
    else:
        flujos(args.url.rstrip("/"), args.repeticiones, args.timeout, args.receta)


if __name__ == "__main__":
    main()