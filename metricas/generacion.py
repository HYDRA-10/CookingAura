"""Evaluación de la GENERACIÓN: ideas de platos, recetas y agente del chat libre.

    python -m metricas.evaluar generacion                  # métricas automáticas
    python -m metricas.evaluar generacion --juez           # + nota de Gemini (1-5) con una rúbrica fija
    python -m metricas.evaluar generacion --solo agente    # solo un tipo de caso (ideas, plato, agente)

Casos en data/eval/generacion.json. Cada caso se ejecuta contra /chat/stream (como el frontend) y se mide:
- Ideas: cantidad, duplicados, uso de los ingredientes, alergias, ideas propias válidas, repeticiones al pedir más.
- Recetas: estructura, nº de pasos, uso de ingredientes, alérgenos mencionados, coherencia con el plato,
  fidelidad a la receta propia de referencia.
- Agente: ruta, herramientas usadas (esperadas y prohibidas), contenido de la respuesta, efecto real en los
  datos, "botones" inventados y tiempos de espera agotados.

Los datos del usuario (SQLite) se copian antes de cada caso y se restauran después: los casos pueden preparar
la despensa, guardar alergias o pedir al agente que modifique cosas sin que tus datos cambien. Mientras se
evalúa, no uses la app: verías los datos de prueba.
"""
import csv
import json
import re
import shutil
import statistics
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path

from metricas.evaluar import CARPETA, _ejecutar, _marca, _percentil

CASOS_GENERACION = CARPETA / "generacion.json"

HERRAMIENTAS_MODIFICAN = {
    "anadir_ingrediente", "actualizar_ingrediente", "consumir_ingredientes", "eliminar_ingrediente",
    "anadir_a_lista_compra", "quitar_de_lista_compra", "guardar_preferencia", "eliminar_preferencia",
    "registrar_plato_cocinado",
}
BOTONES_INVENTADOS = re.compile(r"haz\s+clic|pulsa\s+(el|en)|presiona\s+el|clica|bot[oó]n", re.IGNORECASE)
TIMEOUT_AGENTE = re.compile(r"tardando demasiado", re.IGNORECASE)


# ------------------------------------------------------------------ utilidades
def _norm(texto: str) -> str:
    texto = unicodedata.normalize("NFD", (texto or "").lower())
    return " ".join("".join(c for c in texto if unicodedata.category(c) != "Mn").split())


def _contiene(texto: str, palabra: str) -> bool:
    return _norm(palabra) in _norm(texto)


def _lista(texto: str | None) -> list[str]:
    return [i.strip() for i in (texto or "").split(",") if i.strip()]


def _media(valores: list) -> float | str:
    numeros = [v for v in valores if isinstance(v, (int, float))]
    return round(statistics.mean(numeros), 3) if numeros else ""


def _guardar_csv(ruta: Path, filas: list[dict]):
    """CSV con la unión de columnas de todas las filas (cada tipo de caso tiene las suyas)."""
    if not filas:
        return
    columnas = list(dict.fromkeys(c for f in filas for c in f))
    with ruta.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


# --------------------------------------------------- aislamiento de los datos
@contextmanager
def datos_aislados():
    """Copia la base de datos del usuario y la restaura al terminar, pase lo que pase."""
    from src.data import sqlitebd
    ruta = Path(sqlitebd.RUTA_BD)
    copia = ruta.with_name(ruta.name + ".eval_backup")
    existia = ruta.exists()
    if existia:
        shutil.copy2(ruta, copia)
    try:
        yield
    finally:
        if existia:
            shutil.copy2(copia, ruta)
            copia.unlink(missing_ok=True)
        else:
            ruta.unlink(missing_ok=True)


def _restaurar_si_quedo_copia():
    """Si una evaluación anterior se interrumpió (p. ej. Ctrl+C a destiempo), recupera los datos originales."""
    from src.data import sqlitebd
    ruta = Path(sqlitebd.RUTA_BD)
    copia = ruta.with_name(ruta.name + ".eval_backup")
    if copia.exists():
        shutil.copy2(copia, ruta)
        copia.unlink()
        print("♻️  Se han restaurado tus datos desde la copia de una evaluación anterior interrumpida.\n")


def _preparar(prep: dict):
    """Deja la despensa, la lista y las preferencias en el estado que necesita el caso."""
    from src.data import sqlitebd
    if prep.get("vaciar_lista"):
        sqlitebd.quitar_de_lista("todo")
    for nombre, cantidad, unidad in prep.get("despensa", []):
        # Cantidad exacta: se fija si existe, se crea si no
        if sqlitebd.fijar_ingrediente(nombre, cantidad, unidad).startswith("No hay"):
            sqlitebd.anadir_ingrediente(nombre, cantidad, unidad)
    for nombre in prep.get("quitar_despensa", []):
        sqlitebd.eliminar_ingrediente(nombre)
    if prep.get("lista_compra"):
        sqlitebd.anadir_a_lista(prep["lista_compra"])
    for tipo, valor in prep.get("preferencias", []):
        sqlitebd.guardar_preferencia(tipo, valor)


def _cantidad(nombre: str) -> float | None:
    from src.data import sqlitebd
    clave = sqlitebd._clave(nombre)
    for i in sqlitebd.listar_despensa(incluir_agotados=True):
        if sqlitebd._clave(i["nombre"]) == clave:
            return i["cantidad"]
    return None


def _foto_datos() -> dict:
    from src.data import sqlitebd
    return {
        "despensa": sorted((i["nombre"], i["cantidad"], i["unidad"]) for i in sqlitebd.listar_despensa(incluir_agotados=True)),
        "lista": sorted(sqlitebd.ver_lista_compra()),
        "preferencias": sqlitebd.obtener_preferencias(),
    }


def _comprobar_efectos(efectos: list[dict], antes: dict, cantidades_antes: dict) -> tuple[bool, list[str]]:
    """Comprueba en la base de datos que el agente ha hecho (o no) lo que debía."""
    from src.data import sqlitebd
    fallos = []
    despues = _foto_datos()
    for efecto in efectos:
        (tipo, valor), = efecto.items()
        if tipo == "sin_cambios":
            if despues != antes:
                fallos.append("modificó datos sin que se le pidiera")
        elif tipo == "despensa_cantidad":
            actual = _cantidad(valor["nombre"])
            if actual is None or abs(actual - valor["valor"]) > 0.01:
                fallos.append(f"{valor['nombre']}: esperado {valor['valor']}, hay {actual}")
        elif tipo == "despensa_agotado":
            actual = _cantidad(valor)
            if actual not in (None, 0, 0.0):
                fallos.append(f"{valor}: debía quedar a 0 o eliminado, hay {actual}")
        elif tipo == "despensa_aumenta":
            antes_n, actual = cantidades_antes.get(valor["nombre"]), _cantidad(valor["nombre"])
            if actual is None or antes_n is None or abs((actual - antes_n) - valor["en"]) > 0.01:
                fallos.append(f"{valor['nombre']}: debía aumentar {valor['en']}, pasó de {antes_n} a {actual}")
        elif tipo == "lista_contiene":
            if not any(_contiene(p, valor) for p in despues["lista"]):
                fallos.append(f"la lista de la compra no contiene «{valor}»")
        elif tipo == "preferencia":
            if not any(_contiene(v, valor["valor"]) for v in despues["preferencias"].get(valor["tipo"], [])):
                fallos.append(f"no se guardó {valor['tipo']}: {valor['valor']}")
        elif tipo == "historial_contiene":
            if not any(_contiene(h["plato"], valor) for h in sqlitebd.ver_historial(1)):
                fallos.append(f"el historial no contiene «{valor}»")
        else:
            fallos.append(f"efecto desconocido: {tipo}")
    return not fallos, fallos


# -------------------------------------------------------------- métricas
def _metricas_ideas(ideas: list[dict], ingredientes: list[str], evitar: list[str], minimo_locales: int) -> dict:
    from src.agents.coincidencias import contar_coincidencias
    nombres = [_norm(i.get("nombre", "")) for i in ideas]
    textos = [f"{i.get('nombre', '')}. {i.get('descripcion', '')}" for i in ideas]
    usan = [contar_coincidencias(ingredientes, t) > 0 for t in textos] if ingredientes else []
    con_alergenos = [i["nombre"] for i, t in zip(ideas, textos) if evitar and contar_coincidencias(evitar, t)]

    locales_invalidas = []
    for idea in ideas:
        if idea.get("origen") == "local" and idea.get("ref_id") is not None and ingredientes:
            try:
                from src.agents.rag_node import obtener_receta_por_id
                fila = obtener_receta_por_id(idea["ref_id"]) or {}
                if contar_coincidencias(ingredientes, fila.get("texto", "")) < minimo_locales:
                    locales_invalidas.append(idea["nombre"])
            except Exception:
                pass
    return {
        "n_ideas": len(ideas),
        "duplicadas": len(nombres) - len(set(nombres)),
        "ideas_locales": sum(i.get("origen") == "local" for i in ideas),
        "uso_ingredientes": round(sum(usan) / len(usan), 2) if usan else "",
        "ideas_con_alergenos": "; ".join(con_alergenos),
        "n_ideas_con_alergenos": len(con_alergenos),
        "locales_invalidas": "; ".join(locales_invalidas),
    }


def _ingredientes_referencia(texto: str) -> list[str]:
    """Ingredientes de una receta propia (bloque 'Ingredientes: ...' del recetario)."""
    from src.data import sqlitebd
    m = re.search(r"ingredientes\s*:?(.*?)(?:\n\s*(pasos|preparaci[oó]n|elaboraci[oó]n)\b|$)", texto or "",
                  re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    nombres = []
    for item in re.split(r"[\n,;]", m.group(1)):
        item = item.strip(" -•*\t")
        if len(item) > 2:
            analizado = sqlitebd.parsear_lista(item)
            nombres.append(analizado[0][0] if analizado else item)
    return [n for n in nombres if len(n) > 2][:30]


def _metricas_receta(texto: str, ingredientes: list[str], evitar: list[str], plato: str | None,
                     referencia: str | None) -> dict:
    from src.agents.coincidencias import contar_coincidencias
    pasos = len(re.findall(r"(?m)^\s*(?:\*\*)?(?:paso\s*)?\d+\s*[.)º:-]", texto, flags=re.IGNORECASE))
    tiene_ingredientes = "ingredientes" in _norm(texto) and bool(re.search(r"(?m)^\s*([-•*]|\d+\s)", texto))
    usados = [i for i in ingredientes if contar_coincidencias([i], texto)]
    alergenos = [e for e in evitar if contar_coincidencias([e], texto)]
    fidelidad = ""
    if referencia:
        ref = _ingredientes_referencia(referencia)
        if ref:
            fidelidad = round(sum(bool(contar_coincidencias([r], texto)) for r in ref) / len(ref), 2)
    return {
        "caracteres": len(texto),
        "estructura_ok": int(tiene_ingredientes and 1 <= pasos <= 6),
        "n_pasos": pasos,
        "uso_ingredientes": round(len(usados) / len(ingredientes), 2) if ingredientes else "",
        "alergenos_mencionados": "; ".join(alergenos),
        "coherencia_plato": (int(all(_contiene(texto, p) for p in _norm(plato).split() if len(p) > 3))
                             if plato else ""),
        "fidelidad_referencia": fidelidad,
    }


# ------------------------------------------------------------------ juez
RUBRICAS = {
    "ideas": ("Eres un evaluador experto en cocina. Puntúa de 1 a 5 esta lista de ideas de platos.",
              ["adecuacion_tipo_comida", "uso_ingredientes", "variedad", "realismo"]),
    "receta": ("Eres un evaluador experto en cocina. Puntúa de 1 a 5 esta receta.",
               ["claridad_pasos", "cantidades", "coherencia_plato", "respeto_restricciones"]),
    "agente": ("Eres un evaluador de asistentes conversacionales de cocina. Puntúa de 1 a 5 esta respuesta.",
               ["correccion", "utilidad", "concision"]),
}


class Juez:
    """Gemini como juez con una rúbrica fija. Si falla una vez, se desactiva para no frenar la evaluación."""

    def __init__(self, activo: bool):
        self.activo = activo

    def puntuar(self, tipo: str, contexto: str, respuesta: str) -> dict:
        if not self.activo:
            return {}
        intro, criterios = RUBRICAS[tipo]
        formato = "{" + ", ".join(f'"{c}": n' for c in criterios) + ', "comentario": "una frase"}'
        prompt = (f"{intro}\n\nContexto de la petición:\n{contexto}\n\nRespuesta a evaluar:\n{respuesta[:4000]}\n\n"
                  f"Criterios (1 = muy mal, 5 = excelente): {', '.join(criterios)}.\n"
                  f"Responde SOLO con JSON: {formato}")
        try:
            from src.agents.web_search_node import generar_json_gemini
            texto = generar_json_gemini(prompt)
            m = re.search(r"\{.*\}", texto, re.DOTALL)
            datos = json.loads(m.group(0) if m else texto)
            notas = {f"juez_{c}": datos.get(c) for c in criterios if isinstance(datos.get(c), (int, float))}
            if notas:
                notas["juez_media"] = round(statistics.mean(notas.values()), 2)
            notas["juez_comentario"] = str(datos.get("comentario", ""))[:300]
            return notas
        except Exception as e:
            print(f"⚠️  El juez (Gemini) no está disponible: {type(e).__name__} {e}. Se sigue sin juez.")
            self.activo = False
            return {}


# ------------------------------------------------------------- ejecución
def _payload_ideas(caso: dict) -> dict:
    base = {"mensaje": caso["mensaje"], "fase": "ideas", "tipo_comida": caso.get("tipo_comida")}
    if caso.get("ingredientes"):
        return {**base, "ingredientes_db": caso["ingredientes"]}
    return {**base, "intencion": "RECETA"}


def _evaluar_ideas(caso, url, timeout, juez, minimo_locales):
    filas, detalle = [], []
    evitar = caso.get("evitar", [])
    m = _ejecutar(url, _payload_ideas(caso), timeout)
    contexto = m.get("contexto") or {}
    ideas = contexto.get("ideas") or []
    ingredientes = _lista(caso.get("ingredientes"))
    if not ingredientes and contexto.get("ingredientes"):
        from src.agents.rag_node import limpiar_ingredientes
        ingredientes = _lista(limpiar_ingredientes(contexto["ingredientes"]))

    fila = {"fase": "ideas", "total_s": round(m["total_s"], 2), "error": m.get("error") or "",
            **_metricas_ideas(ideas, ingredientes, evitar, minimo_locales)}
    lista_txt = "\n".join(f"- {i['nombre']}: {i.get('descripcion', '')}" for i in ideas)
    fila.update(juez.puntuar("ideas", f"Petición: {caso['mensaje']}. Tipo de comida: {caso.get('tipo_comida')}. "
                                      f"Ingredientes: {', '.join(ingredientes)}. Evitar: {', '.join(evitar) or 'nada'}.",
                             lista_txt))
    filas.append(fila)
    detalle.append({"fase": "ideas", "respuesta": m.get("respuesta", ""), "ideas": ideas})

    # Más ideas: no deben repetir ninguna ya mostrada
    if caso.get("mas_ideas") and ideas:
        mas = _ejecutar(url, {"mensaje": "✨ Dame ideas más creativas", "fase": "ideas", "fuente_busqueda": "web",
                              "ingredientes_db": contexto.get("ingredientes"), "tipo_comida": contexto.get("tipo_comida"),
                              "ids_excluidos": [i["ref_id"] for i in ideas if i.get("ref_id") is not None],
                              "recetas_descartadas": [i["nombre"] for i in ideas]}, timeout)
        nuevas = (mas.get("contexto") or {}).get("ideas") or []
        vistas = {_norm(i["nombre"]) for i in ideas}
        repetidas = [i["nombre"] for i in nuevas if _norm(i["nombre"]) in vistas]
        filas.append({"fase": "mas_ideas", "total_s": round(mas["total_s"], 2), "error": mas.get("error") or "",
                      "n_ideas": len(nuevas), "repetidas": "; ".join(repetidas), "n_repetidas": len(repetidas)})
        detalle.append({"fase": "mas_ideas", "ideas": nuevas})

    # Receta de la primera idea
    if caso.get("receta") and ideas:
        idea = ideas[0]
        r = _ejecutar(url, {"mensaje": f"Quiero preparar: {idea['nombre']}", "fase": "receta", "idea": idea,
                            "ingredientes_db": contexto.get("ingredientes"), "tipo_comida": contexto.get("tipo_comida")},
                      timeout)
        referencia = None
        if idea.get("origen") == "local" and idea.get("ref_id") is not None:
            try:
                from src.agents.rag_node import obtener_receta_por_id
                referencia = (obtener_receta_por_id(idea["ref_id"]) or {}).get("texto")
            except Exception:
                pass
        texto = r.get("respuesta", "")
        fila_r = {"fase": "receta", "plato": idea["nombre"], "origen": (r.get("contexto") or {}).get("origen"),
                  "total_s": round(r["total_s"], 2),
                  "primer_token_s": round(r["primer_token_s"], 2) if r.get("primer_token_s") else "",
                  "error": r.get("error") or "",
                  **_metricas_receta(texto, ingredientes, evitar, idea["nombre"], referencia)}
        fila_r.update(juez.puntuar("receta", f"Plato: {idea['nombre']}. Tipo de comida: {caso.get('tipo_comida')}. "
                                             f"Ingredientes disponibles: {', '.join(ingredientes)}. "
                                             f"Restricciones: no puede tomar {', '.join(evitar) or 'nada'}.", texto))
        filas.append(fila_r)
        detalle.append({"fase": "receta", "respuesta": texto, "referencia": referencia})
    return filas, detalle


def _evaluar_plato(caso, url, timeout, juez):
    evitar = caso.get("evitar", [])
    r = _ejecutar(url, {"mensaje": caso["mensaje"]}, timeout)
    texto = r.get("respuesta", "")
    contexto = r.get("contexto") or {}
    fila = {"fase": "receta", "plato": caso.get("plato", ""), "origen": contexto.get("origen"),
            "es_receta": int(bool(contexto.get("es_receta"))),
            "total_s": round(r["total_s"], 2),
            "primer_token_s": round(r["primer_token_s"], 2) if r.get("primer_token_s") else "",
            "error": r.get("error") or "",
            **_metricas_receta(texto, [], evitar, caso.get("plato"), None)}
    fila.update(juez.puntuar("receta", f"Petición: {caso['mensaje']}. Restricciones: no puede tomar "
                                       f"{', '.join(evitar) or 'nada'}.", texto))
    return [fila], [{"fase": "receta", "respuesta": texto}]


def _evaluar_agente(caso, url, timeout, juez):
    nombres = [e["despensa_aumenta"]["nombre"] for e in caso.get("efectos", []) if "despensa_aumenta" in e]
    cantidades_antes = {n: _cantidad(n) for n in nombres}
    antes = _foto_datos()

    m = _ejecutar(url, {"mensaje": caso["mensaje"], "historial": caso.get("historial", [])}, timeout)
    texto = m.get("respuesta", "")
    nodos = m.get("nodos") or {}
    usadas = set((m.get("contexto") or {}).get("herramientas") or []) | {n.split(":", 1)[1] for n in nodos
                                                                       if n.startswith("herramienta:")}
    ruta = "asistente" if usadas or "agente" in nodos else (
        "ideas" if any(n in nodos for n in ("ideas_creativas", "presentar_ideas")) else
        "receta" if "generador" in nodos else "directa")

    todas = set(caso.get("herramientas_todas", []))
    alguna = set(caso.get("herramientas_alguna", []))
    prohibidas = set(caso.get("herramientas_prohibidas", []))
    if "modificadoras" in prohibidas:
        prohibidas = (prohibidas - {"modificadoras"}) | HERRAMIENTAS_MODIFICAN
    herramientas_ok = todas <= usadas and (not alguna or bool(alguna & usadas)) and not (prohibidas & usadas)
    contiene = caso.get("respuesta_contiene", [])
    # Cada elemento admite alternativas separadas por "|": "6|seis"
    faltan = [p for p in contiene if not any(_contiene(texto, alt) for alt in p.split("|"))]
    efectos_ok, fallos = _comprobar_efectos(caso.get("efectos", []), antes, cantidades_antes)

    fila = {"fase": "agente", "ruta": ruta, "ruta_ok": int(ruta == caso.get("ruta", "asistente")),
            "herramientas_usadas": "; ".join(sorted(usadas)), "herramientas_ok": int(herramientas_ok),
            "prohibidas_usadas": "; ".join(sorted(prohibidas & usadas)),
            "respuesta_ok": int(not faltan), "faltan_en_respuesta": "; ".join(faltan),
            "efectos_ok": int(efectos_ok), "fallos_efectos": "; ".join(fallos),
            "botones_inventados": int(bool(BOTONES_INVENTADOS.search(texto))),
            "timeout": int(bool(TIMEOUT_AGENTE.search(texto))),
            "total_s": round(m["total_s"], 2), "error": m.get("error") or ""}
    historial = " | ".join(f"{h['role']}: {h['content']}" for h in caso.get("historial", []))
    fila.update(juez.puntuar("agente", f"Mensaje del usuario: {caso['mensaje']}. "
                                       f"Conversación previa: {historial or 'ninguna'}. "
                                       f"Herramientas que usó el asistente: {', '.join(sorted(usadas)) or 'ninguna'}.",
                             texto))
    return [fila], [{"fase": "agente", "respuesta": texto, "herramientas": sorted(usadas)}]


# --------------------------------------------------------------- comando
def evaluar_generacion(url: str, repeticiones: int, timeout: float, con_juez: bool, solo: str | None):
    from src.config import EVAL_MIN_COINCIDENCIAS

    if not CASOS_GENERACION.exists():
        raise SystemExit(f"❌ No existe {CASOS_GENERACION}. Copia allí generacion.json con los casos.")
    casos = [c for c in json.loads(CASOS_GENERACION.read_text(encoding="utf-8")) if not solo or c["tipo"] == solo]
    _restaurar_si_quedo_copia()
    juez = Juez(con_juez)
    print(f"Evaluando {len(casos)} casos de generación contra {url} · {repeticiones} repeticiones"
          f"{' · con juez Gemini' if con_juez else ''}\n⚠️  No uses la app mientras tanto: los casos preparan "
          "datos de prueba (tus datos se restauran al acabar cada caso).\n")

    filas, detalle = [], []
    for rep in range(1, repeticiones + 1):
        for caso in casos:
            t0 = time.perf_counter()
            with datos_aislados():
                _preparar({**caso.get("preparacion", {}),
                           "preferencias": caso.get("preparacion", {}).get("preferencias", [])
                                           + [["alergia", e] for e in caso.get("evitar", [])]})
                try:
                    if caso["tipo"] == "ideas":
                        f, d = _evaluar_ideas(caso, url, timeout, juez, EVAL_MIN_COINCIDENCIAS)
                    elif caso["tipo"] == "plato":
                        f, d = _evaluar_plato(caso, url, timeout, juez)
                    else:
                        f, d = _evaluar_agente(caso, url, timeout, juez)
                except Exception as e:
                    f, d = [{"fase": caso["tipo"], "error": f"{type(e).__name__}: {e}"}], []
            for fila in f:
                filas.append({"caso": caso["id"], "tipo": caso["tipo"], "repeticion": rep, **fila})
            for x in d:
                detalle.append({"caso": caso["id"], "repeticion": rep, **x})
            print(f"  {caso['id']:24} #{rep}  {time.perf_counter() - t0:6.1f}s  " + _resumen_fila(f))

    _imprimir_resumen(filas)
    marca = _marca()
    CARPETA.mkdir(parents=True, exist_ok=True)
    _guardar_csv(CARPETA / f"generacion_{marca}.csv", filas)
    with (CARPETA / f"generacion_{marca}_detalle.jsonl").open("w", encoding="utf-8") as fichero:
        for x in detalle:
            fichero.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"\n✅ Métricas en {CARPETA}/generacion_{marca}.csv")
    print(f"   Textos completos (para revisarlos a mano) en generacion_{marca}_detalle.jsonl")


def _resumen_fila(filas: list[dict]) -> str:
    partes = []
    for f in filas:
        if f.get("error"):
            partes.append(f"❌ {f['fase']}: {f['error'][:60]}")
        elif f["fase"] in ("ideas", "mas_ideas"):
            partes.append(f"{f['fase']}: {f.get('n_ideas', 0)} ideas"
                          + (f", {f['n_ideas_con_alergenos']} con alérgenos" if f.get("n_ideas_con_alergenos") else "")
                          + (f", {f['n_repetidas']} repetidas" if f.get("n_repetidas") else ""))
        elif f["fase"] == "receta":
            partes.append(f"receta: estructura {'✅' if f.get('estructura_ok') else '❌'}, {f.get('n_pasos')} pasos"
                          + (f", menciona {f['alergenos_mencionados']}" if f.get("alergenos_mencionados") else ""))
        elif f["fase"] == "agente":
            marcas = [("ruta", f["ruta_ok"]), ("herramientas", f["herramientas_ok"]),
                      ("respuesta", f["respuesta_ok"]), ("efectos", f["efectos_ok"])]
            partes.append("agente: " + " ".join(f"{n} {'✅' if v else '❌'}" for n, v in marcas))
        if f.get("juez_media") not in (None, ""):
            partes[-1] += f" · juez {f['juez_media']}"
    return " | ".join(partes)


def _imprimir_resumen(filas: list[dict]):
    def de(fase):
        return [f for f in filas if f.get("fase") == fase and not f.get("error")]

    def pct(valores):
        valores = [v for v in valores if v != ""]
        return f"{sum(valores) / len(valores):.0%}" if valores else "-"

    def p50(valores):
        valores = [v for v in valores if isinstance(v, (int, float))]
        return f"{statistics.median(valores):.1f}s" if valores else "-"

    print("\n==================== RESUMEN DE LA GENERACIÓN ====================")
    ideas = de("ideas")
    if ideas:
        print(f"\nIDEAS ({len(ideas)})  nº medio {_media([f['n_ideas'] for f in ideas])} · "
              f"duplicadas {sum(f['duplicadas'] for f in ideas)} · usan tus ingredientes "
              f"{_media([f['uso_ingredientes'] for f in ideas]) or '-'} · con alérgenos "
              f"{sum(f['n_ideas_con_alergenos'] for f in ideas)} · locales no válidas "
              f"{sum(bool(f['locales_invalidas']) for f in ideas)} · tiempo p50 {p50([f['total_s'] for f in ideas])}"
              + (f" · juez {_media([f.get('juez_media') for f in ideas])}" if any(f.get("juez_media") for f in ideas) else ""))
    mas = de("mas_ideas")
    if mas:
        print(f"MÁS IDEAS ({len(mas)})  ideas repetidas {sum(f['n_repetidas'] for f in mas)}")
    recetas = de("receta")
    if recetas:
        print(f"\nRECETAS ({len(recetas)})  estructura correcta {pct([f['estructura_ok'] for f in recetas])} · "
              f"pasos medios {_media([f['n_pasos'] for f in recetas])} · usan tus ingredientes "
              f"{_media([f['uso_ingredientes'] for f in recetas]) or '-'} · fidelidad a la receta propia "
              f"{_media([f['fidelidad_referencia'] for f in recetas]) or '-'} · coherencia con el plato "
              f"{pct([f['coherencia_plato'] for f in recetas])} · mencionan alérgenos "
              f"{sum(bool(f['alergenos_mencionados']) for f in recetas)}\n          "
              f"tiempo p50 {p50([f['total_s'] for f in recetas])} · 1er token p50 "
              f"{p50([f['primer_token_s'] for f in recetas])}"
              + (f" · juez {_media([f.get('juez_media') for f in recetas])}" if any(f.get("juez_media") for f in recetas) else ""))
        print("          (mencionar un alérgeno puede ser un aviso de sustitución: revísalo en el detalle)")
    agente = de("agente")
    if agente:
        print(f"\nAGENTE ({len(agente)})  ruta {pct([f['ruta_ok'] for f in agente])} · herramientas "
              f"{pct([f['herramientas_ok'] for f in agente])} · respuesta {pct([f['respuesta_ok'] for f in agente])} · "
              f"efectos en los datos {pct([f['efectos_ok'] for f in agente])}\n          botones inventados "
              f"{sum(f['botones_inventados'] for f in agente)} · tiempos agotados {sum(f['timeout'] for f in agente)} · "
              f"tiempo p50 {p50([f['total_s'] for f in agente])}"
              + (f" · juez {_media([f.get('juez_media') for f in agente])}" if any(f.get("juez_media") for f in agente) else ""))
        fallidos = [f for f in agente if not (f["ruta_ok"] and f["herramientas_ok"] and f["respuesta_ok"] and f["efectos_ok"])]
        for f in fallidos:
            motivo = f.get("fallos_efectos") or f.get("faltan_en_respuesta") or f.get("prohibidas_usadas") or f.get("ruta")
            print(f"          ❌ {f['caso']}: usó [{f['herramientas_usadas'] or 'ninguna'}] · {motivo}")
    errores = [f for f in filas if f.get("error")]
    if errores:
        print(f"\n⚠️  {len(errores)} ejecuciones con error (ver la columna 'error' del CSV)")