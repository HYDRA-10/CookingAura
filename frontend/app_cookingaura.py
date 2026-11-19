import inspect
import json
import unicodedata
import os
import re
import streamlit as st
import requests

# 1. Configuración inicial
st.set_page_config(page_title="CookingAura", page_icon="👨‍🍳", layout="centered")
st.title("👨‍🍳 CookingAura")
st.caption("¡Hola! Soy tu asistente de cocina personalizado. 🍳\n")

# Ingredientes de la despensa que se muestran de entrada en la selección rápida (el resto, con el buscador)
# Ingredientes que se ofrecen de entrada en la selección rápida (el resto, con el buscador)
MAX_INGREDIENTES_RAPIDOS = int(os.getenv("DESPENSA_MAX_INGREDIENTES", "20") or 20)

# Las versiones recientes de Streamlit permiten grabar audio desde la propia barra del chat
CHAT_CON_MICRO = "accept_audio" in inspect.signature(st.chat_input).parameters

BACKEND_URL = os.getenv("BACKEND_URL", "http://cookingaura-app-backend:8080")


def formatear_ingredientes_despensa(nombres: list[str], despensa: list[dict]) -> str:
    """Convierte los ingredientes seleccionados en 'Huevos (6 unidades), Leche (1 litros)'."""
    por_nombre = {item["nombre"]: item for item in despensa}
    partes = []
    for nombre in nombres:
        item = por_nombre.get(nombre)
        if item:
            cantidad = str(item.get("cantidad") or "").strip()
            unidad = str(item.get("unidad") or "").strip()
            detalle = " ".join(p for p in (cantidad, unidad) if p and p != "None")
            partes.append(f"{nombre} ({detalle})" if detalle else nombre)
        else:
            partes.append(nombre)
    return ", ".join(partes)


def limpiar_ingredientes_manual(texto: str | None) -> str:
    """Normaliza lo escrito a mano: 'huevos; patatas\\ncebolla' -> 'huevos, patatas, cebolla'."""
    if not texto:
        return ""
    trozos = [t.strip() for t in re.split(r"[,;\n]+", texto)]
    return ", ".join(t for t in trozos if t)


def cargar_despensa():
    try:
        respuesta = requests.get(f"{BACKEND_URL}/ingredientes", timeout=10)
        if respuesta.status_code == 200:
            st.session_state.despensa = respuesta.json().get("ingredientes", [])
        else:
            st.session_state.despensa = []
            st.error("No se pudo cargar la despensa desde la base de datos.")
    except requests.exceptions.RequestException:
        st.session_state.despensa = []
        st.error("Error de conexión al cargar la despensa.")


ICONOS_DOCUMENTO = {"pdf": "📕", "docx": "📘", "xlsx": "📗", "csv": "📊"}


def cargar_documentos():
    """Pide al backend los documentos indexados y los guarda en sesión."""
    st.session_state.error_documentos = None
    try:
        r = requests.get(f"{BACKEND_URL}/recetas/documentos", timeout=30)
        if r.status_code == 200:
            st.session_state.documentos = r.json().get("documentos", [])
        else:
            st.session_state.documentos = []
            st.session_state.error_documentos = f"Error {r.status_code} al cargar los documentos."
    except requests.exceptions.RequestException:
        st.session_state.documentos = []
        st.session_state.error_documentos = "No se pudo conectar con el backend para listar los documentos."
    if not st.session_state.error_documentos:
        st.session_state.hay_documentos_milvus = bool(st.session_state.documentos)


def formatear_tamano(n: int | None) -> str | None:
    if not n:
        return None
    for unidad in ("B", "KB", "MB"):
        if n < 1024 or unidad == "MB":
            return f"{n:.0f} {unidad}" if unidad == "B" else f"{n:.1f} {unidad}"
        n /= 1024


def formatear_fecha(iso: str | None) -> str | None:
    if not iso:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(iso).strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return iso


def transcribir(audio) -> tuple[str, str | None]:
    """Envía la grabación al backend y devuelve (texto, error)."""
    try:
        r = requests.post(f"{BACKEND_URL}/transcribir",
                          files={"audio": (audio.name or "voz.wav", audio.getvalue(), audio.type or "audio/wav")},
                          timeout=180)
        if r.status_code == 200:
            return (r.json().get("texto") or "").strip(), None
        try:
            return "", f"❌ {r.json().get('detail', r.status_code)}"
        except ValueError:
            return "", f"❌ Error en el servidor: código {r.status_code}"
    except requests.exceptions.RequestException:
        return "", "❌ No se pudo conectar con el backend para transcribir."


def historial_reciente(n: int = 8) -> list[dict]:
    """Últimos mensajes del chat (para que el asistente entienda "sí", "para 6 personas"...)."""
    return [{"role": m["role"], "content": m["content"][:1500]} for m in st.session_state.messages[-n:]]


def a_numero(valor) -> float:
    try:
        return float(str(valor).replace(",", "."))
    except (TypeError, ValueError):
        return 0.0


def _sin_tildes(texto: str) -> str:
    texto = unicodedata.normalize("NFD", (texto or "").lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def _raiz(palabra: str) -> str:
    """Iguala singular y plural para buscar: huevos -> huevo, tomates -> tomat, limones -> limon."""
    if palabra.endswith("ces") and len(palabra) > 4:
        return palabra[:-3] + "z"
    if palabra.endswith("s") and len(palabra) > 3:
        palabra = palabra[:-1]
    if palabra.endswith("e") and len(palabra) > 3:
        palabra = palabra[:-1]
    return palabra


def coincide_busqueda(nombre: str, busqueda: str) -> bool:
    """Sin importar tildes, mayúsculas ni plural. Varias palabras se buscan juntas ("pechuga pollo");
    separadas por comas se busca cualquiera de ellas ("ajo, arroz")."""
    nombre = _sin_tildes(nombre)
    grupos = [g.split() for g in _sin_tildes(busqueda).split(",") if g.strip()]
    return any(all(_raiz(palabra) in nombre for palabra in grupo) for grupo in grupos)


def ingredientes_sugeridos(tipo_comida: str, nombres: list[str]) -> list[str]:
    """Ingredientes principales de la despensa adecuados al tipo de comida (los elige el backend al azar).
    Se guardan en sesión para que no cambien en cada interacción; si el backend falla, los primeros por orden."""
    clave = (tipo_comida, st.session_state.version_form)
    cache = st.session_state.setdefault("sugeridos_rapidos", {})
    if clave not in cache:
        try:
            r = requests.get(f"{BACKEND_URL}/ingredientes/principales",
                             params={"tipo_comida": tipo_comida, "cantidad": MAX_INGREDIENTES_RAPIDOS}, timeout=10)
            r.raise_for_status()
            disponibles = {n.lower(): n for n in nombres}
            cache[clave] = [disponibles[n.lower()] for n in r.json().get("ingredientes", []) if n.lower() in disponibles]
        except requests.exceptions.RequestException:
            cache[clave] = []
        if not cache[clave]:
            cache[clave] = nombres[:MAX_INGREDIENTES_RAPIDOS]
    return cache[clave]


def formatear_numero(n: float) -> str:
    """2.0 -> '2', 1.5 -> '1.5'"""
    return str(int(n)) if float(n).is_integer() else f"{n:g}"


# 2. Inicializar variables de estado (solo la primera vez, nunca se resetean en cada rerun)
valores_iniciales = {
    "messages": [],
    "tipo_comida": None,
    "chat_bloqueado": False,
    "peticion_pendiente": None,   # payload completo que se enviará al backend tras el rerun
    "ultimas_ideas": None,        # ideas de platos propuestas y su contexto (ingredientes, tipo...)
    "version_ideas": 0,           # para que cada lista de ideas tenga claves de botón propias
    "modo_voz": False,            # grabador de voz visible en el chat
    "version_audio": 0,           # se incrementa para vaciar el grabador tras cada envío
    "revisar_voz": True,          # mostrar lo dictado para corregirlo antes de enviarlo
    "dictado_pendiente": None,    # texto transcrito esperando revisión
    "version_form": 0,            # se incrementa para limpiar los widgets de selección rápida
    "hay_documentos_milvus": False,
}
for clave, valor in valores_iniciales.items():
    if clave not in st.session_state:
        st.session_state[clave] = valor

if "despensa" not in st.session_state:
    cargar_despensa()

# Al abrir la app se consulta si Milvus ya tiene recetas (antes solo se activaba tras subir una)
if "estado_milvus_comprobado" not in st.session_state:
    try:
        r = requests.get(f"{BACKEND_URL}/estado_milvus", timeout=10)
        st.session_state.hay_documentos_milvus = r.ok and r.json().get("hay_documentos", False)
    except requests.exceptions.RequestException:
        st.session_state.hay_documentos_milvus = False
    st.session_state.estado_milvus_comprobado = True

# 3. Creación de las Pestañas (Tabs)
tab_chat, tab_despensa, tab_recetas = st.tabs(["💬 Chatbot", "🥫 Mi Despensa", "🍽️ Recetas"])

# ==========================================
# PESTAÑA 1: CHATBOT Y SELECCIÓN RÁPIDA
# ==========================================
with tab_chat:

    st.markdown("### 💬 Chat con CookingAura")

    chat_container = st.container(height=400)
    bloqueado = st.session_state.chat_bloqueado
    version = st.session_state.version_form

    def encolar(peticion: dict):
        """Guarda la petición, bloquea la UI y relanza: se envía en la siguiente ejecución."""
        st.session_state.peticion_pendiente = peticion
        st.session_state.chat_bloqueado = True
        st.rerun()

    with chat_container:
        with st.chat_message("assistant"):
            st.markdown("¡Hola! Soy CookingAura, tu asistente de cocina. 👨‍🍳\n\n"
                        "Pregúntame qué cocinar y te propondré varias ideas: elige una y te preparo la receta. "
                        "También puedes hablarme con total libertad, por ejemplo:\n"
                        "- *«¿Me quedan huevos?»* o *«He comprado 2 kg de patatas»*\n"
                        "- *«He hecho la tortilla, descuenta los ingredientes»*\n"
                        "- *«Soy alérgico al marisco»* o *«No me gusta el cilantro»*\n"
                        "- *«Añade leche y tomates a la lista de la compra»*\n"
                        "- *«¿Qué cociné esta semana?»* o *«¿Tengo alguna receta de lentejas guardada?»* 😋")

        with st.expander("🥕 Selección rápida", expanded=False):
            st.write("### ¿Por dónde empezamos? 🤗")
            col1, col2, col3 = st.columns(3)
            if col1.button("🌅 DESAYUNO", use_container_width=True, disabled=bloqueado):
                st.session_state.tipo_comida = "Desayunar"
            if col2.button("🍱 COMIDA", use_container_width=True, disabled=bloqueado):
                st.session_state.tipo_comida = "Comer"
            if col3.button("🌙 CENA", use_container_width=True, disabled=bloqueado):
                st.session_state.tipo_comida = "Cenar"

            if st.session_state.tipo_comida:
                st.success(f"Has seleccionado: **{st.session_state.tipo_comida}**")
                st.write("### ¿Qué ingredientes utilizamos? 🥤🥗🍔🍗🍟🥓")

                # Solo se ofrecen ingredientes con existencias
                despensa = [i for i in st.session_state.get("despensa", []) if str(i.get("cantidad", "")).strip() != "0"]

                seleccionados = []
                if despensa:
                    nombres = sorted(dict.fromkeys(item["nombre"] for item in despensa), key=str.lower)
                    # La selección se guarda aparte: así se conserva aunque cambie lo que muestra el buscador
                    seleccion = [n for n in st.session_state.get("seleccion_rapida", []) if n in nombres]

                    busqueda_rapida = st.text_input(
                        "🔎 Busca entre todos tus ingredientes",
                        placeholder="Ej: pollo · o varios separados por comas: ajo, arroz",
                        key=f"buscar_rapida_{version}",
                        disabled=bloqueado,
                    )
                    if busqueda_rapida.strip():
                        candidatos = [n for n in nombres if coincide_busqueda(n, busqueda_rapida)]
                        aviso = (f"{len(candidatos)} de {len(nombres)} ingredientes coinciden con «{busqueda_rapida.strip()}»"
                                 if candidatos else f"Ningún ingrediente coincide con «{busqueda_rapida.strip()}».")
                    else:
                        # De entrada, los principales y adecuados al tipo de comida (sin sal, aceite, especias...)
                        candidatos = ingredientes_sugeridos(st.session_state.tipo_comida, nombres)
                        aviso = (f"Sugeridos para {st.session_state.tipo_comida.lower()}: {len(candidatos)} de "
                                 f"{len(nombres)} ingredientes. Usa el buscador para encontrar el resto."
                                 if len(nombres) > len(candidatos) else "")

                    # Los ya elegidos siempre visibles (primero), para poder quitarlos aunque no salgan en la búsqueda
                    visibles = list(dict.fromkeys(seleccion + candidatos))
                    if visibles:
                        # Opciones distintas -> widget nuevo, que arranca con la selección guardada
                        clave_pills = f"pills_{version}_{abs(hash(tuple(visibles))) % 10**8}"
                        elegidos = st.pills(
                            "Selecciona ingredientes de tu despensa:",
                            options=visibles,
                            default=seleccion,
                            selection_mode="multi",
                            key=clave_pills,
                            disabled=bloqueado,
                        ) or []
                        st.session_state.seleccion_rapida = list(elegidos)
                        seleccionados = list(elegidos)
                    if aviso:
                        st.caption(aviso)
                    if seleccionados:
                        st.caption(f"✅ {len(seleccionados)} seleccionados: {', '.join(seleccionados)}")

                # Si no se selecciona nada de la despensa, se pueden escribir a mano
                manual = ""
                if not seleccionados:
                    manual = st.text_input(
                        "...o escríbelos manualmente (separados por comas):",
                        placeholder="Ej: huevos, patatas, cebolla",
                        key=f"manual_{version}",
                        disabled=bloqueado,
                    )
                    if despensa:
                        st.caption(f"Si no eliges ni escribes nada, usaré hasta {MAX_INGREDIENTES_RAPIDOS} ingredientes "
                                   f"principales de tu despensa adecuados para {st.session_state.tipo_comida.lower()}.")
                    else:
                        st.caption("Tu despensa está vacía: si no escribes nada, improvisaré con ingredientes básicos.")

                if st.button("💡 Dame ideas", type="primary", use_container_width=True, disabled=bloqueado):
                    comida = st.session_state.tipo_comida.lower()
                    tipo_comida = st.session_state.tipo_comida   # el backend lo normaliza
                    ingredientes_db = formatear_ingredientes_despensa(seleccionados, despensa) if seleccionados else ""
                    ingredientes_manual = limpiar_ingredientes_manual(manual) if not seleccionados else ""

                    # Una sola fuente de ingredientes:
                    #  - seleccionados -> ingredientes_db (el grafo entra por 'mcp' y los usa tal cual)
                    #  - manuales      -> ingredientes_manual (se validan en 'evaluar_ingredientes')
                    #  - ninguno       -> intencion RECETA (ingredientes principales de la despensa para ese
                    #                     tipo de comida; si está vacía, improvisa)
                    base = {"fase": "ideas", "tipo_comida": tipo_comida,
                            "intencion": None, "ingredientes_db": None, "ingredientes_manual": None}
                    if ingredientes_db:
                        peticion = {**base, "mensaje": f"¿Qué puedo {comida} con: {ingredientes_db}?",
                                    "ingredientes_db": ingredientes_db}
                    elif ingredientes_manual:
                        peticion = {**base, "mensaje": f"¿Qué puedo {comida} con: {ingredientes_manual}?",
                                    "ingredientes_manual": ingredientes_manual}
                    elif despensa:
                        peticion = {**base, "mensaje": f"¿Qué puedo {comida} con lo que tengo en la despensa?",
                                    "intencion": "RECETA"}
                    else:
                        peticion = {**base, "mensaje": f"No tengo ingredientes, ¿qué puedo {comida}? Improvisa 🙂",
                                    "intencion": "RECETA"}

                    st.session_state.tipo_comida = None
                    st.session_state.version_form += 1   # limpia pills, buscador y texto manual
                    st.session_state.seleccion_rapida = []
                    encolar(peticion)

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # --- Ideas propuestas: elegir una para la receta o pedir más ---
        ctx = st.session_state.ultimas_ideas
        if ctx and not bloqueado:
            with st.container(border=True):
                st.markdown("**¿Te apetece otra de estas ideas?**" if ctx["elegidas"]
                            else "**Elige un plato y te preparo la receta:**")
                idea_elegida = None
                for idea in ctx["ideas"]:
                    icono = "📚" if idea.get("origen") == "local" else "✨"
                    hecha = " ✅" if idea["id"] in ctx["elegidas"] else ""
                    if st.button(f"{icono} {idea['nombre']}{hecha}", key=f"idea_{ctx['version']}_{idea['id']}",
                                 use_container_width=True, help=idea.get("descripcion") or None):
                        idea_elegida = idea

                st.caption("¿Ninguna te convence?")
                c_local, c_web = st.columns(2)
                mas_locales = c_local.button(
                    "📚 Más de mis recetas", use_container_width=True,
                    disabled=not st.session_state.hay_documentos_milvus,
                    help="Otras ideas sacadas de tus documentos, sin repetir."
                         if st.session_state.hay_documentos_milvus
                         else "Sube recetas en la pestaña Recetas para usar esta opción.",
                )
                mas_creativas = c_web.button("✨ Ideas más creativas", use_container_width=True,
                                             help="Platos nuevos inventados con los mismos ingredientes.")

            # Sin ingredientes en el contexto (improvisado): se relee la despensa con intencion RECETA
            ingredientes_ctx = ({"intencion": None, "ingredientes_db": ctx["ingredientes"]} if ctx["ingredientes"]
                                else {"intencion": "RECETA", "ingredientes_db": None})

            if idea_elegida:
                encolar({
                    "mensaje": f"👉 Quiero preparar: {idea_elegida['nombre']}",
                    "fase": "receta",
                    "idea": idea_elegida,
                    "intencion": None,
                    "ingredientes_db": ctx["ingredientes"],
                    "ingredientes_manual": None,
                    "tipo_comida": ctx["tipo_comida"],
                    "idea_id": idea_elegida["id"],          # solo para el frontend
                })
            if mas_locales or mas_creativas:
                encolar({
                    "mensaje": "🔄 Más ideas de mis recetas" if mas_locales else "✨ Dame ideas más creativas",
                    "fase": "ideas",
                    "fuente_busqueda": "local" if mas_locales else "web",
                    **ingredientes_ctx,
                    "ingredientes_manual": None,
                    "tipo_comida": ctx["tipo_comida"],
                    "ids_excluidos": ctx["ids_excluidos"],
                    "recetas_descartadas": ctx["nombres_mostrados"],
                    "mas_ideas": True,                       # solo para el frontend
                })

    def enviar_chat(texto: str, de_voz: bool = False):
        """Encola un mensaje del chat libre (escrito o dictado)."""
        encolar({"mensaje": texto, "mensaje_visible": f"🎙️ {texto}" if de_voz else texto, "fase": "ideas",
                 "intencion": None, "ingredientes_db": None, "ingredientes_manual": None,
                 "tipo_comida": None,                       # el backend deduce el tipo del texto
                 "historial": historial_reciente()})

    def recibir_dictado(texto_voz: str, texto_escrito: str = ""):
        """Lo transcrito se envía directamente o se muestra para corregirlo, según la opción elegida."""
        completo = f"{texto_escrito} {texto_voz}".strip()
        if st.session_state.revisar_voz:
            st.session_state.dictado_pendiente = completo
            st.session_state.version_audio += 1      # caja de revisión nueva con el texto recién dictado
            st.rerun()
        enviar_chat(completo, de_voz=True)

    if st.session_state.get("aviso_voz"):
        st.warning(st.session_state.pop("aviso_voz"))

    # --- Revisión de lo dictado antes de enviarlo ---
    if st.session_state.dictado_pendiente is not None and not bloqueado:
        with st.container(border=True):
            st.caption("🎙️ Esto es lo que he entendido. Corrígelo si hace falta y envíalo:")
            revisado = st.text_input("Texto dictado", value=st.session_state.dictado_pendiente,
                                     key=f"dictado_{st.session_state.version_audio}", label_visibility="collapsed")
            c_enviar, c_descartar = st.columns(2)
            if c_enviar.button("📨 Enviar", type="primary", use_container_width=True):
                st.session_state.dictado_pendiente = None
                st.session_state.version_audio += 1
                if revisado.strip():
                    enviar_chat(revisado.strip(), de_voz=True)
                st.rerun()
            if c_descartar.button("✖ Descartar", use_container_width=True):
                st.session_state.dictado_pendiente = None
                st.session_state.version_audio += 1
                st.rerun()

    st.toggle("✏️ Revisar lo dictado antes de enviarlo", key="revisar_voz", disabled=bloqueado,
              help="Muestra la transcripción para corregirla antes de mandarla al asistente.")

    if CHAT_CON_MICRO:
        # Micrófono dentro de la barra de escritura, a la izquierda del botón de enviar
        envio = st.chat_input("Escribe o pulsa el micro para hablar · ¿Qué como hoy?", disabled=bloqueado,
                              accept_audio=True, audio_sample_rate=16000)
        if envio and not bloqueado:
            texto = (envio.text or "").strip()
            if envio.audio is not None:
                with st.spinner("🎧 Transcribiendo..."):
                    texto_voz, error_voz = transcribir(envio.audio)
                if texto_voz:
                    # Si además se escribió algo, va junto con lo dictado
                    recibir_dictado(texto_voz, texto)
                st.session_state.aviso_voz = error_voz or "🤔 No he entendido nada. Prueba a hablar más cerca del micrófono."
                st.rerun()
            elif texto:
                enviar_chat(texto)
    else:
        # Streamlit sin micro en el chat: botón aparte con grabador
        col_voz, col_aviso = st.columns([1, 2], vertical_alignment="center")
        etiqueta_voz = "⌨️ Volver a escribir" if st.session_state.modo_voz else "🎙️ Hablar con el asistente"
        if col_voz.button(etiqueta_voz, disabled=bloqueado, use_container_width=True):
            st.session_state.modo_voz = not st.session_state.modo_voz
            st.rerun()

        if st.session_state.modo_voz:
            grabador = getattr(st, "audio_input", None) or getattr(st, "experimental_audio_input", None)
            if grabador is None:
                st.warning("Tu versión de Streamlit no permite grabar audio: actualízala.")
            else:
                col_aviso.caption("Pulsa el micrófono, habla y vuelve a pulsarlo para enviar.")
                audio = grabador("Tu mensaje de voz", key=f"audio_{st.session_state.version_audio}",
                                 disabled=bloqueado, label_visibility="collapsed")
                if audio is not None and not bloqueado:
                    with st.spinner("🎧 Transcribiendo..."):
                        texto_voz, error_voz = transcribir(audio)
                    st.session_state.version_audio += 1          # el grabador vuelve a empezar vacío
                    if texto_voz:
                        recibir_dictado(texto_voz)
                    st.session_state.aviso_voz = error_voz or "🤔 No he entendido nada. Prueba a hablar más cerca del micrófono."
                    st.rerun()

        # La barra de input se coloca FUERA y DESPUÉS del contenedor
        prompt = st.chat_input("¿Qué como hoy?", disabled=bloqueado)
        if prompt and not bloqueado:
            enviar_chat(prompt)

    peticion = st.session_state.peticion_pendiente
    if peticion:
        st.session_state.peticion_pendiente = None
        mensaje_visible = peticion.get("mensaje_visible", peticion["mensaje"])
        st.session_state.messages.append({"role": "user", "content": mensaje_visible})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(mensaje_visible)

            with st.chat_message("assistant"):
                estado_ph = st.empty()     # progreso del grafo hasta que llega el primer token
                estado_ph.caption("🔎 Pensando con aura... 😎")
                payload = {k: v for k, v in peticion.items() if k not in ("mas_ideas", "idea_id", "mensaje_visible")}
                contexto, final, pasos = {}, {}, []

                def leer_stream():
                    """Lee el NDJSON de /chat/stream: actualiza el progreso y va soltando el texto."""
                    with requests.post(f"{BACKEND_URL}/chat/stream", json=payload,
                                       stream=True, timeout=(10, 600)) as r:
                        if r.status_code != 200:
                            final["error"] = f"❌ Error en el servidor: código {r.status_code}"
                            return
                        r.encoding = "utf-8"
                        for linea in r.iter_lines(chunk_size=None, decode_unicode=True):
                            if not linea:
                                continue
                            evento = json.loads(linea)
                            tipo = evento.get("tipo")
                            if tipo == "estado":
                                segundos = evento.get("segundos")
                                pasos.append(f"{evento['texto']} ({segundos:.1f} s)" if segundos is not None
                                             else evento["texto"])
                                estado_ph.caption(" → ".join(pasos) + " …")
                            elif tipo == "token":
                                estado_ph.empty()
                                yield evento["texto"]
                            elif tipo == "fin":
                                final["respuesta"] = evento.get("respuesta", "")
                                contexto.update(evento.get("contexto") or {})
                            elif tipo == "error":
                                final["error"] = f"❌ {evento.get('mensaje', 'Error desconocido')}"

                texto_mostrado = ""
                try:
                    texto_mostrado = st.write_stream(leer_stream())
                except requests.exceptions.Timeout:
                    final["error"] = "⏳ El asistente ha tardado demasiado en responder. Inténtalo de nuevo."
                except requests.exceptions.RequestException:
                    final["error"] = "❌ Error de conexión con el agente. Asegúrate de que FastAPI está en ejecución."
                finally:
                    st.session_state.chat_bloqueado = False
                    estado_ph.empty()

                respuesta = final.get("respuesta") or (texto_mostrado if isinstance(texto_mostrado, str) else "")
                if final.get("error"):
                    respuesta = f"{respuesta}\n\n{final['error']}" if respuesta else final["error"]
                if not respuesta:
                    respuesta = "No he podido generar una respuesta."

        st.session_state.messages.append({"role": "assistant", "content": respuesta})

        # --- Actualizar el contexto de ideas ---
        anterior = st.session_state.ultimas_ideas
        if contexto.get("ideas"):
            # Nueva lista de ideas (pregunta nueva o "más ideas")
            acumular = peticion.get("mas_ideas") and anterior
            ideas = contexto["ideas"]
            st.session_state.version_ideas += 1
            st.session_state.ultimas_ideas = {
                "ideas": ideas,
                "ingredientes": contexto.get("ingredientes"),
                "tipo_comida": contexto.get("tipo_comida"),
                "elegidas": [],
                "version": st.session_state.version_ideas,
                # Lo ya propuesto no se repite al pedir más ideas
                "ids_excluidos": (anterior["ids_excluidos"] if acumular else [])
                                 + [i["ref_id"] for i in ideas if i.get("ref_id") is not None],
                "nombres_mostrados": ((anterior["nombres_mostrados"] if acumular else [])
                                      + [i["nombre"] for i in ideas])[-20:],
            }
        elif peticion.get("fase") == "receta" and anterior:
            # Receta generada: se marca la idea y la lista sigue disponible para elegir otra
            if contexto.get("es_receta") and peticion.get("idea_id") not in anterior["elegidas"]:
                anterior["elegidas"].append(peticion["idea_id"])
        elif not peticion.get("mas_ideas") and not contexto.get("agente"):
            # Pregunta nueva sin ideas (saludo, error...): se cierra la lista anterior.
            # Una conversación con el asistente ("¿alguna sin gluten?") no la cierra.
            st.session_state.ultimas_ideas = None

        if contexto.get("despensa_modificada"):
            st.session_state.pop("despensa", None)        # se recarga al relanzar la página
        # Si "más ideas" no encontró nada, se conserva la lista para poder probar la otra opción

        st.rerun()   # vuelve a pintar todo con los widgets ya desbloqueados

# ==========================================
# PESTAÑA 2: GESTIÓN DE LA DESPENSA
# ==========================================
with tab_despensa:
    st.header("Tus ingredientes disponibles")
    st.write("Aquí puedes ver los alimentos que el asistente tomará en cuenta al sugerirte recetas.")
    
    # Utilizamos directamente la lista cargada en el estado de sesión
    ingredientes = st.session_state.get("despensa", [])
    opciones_medida = ["g", "kg", "unidades", "litros", "ml", "botella", "paquete", "taza", "sobre", "onzas", "bote", "garrafa", "ud", "bandeja", "malla", "caja", "brik", "lata", "bolsa", "manojo", "cuña", "tarrina"]

    if st.session_state.get("aviso_despensa"):
        st.toast(st.session_state.pop("aviso_despensa"))

    if not ingredientes:
        st.info("Tu despensa está vacía.")
    else:
        disponibles = sum(a_numero(i["cantidad"]) > 0 for i in ingredientes)
        st.caption(f"{len(ingredientes)} ingredientes · {disponibles} disponibles · "
                   f"{len(ingredientes) - disponibles} agotados")

        nombres = [i["nombre"].lower() for i in ingredientes]
        repetidos = sorted({n for n in nombres if nombres.count(n) > 1})
        if repetidos:
            st.warning(f"Hay ingredientes repetidos: {', '.join(n.capitalize() for n in repetidos)}. "
                       "Puedes eliminar el que sobre o ajustar sus cantidades.")

        # Buscador por nombre: sin tildes ni mayúsculas, y "huevos" encuentra "huevo"
        busqueda = st.text_input("🔎 Buscar ingrediente", placeholder="Ej: pollo · o varios separados por comas: ajo, arroz",
                                 key="buscar_despensa")
        ordenados = sorted(ingredientes, key=lambda i: i["nombre"].lower())
        if busqueda.strip():
            ordenados = [i for i in ordenados if coincide_busqueda(i["nombre"], busqueda)]
            if ordenados:
                st.caption(f"{len(ordenados)} de {len(ingredientes)} ingredientes coinciden con «{busqueda.strip()}»")
            else:
                st.info(f"No hay ningún ingrediente que coincida con «{busqueda.strip()}».")

        # Rejilla de 3 columnas: los ingredientes se reparten por filas de 3
        for inicio in range(0, len(ordenados), 3):
            columnas = st.columns(3)
            for columna, item in zip(columnas, ordenados[inicio:inicio + 3]):
                nombre = item["nombre"]
                cantidad = a_numero(item["cantidad"])
                unidad = item.get("unidad") or ""
                categoria = item.get("categoria") or "Otros"
                # Clave única: el id de SQLite (dos ingredientes pueden llamarse igual). Sin id, nombre + posición.
                id_ing = item.get("id")
                clave = f"id{id_ing}" if id_ing is not None else f"{nombre.lower()}_{inicio}_{columnas.index(columna)}"

                with columna.container(border=True):
                    icono = "❌" if cantidad <= 0 else "✅"
                    st.markdown(f"**{icono} {nombre}**")
                    st.caption(f"{formatear_numero(cantidad)} {unidad} · {categoria.capitalize()}")

                    c_cant, c_unidad = st.columns(2)
                    entero = cantidad.is_integer()
                    nueva_cantidad = c_cant.number_input(
                        "Cantidad", min_value=0.0, value=float(cantidad), step=1.0 if entero else 0.5,
                        format="%.0f" if entero else "%.2f", key=f"cant_{clave}", label_visibility="collapsed",
                    )
                    indice = opciones_medida.index(unidad) if unidad in opciones_medida else 0
                    nueva_unidad = c_unidad.selectbox("Unidad", opciones_medida, index=indice,
                                                      key=f"unidad_{clave}", label_visibility="collapsed")

                    c_act, c_del = st.columns(2)
                    if c_act.button("💾 Actualizar", key=f"act_{clave}", use_container_width=True):
                        if nueva_cantidad == cantidad and nueva_unidad == unidad:
                            st.toast(f"'{nombre}' no tiene cambios.")
                        else:
                            payload = {"nombre": nombre, "cantidad": formatear_numero(nueva_cantidad),
                                       "categoria": categoria, "unidad": nueva_unidad, "id": id_ing}
                            try:
                                res = requests.put(f"{BACKEND_URL}/actualizar_ingrediente", json=payload, timeout=10)
                                if res.status_code == 200 and "error" not in res.json():
                                    st.session_state.aviso_despensa = (f"💾 '{nombre}' actualizado a "
                                                                       f"{formatear_numero(nueva_cantidad)} {nueva_unidad}.")
                                    del st.session_state.despensa
                                    st.rerun()
                                else:
                                    st.error("Error al actualizar en la base de datos.")
                            except requests.exceptions.RequestException:
                                st.error("Error de conexión al actualizar.")

                    with c_del.popover("🗑️ Eliminar", use_container_width=True):
                        st.write(f"¿Eliminar **{nombre}** de la despensa?")
                        if st.button("Sí, eliminar", key=f"del_{clave}", type="primary"):
                            try:
                                params = {"nombre": nombre, **({"id": id_ing} if id_ing is not None else {})}
                                res = requests.delete(f"{BACKEND_URL}/ingrediente", params=params, timeout=10)
                                if res.status_code == 200:
                                    st.session_state.aviso_despensa = f"🗑️ '{nombre}' eliminado de la despensa."
                                    del st.session_state.despensa
                                    st.rerun()
                                else:
                                    st.error("No se pudo eliminar el ingrediente.")
                            except requests.exceptions.RequestException:
                                st.error("Error de conexión al eliminar.")

    st.divider()
    
    # Añadir nuevos ingredientes manualmente
    st.subheader("Añadir nuevo ingrediente a la despensa")
    st.divider()

    with st.form("formulario_nuevo_ingrediente", clear_on_submit=True):
        col_nom, col_cant, col_unidad, col_cat = st.columns(4)
        st.session_state.existe = False
        
        with col_nom:
            st.session_state.nuevo_ingrediente = st.text_input("Ingrediente", placeholder="Tomate, Lechuga, ..")
            if st.session_state.nuevo_ingrediente:
                # Verificamos si el nombre existe dentro de la lista de listas
                nombres_existentes = [item["nombre"].lower() for item in ingredientes]
                if st.session_state.nuevo_ingrediente.strip().lower() in nombres_existentes:
                    st.warning("Este ingrediente ya está en tu despensa.")
                    st.session_state.existe = True

        with col_cant:
            st.session_state.nueva_cantidad = st.number_input("Cantidad:", placeholder="Ej: 4", min_value=0, step=1)

        with col_unidad:
            st.session_state.nueva_unidad_medida = st.selectbox("Unidad de medida:", ["g", "kg", "unidades", "litros", "ml", "botella", "paquete", "taza", "brick", "sobre", "onzas"], index=0, placeholder="Ej: unidades")
            
        with col_cat:
            categorias_alimentos = [
                "Frutas", "Verduras y Hortalizas", "Cereales y Tubérculos",
                "Legumbres", "Carnes y Aves", "Pescados y Mariscos",
                "Lácteos y Derivados", "Huevos", "Frutos secos y Semillas",
                "Grasas y Aceites", "Especias y Condimentos", "Dulces y Repostería",
                "Salsas y Aderezos", "Bebidas", "Alimentos procesados y Conservas",
                "Alternativas vegetales (Veganas)", "Panadería y Bollería", "Cafés e Infusiones", "Otros"
            ]
            st.session_state.nueva_categoria = st.selectbox(
                "Categoría", 
                categorias_alimentos,
                placeholder="Ej: Verduras y Hortalizas"
            )
            anadir = st.form_submit_button("➕ Añadir", type="primary", use_container_width=True)
    
    # Botón para accionar la escritura en el Backend
    if not st.session_state.existe and anadir:
        # Aseguramos pasar a string antes de hacer un strip
        if st.session_state.nuevo_ingrediente.strip() and str(st.session_state.nueva_cantidad).strip():
            payload = {
                "nombre": st.session_state.nuevo_ingrediente.capitalize().strip(),
                "cantidad": str(st.session_state.nueva_cantidad),
                "categoria": st.session_state.nueva_categoria,
                "unidad": st.session_state.nueva_unidad_medida.strip()
            }
            
            try:
                respuesta = requests.post(f"{BACKEND_URL}/nuevo_ingrediente", json=payload)
                if respuesta.status_code == 200:
                    st.success(f"✅ '{st.session_state.nuevo_ingrediente.capitalize()}' añadido a la base de datos.")
                    del st.session_state.despensa
                    st.rerun()
                else:
                    st.error("Error al guardar en el servidor.")
            except requests.exceptions.ConnectionError:
                st.error("❌ No se pudo conectar con el backend de FastAPI.")
        else:
            st.warning("⚠️ Debes rellenar al menos el nombre y la cantidad para guardarlo.")

# ==========================================
# PESTAÑA 3: RECETAS
# ==========================================
with tab_recetas:
    st.header("Tus recetas disponibles")
    st.write("Aquí puedes ver las recetas que el asistente tomará en cuenta al sugerirte.")

    col_busqueda, col_upload = st.columns([1, 1])

    # --- SECCIÓN DE BÚSQUEDA ---
    with col_busqueda:
        st.subheader("🔍 Buscar Recetas")
        sin_documentos = not st.session_state.hay_documentos_milvus

        if sin_documentos:
            st.warning("⚠️ Sube al menos un documento a la derecha para habilitar la búsqueda.")

        modos = {
            "Híbrida": "hibrida",
            "Semántica": "semantica",
            "Léxica (BM25)": "lexica",
        }
        etiqueta_modo = st.radio(
            "Tipo de búsqueda:", list(modos), horizontal=True, disabled=sin_documentos,
            help="Híbrida combina significado y palabras clave. Semántica entiende el contexto "
                 "('algo ligero con pollo'). Léxica busca las palabras exactas ('garbanzos').",
        )
        modo = modos[etiqueta_modo]
        query = st.text_input("¿Qué te apetece buscar?", placeholder="Ej: Algo ligero con pollo",
                              disabled=sin_documentos)

        ranker, peso_denso = "rrf", 0.7
        with st.expander("⚙️ Ajustes de búsqueda"):
            top_k = st.slider("Número de resultados", 1, 10, 5, disabled=sin_documentos)
            if modo == "hibrida":
                etiqueta_ranker = st.radio(
                    "Cómo combinar los resultados:",
                    ["RRF (por posición)", "Ponderada (por puntuación)"],
                    disabled=sin_documentos,
                    help="RRF premia lo que aparece arriba en ambas búsquedas y no necesita ajuste. "
                         "Ponderada te deja decidir cuánto pesa cada tipo de búsqueda.",
                )
                if etiqueta_ranker.startswith("Ponderada"):
                    ranker = "ponderado"
                    peso_denso = st.slider("Peso de la búsqueda semántica", 0.0, 1.0, 0.7, 0.05,
                                           disabled=sin_documentos,
                                           help="1 = solo semántica · 0 = solo palabras clave")
                    st.caption(f"Semántica {peso_denso:.0%} · Léxica {1 - peso_denso:.0%}")

        if st.button("Buscar en Milvus", use_container_width=True, disabled=sin_documentos):
            if not query.strip():
                st.warning("Escribe qué quieres buscar.")
            else:
                with st.spinner("Buscando en el espacio vectorial..."):
                    try:
                        res = requests.get(
                            f"{BACKEND_URL}/recetas/buscar",
                            params={"query": query, "modo": modo, "top_k": top_k,
                                    "ranker": ranker, "peso_denso": peso_denso},
                            timeout=120,
                        )
                        if res.status_code == 200:
                            st.session_state.resultados_busqueda = res.json().get("resultados", [])
                        else:
                            try:
                                detalle = res.json().get("detail", res.status_code)
                            except ValueError:
                                detalle = res.status_code
                            st.error(f"Error en la búsqueda: {detalle}")
                    except requests.exceptions.RequestException:
                        st.error("❌ No se pudo conectar con el backend.")

        resultados = st.session_state.get("resultados_busqueda")
        if resultados is not None:
            if not resultados:
                st.info("No se encontraron recetas para esa búsqueda.")
            for i, r in enumerate(resultados, start=1):
                texto = (r.get("texto") or "").strip()
                titulo = texto.splitlines()[0][:60] if texto else "Receta"
                with st.expander(f"{i}. {titulo}"):
                    st.caption(f"📄 {r.get('fuente') or 'Sin fuente'} · Puntuación: {r.get('score')}")
                    st.markdown(texto)
            if resultados:
                st.caption("Las puntuaciones solo sirven para comparar resultados de una misma búsqueda: "
                           "cada modo usa una escala distinta.")

    # --- SECCIÓN DE INDEXACIÓN (SUBIDA DE ARCHIVOS) ---
    with col_upload:
        st.subheader("📄 Añadir Nuevas Recetas")
        st.info("Sube documentos para indexarlos automáticamente en Milvus: PDF o Word (tablas incluidas), "
                "o un recetario en Excel/CSV con una receta por fila (columnas como Nombre, Ingredientes, Pasos).")
        archivo_subido = st.file_uploader("Selecciona un documento", type=["pdf", "docx", "xlsx", "csv", "txt", "md"])

        if st.button("Vectorizar e Indexar", type="primary", use_container_width=True):
            if archivo_subido:
                with st.spinner("Troceando el documento y generando embeddings..."):
                    archivos = {"archivo": (archivo_subido.name, archivo_subido.getvalue(), archivo_subido.type)}
                    try:
                        res = requests.post(f"{BACKEND_URL}/recetas/upload", files=archivos, timeout=600)
                        if res.status_code == 200:
                            datos = res.json()
                            st.success(datos.get("mensaje"))
                            for aviso in datos.get("avisos") or []:
                                st.warning(aviso)
                            tiempos = datos.get("tiempos") or {}
                            if tiempos:
                                st.caption("⏱ " + " · ".join(f"{fase} {seg:.1f} s" for fase, seg in tiempos.items()))
                            st.session_state.hay_documentos_milvus = True
                            st.session_state.documentos = None   # la lista de abajo se recarga
                            st.balloons()
                        else:
                            try:
                                detalle = res.json().get("detail", "Desconocido")
                            except ValueError:
                                detalle = res.status_code
                            st.error(f"Error al indexar: {detalle}")
                    except requests.exceptions.RequestException:
                        st.error("❌ No se pudo conectar con el backend.")
            else:
                st.warning("Selecciona un archivo primero.")

    # --- DOCUMENTOS GUARDADOS ---
    st.divider()
    cab_titulo, cab_boton = st.columns([4, 1], vertical_alignment="bottom")
    cab_titulo.subheader("🗂️ Documentos guardados")
    if cab_boton.button("🔄 Actualizar", use_container_width=True):
        st.session_state.documentos = None

    if st.session_state.get("documentos") is None:
        cargar_documentos()
    documentos = st.session_state.documentos or []

    if st.session_state.get("error_documentos"):
        st.error(st.session_state.error_documentos)
    elif not documentos:
        st.info("Aún no hay documentos guardados. Sube el primero arriba ☝️")
    else:
        total_fragmentos = sum(d["fragmentos"] for d in documentos)
        st.caption(f"{len(documentos)} documentos · {total_fragmentos} fragmentos indexados en Milvus")

        for doc in documentos:
            extension = doc["nombre"].rsplit(".", 1)[-1].lower() if "." in doc["nombre"] else ""
            detalles = [f"{doc['fragmentos']} fragmentos"]
            if doc.get("recetas_tabla"):
                detalles.append(f"{doc['recetas_tabla']} recetas de tabla")
            if not doc.get("archivo_guardado"):
                detalles.append("original no guardado (vuelve a subirlo para poder descargarlo)")
            for extra in (formatear_tamano(doc.get("tamano_bytes")), formatear_fecha(doc.get("fecha_subida"))):
                if extra:
                    detalles.append(extra)

            with st.container(border=True):
                c_icono, c_info, c_bajar, c_borrar = st.columns([0.5, 5.2, 1.3, 1.3], vertical_alignment="center")
                c_icono.markdown(f"## {ICONOS_DOCUMENTO.get(extension, '📄')}")
                c_info.markdown(f"**{doc['nombre']}**  \n{' · '.join(detalles)}")

                # Descarga en dos pasos: el archivo solo se pide al backend cuando se va a descargar
                if doc.get("archivo_guardado"):
                    clave = f"original_{doc['nombre']}"
                    if clave not in st.session_state:
                        if c_bajar.button("⬇️ Original", key=f"pedir_{doc['nombre']}", use_container_width=True):
                            try:
                                r = requests.get(f"{BACKEND_URL}/recetas/documentos/archivo",
                                                 params={"nombre": doc["nombre"]}, timeout=120)
                                if r.status_code == 200:
                                    st.session_state[clave] = r.content
                                    st.rerun()
                                else:
                                    st.toast(f"❌ No se pudo obtener el original (código {r.status_code})")
                            except requests.exceptions.RequestException:
                                st.toast("❌ No se pudo conectar con el backend.")
                    else:
                        c_bajar.download_button("💾 Guardar", data=st.session_state[clave], file_name=doc["nombre"],
                                                key=f"bajar_{doc['nombre']}", use_container_width=True)
                with c_borrar.popover("🗑️ Borrar", use_container_width=True):
                    st.write(f"¿Eliminar **{doc['nombre']}** y sus {doc['fragmentos']} fragmentos? "
                             "Dejará de usarse en las ideas y búsquedas.")
                    if st.button("Sí, eliminar", key=f"borrar_{doc['nombre']}", type="primary"):
                        try:
                            r = requests.delete(f"{BACKEND_URL}/recetas/documentos",
                                                params={"nombre": doc["nombre"]}, timeout=60)
                            if r.status_code == 200:
                                st.toast(f"🗑️ {r.json().get('mensaje')}")
                            else:
                                try:
                                    detalle = r.json().get("detail", r.status_code)
                                except ValueError:
                                    detalle = r.status_code
                                st.toast(f"❌ No se pudo borrar: {detalle}")
                        except requests.exceptions.RequestException:
                            st.toast("❌ No se pudo conectar con el backend.")
                        st.session_state.pop(f"original_{doc['nombre']}", None)
                        st.session_state.documentos = None
                        st.session_state.resultados_busqueda = None
                        st.rerun()