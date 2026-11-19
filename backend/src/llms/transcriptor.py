"""Transcripción de voz con faster-whisper.

Calidad:
- Modelo configurable (WHISPER_MODELO): "small" entiende el español bastante mejor que "base".
- Vocabulario de cocina y los ingredientes de la despensa como contexto inicial (initial_prompt):
  Whisper se inclina hacia esas palabras ("huevos" en vez de "juegos", "despensa" en vez de "dispensa").
- Filtro de voz (VAD): descarta silencios y ruido, que son los que provocan frases inventadas.
- Audio bajo de volumen: se normaliza antes de transcribir.

Compatibilidad: los WAV (lo que envía el grabador del frontend) se leen aquí directamente, sin PyAV,
para evitar el error "open() got an unexpected keyword argument 'metadata_errors'" de algunas versiones.
"""
import os
import time
import wave

import numpy as np
from faster_whisper import WhisperModel

FRECUENCIA_WHISPER = 16000
WHISPER_MODELO = os.getenv("WHISPER_MODELO", "small").strip() or "small"
WHISPER_BEAM = int(os.getenv("WHISPER_BEAM", "5") or 5)

VOCABULARIO = ("Conversación con CookingAura, un asistente de cocina. Despensa, nevera, lista de la compra, "
               "recetas, ingredientes, desayuno, comida, cena, merienda. Añade, quita, me quedan, he comprado, "
               "he cocinado, soy alérgico, vegetariano, sin gluten. Kilos, gramos, litros, unidades.")


def _cargar_modelo() -> tuple[WhisperModel, str]:
    """Carga el modelo configurado; si no se puede (p. ej. sin internet para descargarlo), usa 'base'."""
    try:
        return WhisperModel(WHISPER_MODELO, device="cpu", compute_type="int8"), WHISPER_MODELO
    except Exception as e:
        if WHISPER_MODELO == "base":
            raise
        print(f"⚠️  Whisper '{WHISPER_MODELO}' no disponible ({e}); se usa 'base'.", flush=True)
        return WhisperModel("base", device="cpu", compute_type="int8"), "base"


modelo, MODELO_CARGADO = _cargar_modelo()
print(f"🎙️ Whisper '{MODELO_CARGADO}' cargado (CPU, int8)", flush=True)


def _prompt_inicial() -> str:
    """Vocabulario de cocina más los ingredientes de la despensa del usuario."""
    try:
        from src.data import sqlitebd
        nombres = list(dict.fromkeys(i["nombre"] for i in sqlitebd.listar_despensa(incluir_agotados=True)))[:40]
    except Exception:
        nombres = []
    return VOCABULARIO + (f" Ingredientes: {', '.join(nombres)}." if nombres else "")


def _leer_wav(ruta: str) -> np.ndarray:
    """WAV PCM -> audio mono float32 a 16 kHz, el formato que espera Whisper."""
    with wave.open(ruta, "rb") as w:
        canales, ancho, frecuencia = w.getnchannels(), w.getsampwidth(), w.getframerate()
        datos = w.readframes(w.getnframes())

    if ancho == 2:
        audio = np.frombuffer(datos, dtype=np.int16).astype(np.float32) / 32768.0
    elif ancho == 4:
        audio = np.frombuffer(datos, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif ancho == 1:
        audio = (np.frombuffer(datos, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"WAV con {ancho * 8} bits por muestra no soportado")

    if canales > 1:
        audio = audio.reshape(-1, canales).mean(axis=1)
    if frecuencia != FRECUENCIA_WHISPER and len(audio):
        n = int(round(len(audio) * FRECUENCIA_WHISPER / frecuencia))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)

    # Grabación muy baja (micro lejos): se sube el volumen hasta un pico del 90 %
    pico = float(np.abs(audio).max()) if len(audio) else 0.0
    if 0 < pico < 0.3:
        audio = audio * (0.9 / pico)
    return audio.astype(np.float32)


def transcribir_audio(ruta_archivo: str) -> str:
    try:
        # WAV: decodificado aquí, sin pasar por PyAV
        entrada = _leer_wav(ruta_archivo)
    except (wave.Error, EOFError, ValueError):
        # Otros formatos (webm, mp3, ogg...): los decodifica faster-whisper con PyAV
        entrada = ruta_archivo

    t0 = time.perf_counter()
    segmentos, info = modelo.transcribe(
        entrada,
        language="es",
        beam_size=WHISPER_BEAM,
        initial_prompt=_prompt_inicial(),
        vad_filter=True,                                  # fuera silencios y ruido
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,                 # mensajes cortos: evita arrastrar errores
    )
    texto = " ".join(segmento.text.strip() for segmento in segmentos).strip()
    print(f"   🎙️ Whisper '{MODELO_CARGADO}': {getattr(info, 'duration', 0):.1f} s de audio "
          f"en {time.perf_counter() - t0:.1f} s", flush=True)
    return texto