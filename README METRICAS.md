# 📊 CookingAura · Métricas de ejecución y dibujo del grafo

Guía práctica para medir la calidad y el rendimiento de CookingAura y para visualizar su grafo de LangGraph.

| Herramienta | Comando | Qué responde |
|---|---|---|
| [Dibujar el grafo](#1-dibujar-el-grafo) | `python -m metricas.dibujarGrafo` | ¿Qué nodos y rutas tiene el asistente? |
| [Consultas de prueba](#3-generar-consultas-de-prueba-generar) | `python -m metricas.evaluar generar` | Crea el conjunto de prueba de la búsqueda |
| [Recall de la búsqueda](#4-calidad-de-la-búsqueda-recall) | `python -m metricas.evaluar recall` | ¿Encuentra la receta correcta? ¿Qué modo de búsqueda es mejor? |
| [Latencia](#5-tiempos-de-extremo-a-extremo-latencia) | `python -m metricas.evaluar latencia` | ¿Cuánto tarda cada escenario y cada nodo? |
| [Casos de voz](#6-preparar-los-casos-de-voz-preparar-voz) | `python -m metricas.evaluar preparar-voz` | Prepara los audios de prueba |
| [Texto frente a voz](#7-texto-frente-a-voz-flujos) | `python -m metricas.evaluar flujos` | ¿Cuánto añade la voz? ¿Se transcribe bien? ¿Se entiende igual? |
| [Calidad de la generación](#calidad-de-la-generación-generacion) | `python -m metricas.evaluar generacion` | ¿Son buenas las ideas y las recetas? ¿Hace el agente lo que se le pide? |
| [Fragmentos indexados](#casos-propios-de-búsqueda-fragmentos) | `python -m metricas.evaluar fragmentos` | ¿Qué id tiene cada receta, para escribir casos a mano? |

---

## Dónde están los scripts

Los dos scripts están en la carpeta **`metricas/`** de la raíz del proyecto:

```
.
├── backend/src/        # código del backend  →  /app/src en el contenedor
├── data/               # resultados          →  /app/data
└── metricas/
    ├── dibujarGrafo.py
    └── evaluar.py      →  /app/metricas
```

### Montar la carpeta en el backend

El contenedor del backend solo ve las carpetas que se montan en él, así que `metricas/` debe estar en los `volumes` de `cookingaura-app-backend` en `docker/docker-compose.yml`:

```yaml
  cookingaura-app-backend:
    volumes:
      - ../backend/src:/app/src
      - ../data:/app/data
      - ../metricas:/app/metricas        # scripts de métricas y dibujo del grafo
      # ...el resto de volúmenes
```

Después de añadirlo, recrea el contenedor (un `restart` no basta para un volumen nuevo):

```bash
docker compose up -d cookingaura-app-backend
```

Al estar montada, cualquier cambio en los scripts se aplica al momento, sin reconstruir la imagen.

### Cómo se lanzan

Siempre **como módulo**, con `python -m` y el nombre con puntos:

```bash
docker exec cookingaura-app-backend python -m metricas.dibujarGrafo
docker exec cookingaura-app-backend python -m metricas.evaluar <comando>
```

- **Sin `.py`, sin rutas y sin barras.** `python -m .\metricas\dibujarGrafo.py` no funciona: dentro del contenedor (Linux) las rutas usan `/`, y `-m` espera un nombre de módulo.
- **Desde `/app`**, que es el directorio de trabajo del contenedor. Así Python encuentra tanto `metricas` como `src`, del que los scripts importan el grafo y el RAG. Lanzarlos por ruta (`python metricas/evaluar.py`) falla al importar `src`.

### Rutas de salida

Los scripts guardan los resultados en `data/`, que calculan subiendo **un nivel** desde su propia ubicación (`metricas/` → raíz del proyecto, o `/app` en el contenedor):

```python
# metricas/dibujarGrafo.py
SALIDA_POR_DEFECTO = Path(__file__).resolve().parents[1] / "data" / "grafo"
# metricas/evaluar.py
CARPETA = Path(__file__).resolve().parents[1] / "data" / "eval"
```

Si movieras los scripts a otro nivel de carpetas, ajusta ese `parents[...]`. Si no, los resultados se escribirían fuera de `/app/data` y no aparecerían en tu carpeta `data/`.

---

## Índice

1. [Dibujar el grafo](#1-dibujar-el-grafo)
2. [Antes de medir](#2-antes-de-medir)
3. [Generar consultas de prueba (`generar`)](#3-generar-consultas-de-prueba-generar)
4. [Calidad de la búsqueda (`recall`)](#4-calidad-de-la-búsqueda-recall)
5. [Tiempos de extremo a extremo (`latencia`)](#5-tiempos-de-extremo-a-extremo-latencia)
6. [Preparar los casos de voz (`preparar-voz`)](#6-preparar-los-casos-de-voz-preparar-voz)
7. [Texto frente a voz (`flujos`)](#7-texto-frente-a-voz-flujos)
8. [Flujo de trabajo recomendado](#8-flujo-de-trabajo-recomendado)
9. [Otras fuentes de métricas](#9-otras-fuentes-de-métricas)
10. [Problemas habituales](#10-problemas-habituales)

---

## 1. Dibujar el grafo

`metricas/dibujarGrafo.py` exporta los nodos y las aristas del grafo de LangGraph a partir del código, así que el dibujo siempre coincide con lo que se ejecuta.

### Cómo lanzarlo

Dentro del contenedor del backend (recomendado):

```bash
docker exec cookingaura-app-backend python -m metricas.dibujarGrafo
```

En local, desde la raíz del proyecto y con las dependencias del backend instaladas, añadiendo `backend/` al `PYTHONPATH` para que encuentre `src`:

```bash
# Linux / macOS
PYTHONPATH=backend python -m metricas.dibujarGrafo
# Windows (PowerShell)
$env:PYTHONPATH = "backend"; python -m metricas.dibujarGrafo
```

En local los ficheros se guardan en la carpeta `data/grafo/` de la raíz del proyecto.

Importar el grafo solo construye su estructura: **no conecta con Milvus, Ollama, Google ni MCP**, así que funciona aunque esos servicios estén parados.

### Opciones

| Opción | Valor por defecto | Descripción |
|---|---|---|
| `--salida RUTA` | `data/grafo` (en el contenedor, `/app/data/grafo`) | Carpeta donde se guardan los ficheros |
| `--formatos ...` | todos | Uno o varios de `mermaid`, `html`, `png`, `ascii` |

```bash
python -m metricas.dibujarGrafo --formatos html mermaid
python -m metricas.dibujarGrafo --salida /app/data/otra_carpeta
```

### Qué genera

| Formato | Fichero | Cómo verlo |
|---|---|---|
| `html` | `data/grafo/grafo.html` | Ábrelo en el navegador. Se adapta a modo claro u oscuro |
| `mermaid` | `data/grafo/grafo.mmd` | Pégalo en [mermaid.live](https://mermaid.live) o dentro de un bloque ```` ```mermaid ```` en un README de GitHub |
| `png` | `data/grafo/grafo.png` | Imagen. Se genera con el servicio mermaid.ink, así que **necesita internet**; si falla, se avisa y se generan los demás |
| `ascii` | (por consola) | Dibujo en texto. Necesita `pip install grandalf` |

Además, siempre imprime por consola la lista de nodos y aristas:

```
Nodos (14):
  · __start__
  · intencion
  · mcp
  ...
Aristas (…):
               __start__ → preparar_receta        (condicional)
               intencion → agente                 (condicional)
          presentar_ideas → __end__               (fija)
```

**Cómo leer el dibujo:** las aristas continuas son fijas (siempre se recorren) y las discontinuas son condicionales (las decide una función `decidir_*`).

Como `data/` está montada en el contenedor, los ficheros aparecen directamente en la carpeta `data/grafo/`.

---

## 2. Antes de medir

### Dónde se ejecutan

Todos los comandos de evaluación se lanzan dentro del contenedor del backend:

```bash
docker exec cookingaura-app-backend python -m metricas.evaluar <comando> [opciones]
```

Los resultados se guardan en `data/eval/`. Para ver la ayuda de cualquier comando:

```bash
docker exec cookingaura-app-backend python -m metricas.evaluar <comando> --help
```

### Qué necesita cada comando

| Comando | Necesita |
|---|---|
| `generar` | Documentos indexados en Milvus y un LLM (Ollama o Gemini) |
| `recall` | Documentos indexados y el fichero de consultas de `generar` |
| `latencia` | La aplicación completa en marcha (backend, Ollama, Milvus) |
| `preparar-voz` | Nada (opcional: `espeak-ng` para sintetizar los audios) |
| `flujos` | La aplicación completa en marcha y, para la parte de voz, los audios de `preparar-voz` |

### Consejos para que las medidas sean fiables

- **Calienta el sistema antes.** La primera petición tras arrancar carga bge-m3, Whisper y los modelos de Ollama, y tarda mucho más. Haz una petición cualquiera desde el chat antes de medir, o descarta la primera repetición.
- **No uses la app mientras se mide.** Las peticiones de otros usuarios comparten CPU y modelos, y alteran los tiempos.
- **Anota la configuración.** Los resultados dependen de los modelos (`OLLAMA_MODELO_*`, `AGENTE_MODELO`), de `IDEAS_MOTOR`, de `RAG_*` y de si hay GPU. Guarda junto a cada CSV qué configuración tenías.
- **Las medidas no modifican tus datos.** Ningún escenario añade, gasta ni borra nada de la despensa, la lista de la compra o el historial.

---

## 3. Generar consultas de prueba (`generar`)

Para medir si la búsqueda encuentra la receta correcta hace falta un conjunto de consultas cuya respuesta se conozca. Etiquetarlas a mano es lento, así que este comando las crea a partir de tus propias recetas indexadas.

```bash
docker exec cookingaura-app-backend python -m metricas.evaluar generar --n 40
```

### Cómo funciona

1. Toma fragmentos al azar de la colección `recetas` (descarta los de menos de 200 caracteres).
2. Para cada uno, pide al LLM una consulta como la haría un usuario, **sin el nombre exacto del plato**: por ejemplo, «algo con garbanzos y espinacas para comer». Así la búsqueda léxica no acierta solo porque la consulta copia el título.
3. La respuesta correcta de cada consulta es el fragmento del que salió.

### Opciones

| Opción | Valor por defecto | Descripción |
|---|---|---|
| `--n` | `40` | Número de consultas |
| `--motor` | el de `IDEAS_MOTOR` | `ollama` o `gemini` para redactar las consultas (Gemini es mucho más rápido en CPU) |
| `--semilla` | `42` | Semilla del muestreo: misma semilla, mismos fragmentos |

### Resultado

`data/eval/consultas.jsonl`, una consulta por línea:

```json
{"consulta": "algo con garbanzos y espinacas para comer", "ids_relevantes": [454123987654321], "fuente": "Recetario_CookingAura.xlsx", "titulo": "Potaje de garbanzos con espinacas"}
```

**Revísalo.** Puedes corregir consultas mal redactadas, borrar las que no tengan sentido y **añadir consultas reales tuyas** (con el `id` del fragmento correcto en `ids_relevantes`). Un conjunto mixto, sintético y real, da una medida más honesta.

> Cada ejecución sobrescribe `consultas.jsonl`. Si lo has editado a mano, haz una copia antes de volver a generarlo.

---

## 4. Calidad de la búsqueda (`recall`)

Compara las configuraciones de búsqueda con las consultas de prueba.

```bash
docker exec cookingaura-app-backend python -m metricas.evaluar recall
```

### Configuraciones comparadas

| Nombre | Búsqueda |
|---|---|
| `semantica` | Solo el vector denso (bge-m3) |
| `lexica_bm25` | Solo BM25 |
| `hibrida_rrf` | Las dos, fusionadas por posición (RRF) |
| `hibrida_pond_0.3` | Las dos, 30 % semántica y 70 % léxica |
| `hibrida_pond_0.5` | Las dos, a partes iguales |
| `hibrida_pond_0.7` | Las dos, 70 % semántica y 30 % léxica |

### Opciones

| Opción | Valor por defecto | Descripción |
|---|---|---|
| `--k` | `1 3 5 10` | Valores de k para recall@k y doc_hit@k |
| `--configs` | todas | Solo algunas configuraciones, por ejemplo `--configs semantica hibrida_rrf` |

### Métricas

| Métrica | Qué significa | Cómo interpretarla |
|---|---|---|
| `recall@k` | Porcentaje de consultas en las que la receta correcta sale entre los k primeros | `recall@3 = 0.85`: en el 85 % de los casos está en el top 3 |
| `mrr` | Media de 1/posición de la receta correcta | 1 = siempre primera; 0,5 = de media segunda |
| `doc_hit@k` | Acierta si sale cualquier fragmento del mismo documento | Más permisiva: útil porque una receta puede estar partida en varios fragmentos |
| `latencia_p50_ms` / `p95_ms` | Tiempo de búsqueda típico y del 5 % más lento | Incluye el embedding de la consulta |

Ejemplo de salida (valores ilustrativos):

```
           config         recall@1         recall@3   ...              mrr       doc_hit@10  latencia_p50_ms
        semantica             0.62             0.81   ...            0.718            0.950            412.3
      lexica_bm25             0.48             0.70   ...            0.592            0.875             18.6
      hibrida_rrf             0.67             0.86   ...            0.762            0.975            431.0
...
🏆 Mejor configuración: hibrida_rrf
```

### Resultados

| Fichero | Contenido |
|---|---|
| `data/eval/recall_<fecha>.csv` | Resumen: una fila por configuración |
| `data/eval/recall_detalle_<fecha>.csv` | Posición de la receta correcta en cada consulta y configuración (vacía si no apareció) |

### Qué hacer con el resultado

- Pon la mejor configuración en el `.env` (`RAG_MODO`, `RAG_RANKER`, `RAG_PESO_DENSO`) y reinicia el backend.
- Revisa en el detalle las consultas que fallan en todas las configuraciones: suelen ser fragmentos mal extraídos o consultas ambiguas.
- Si `recall@k` es bajo pero `doc_hit@k` es alto, el problema está en el troceado: la receta se reparte entre varios fragmentos.

El comando hace una búsqueda de calentamiento antes de medir, para que la carga de bge-m3 no cuente en las latencias.

---

## 5. Tiempos de extremo a extremo (`latencia`)

Mide el asistente completo, contra `/chat/stream` y tal como lo usa el frontend.

```bash
docker exec cookingaura-app-backend python -m metricas.evaluar latencia --repeticiones 3
```

### Escenarios

| Escenario | Mensaje | Además |
|---|---|---|
| `ideas_con_ingredientes` | «¿Qué ceno con huevos, patatas y cebolla?» (con ingredientes y tipo de comida) | Pide la receta de una idea propia y de una creativa |
| `ideas_despensa` | «¿Qué como hoy con lo que tengo?» (despensa completa) | Ídem |
| `chat_libre` | «Quiero algo rápido para desayunar con avena» | — |

Las recetas solo se piden si la fase de ideas devolvió ideas de ese tipo.

### Opciones

| Opción | Valor por defecto | Descripción |
|---|---|---|
| `--url` | `http://localhost:8080` | Dirección del backend (dentro del contenedor, `localhost` es el propio backend) |
| `--repeticiones` | `3` | Veces que se repite cada escenario |
| `--timeout` | `900` | Segundos máximos por petición |

### Métricas

Ejemplo de salida más abajo (valores ilustrativos).

- **Total** por escenario y fase (ideas o receta): p50 y p95.
- **Primer token** de la receta: lo que el usuario percibe como espera.
- **Tiempo por nodo** del grafo, ordenado de más lento a más rápido: indica dónde optimizar.
- **Número de ideas** devueltas: si es menor de lo esperado, el LLM probablemente no generó un JSON válido.

```
Resumen
  ideas_con_ingredientes           ideas   total p50  21.40s · p95  24.10s
  ideas_con_ingredientes/creativa  receta  total p50  48.20s · p95  52.70s | 1er token p50  9.80s · p95 11.20s

Tiempo por nodo (todas las ejecuciones)
  generador              p50  38.10s · p95  42.30s  (n=6)
  ideas_creativas        p50  18.90s · p95  21.00s  (n=9)
  ...
```

### Resultado

`data/eval/latencia_<fecha>.csv`, una fila por ejecución: escenario, fase, repetición, total, primer token, tiempo de generación, motor (`vllm`, `ollama` o `agente`), número de ideas, tiempos de cada nodo (JSON) y error si lo hubo.

---

## 6. Preparar los casos de voz (`preparar-voz`)

Prepara los casos y los audios que usa `flujos`.

```bash
docker exec cookingaura-app-backend python -m metricas.evaluar preparar-voz
```

### Qué hace

1. Crea `data/eval/voz/casos.json` (si no existe) con 6 casos que cubren las tres rutas del asistente:

   | `id` | Mensaje | Ruta esperada |
   |---|---|---|
   | `saludo_comer` | Hola, ¿qué puedo comer hoy? | `ideas` |
   | `cena_ingredientes` | ¿Qué ceno con huevos y patatas? | `ideas` |
   | `consulta_despensa` | ¿Me quedan huevos en la despensa? | `asistente` |
   | `consulta_compra` | ¿Qué tengo en la lista de la compra? | `asistente` |
   | `duda_cocina` | ¿Cómo se pocha una cebolla? | `asistente` |
   | `saludo` | Buenos días | `directa` |

2. Para cada caso sin audio, lo sintetiza con `espeak-ng` si está instalado. Si no, indica qué WAV faltan. Los audios que ya existen nunca se sobrescriben.

### Audios: sintéticos o propios

| Opción | Cómo | Cuándo |
|---|---|---|
| **Voz sintética** | Añade `espeak-ng` al `Dockerfile` del backend: `RUN apt-get update && apt-get install -y espeak-ng` | Para medir el sistema de forma repetible |
| **Grabaciones propias** | Graba los mensajes y guárdalos como WAV en `data/eval/voz/` con el nombre de su caso (`saludo_comer.wav`…) | **Recomendado** para el WER: Whisper transcribe mucho mejor una voz humana que una sintética, y así la medida refleja tu micro y tu forma de hablar |

Formato de los audios: WAV PCM, mono o estéreo, a cualquier frecuencia (se convierte a 16 kHz al transcribir).

### Añadir casos propios

Edita `data/eval/voz/casos.json`:

```json
[
  {"id": "mi_caso", "texto": "¿Qué puedo cenar ligero?", "intencion": "ideas", "audio": "mi_caso.wav"}
]
```

- `texto`: lo que se dice en el audio. Es la referencia para el WER y el mensaje del modo texto.
- `intencion`: ruta esperada: `ideas`, `asistente` o `directa` (saludos y temas ajenos a la cocina). Opcional; si falta, no se mide el acierto de ruta.
- `audio`: nombre del WAV dentro de `data/eval/voz/`.

> Usa solo mensajes de **consulta**. Un caso como «añade leche a la lista de la compra» modificaría tus datos en cada repetición.

---

## 7. Texto frente a voz (`flujos`)

Lanza los mismos casos por las dos vías que tiene el usuario y compara.

```bash
docker exec cookingaura-app-backend python -m metricas.evaluar flujos --repeticiones 3
```

### Qué hace con cada caso

| Modo | Recorrido (igual que en el frontend) |
|---|---|
| **Texto** | Mensaje → `/chat/stream` |
| **Voz** | Audio → `/transcribir` → texto transcrito → `/chat/stream` |

Con `--receta`, en los casos que devuelven ideas pide además la receta de la primera, para medir el flujo hasta el final.

Si no hay audios en `data/eval/voz/`, avisa y mide solo el modo texto.

### Opciones

| Opción | Valor por defecto | Descripción |
|---|---|---|
| `--url` | `http://localhost:8080` | Dirección del backend |
| `--repeticiones` | `3` | Veces que se repite cada caso |
| `--timeout` | `900` | Segundos máximos por petición |
| `--receta` | desactivado | Mide también la fase de receta |

### Métricas

| Métrica | Qué mide |
|---|---|
| **Total** (p50/p95) | Desde enviar el mensaje o el audio hasta la respuesta completa. En voz incluye la transcripción |
| **1ª respuesta** | Tiempo hasta el primer texto visible |
| **Ruta correcta** | Porcentaje de casos que acabaron en la ruta esperada. Compararlo entre texto y voz muestra si la voz hace que el asistente entienda peor |
| **Transcripción** | Tiempo de Whisper y qué porcentaje del total supone |
| **RTF** | *Real-time factor*: segundos de transcripción por segundo de audio. Por debajo de 1, transcribe más rápido de lo que se habla |
| **WER** | *Word error rate*: porcentaje de palabras mal transcritas (sustituidas, omitidas o añadidas), sin contar tildes, mayúsculas ni puntuación. 0 % es perfecto |
| **Receta** | Con `--receta`, el tiempo de la fase de receta en cada modo |

### Salida por consola (valores ilustrativos)

```
================ RESUMEN: TEXTO frente a VOZ ================

TEXTO  total        p50   6.20s · p95   9.80s
       1ª respuesta p50   6.10s · p95   9.70s
       ruta correcta 18/18 (100%)

VOZ    total        p50   8.90s · p95  12.40s
       1ª respuesta p50   8.80s · p95  12.30s
       ruta correcta 17/18 (94%)
       transcripción p50   2.60s · p95   3.10s · RTF medio 0.85 · 29% del tiempo total
       WER medio 4.2% (0% = transcripción perfecta)

Por caso (p50 del total, texto → voz):
  saludo_comer             5.1s →    7.6s  (+2.5s por voz)
  consulta_despensa        7.3s →    9.9s  (+2.6s por voz)  · oído: «¿Me quedan juegos en la despensa?»
```

Cuando Whisper no transcribe exactamente el texto, la tabla por caso muestra lo que entendió. Así ves de un vistazo qué palabras le cuestan.

### Resultado

`data/eval/flujos_<fecha>.csv`, una fila por ejecución:

| Columna | Contenido |
|---|---|
| `caso`, `modo`, `fase`, `repeticion` | Qué se midió (`modo`: texto o voz; `fase`: chat o receta) |
| `intencion_esperada`, `ruta`, `acierto_ruta` | Ruta esperada, ruta real y si coinciden (1/0) |
| `total_s`, `primer_token_s` | Tiempos (en voz, con la transcripción incluida) |
| `transcripcion_s`, `audio_s`, `rtf`, `wer`, `transcripcion` | Solo en voz |
| `nodos` | Tiempo de cada nodo y de cada herramienta del agente (JSON) |
| `error` | Error, si lo hubo |

### Cómo interpretar los resultados

- **La voz añade siempre la transcripción.** Si supone una parte grande del total y el RTF es alto, Whisper va lento en tu CPU: un modelo más pequeño (`tiny` en lugar de `base`, en `src/llms/transcriptor.py`) transcribe más rápido, aunque con más errores.
- **Un WER alto con voz sintética no es preocupante:** compáralo con tus grabaciones antes de sacar conclusiones.
- **Si la ruta correcta baja en voz**, mira las transcripciones: una palabra mal entendida («juegos» por «huevos») puede cambiar la intención.
- **Si la ruta correcta baja también en texto**, el problema está en la detección de intención, no en la voz. El log del backend muestra la respuesta del clasificador (`Clasificador (...) respondió: ...`).

---

## Casos propios de búsqueda (`fragmentos`)

`consultas.jsonl` admite casos escritos a mano. La respuesta correcta se indica con `ids_relevantes` (ids de fragmentos de Milvus) o con `texto_relevante` (un trozo del texto, normalmente el título):

```json
{"consulta": "algo con huevos y patatas para cenar", "texto_relevante": "Tortilla de patatas"}
{"consulta": "sopa fría de tomate y pepino", "ids_relevantes": [4521], "fuente": "Recetario_CookingAura.xlsx"}
```

Para encontrar los ids:

```bash
docker exec cookingaura-app-backend python -m metricas.evaluar fragmentos --buscar "gazpacho"
```

Lista los fragmentos con su id, documento y título, y los guarda todos en `data/eval/fragmentos.csv`. **Los ids cambian al reindexar o volver a subir un documento**, así que para los casos escritos a mano es preferible `texto_relevante`. Las líneas que empiezan por `#` se ignoran.

---

## Calidad de la generación (`generacion`)

Evalúa si las ideas, las recetas y las respuestas del agente son correctas, no solo cuánto tardan. Los casos están en `data/eval/generacion.json` y son de tres tipos:

| Tipo | Qué hace el caso | Qué mide |
|---|---|---|
| `ideas` | Pide ideas (con ingredientes o con la despensa); opcionalmente la receta de la primera (`"receta": true`) y más ideas (`"mas_ideas": true`) | Nº de ideas, duplicados, uso de los ingredientes, ideas con alérgenos, ideas propias válidas, repeticiones al pedir más |
| `plato` | Pide la receta de un plato por nombre | Estructura (ingredientes y pasos numerados, como máximo 6), coherencia con el plato, alérgenos mencionados |
| `agente` | Mensaje al chat libre (con historial opcional) | Ruta, herramientas usadas (`herramientas_todas`, `herramientas_alguna`, `herramientas_prohibidas`), contenido de la respuesta (`respuesta_contiene`, con alternativas `"6\|seis"`), **efecto real en los datos** (`efectos`), botones inventados y tiempos agotados |

Las recetas de ideas propias se comparan además con la receta original (**fidelidad**: qué parte de sus ingredientes conserva).

```bash
docker exec cookingaura-app-backend python -m metricas.evaluar generacion
docker exec cookingaura-app-backend python -m metricas.evaluar generacion --solo agente
docker exec cookingaura-app-backend python -m metricas.evaluar generacion --juez --repeticiones 2
```

**Datos aislados.** Cada caso puede preparar datos (`preparacion`: despensa, lista de la compra, preferencias; `evitar`: alergias temporales). Antes de cada caso se copia `data/ingredients.db` y después se restaura, así que tus datos no cambian aunque el agente añada, gaste o borre. Si la evaluación se interrumpe, la siguiente ejecución restaura la copia al empezar. **No uses la app mientras se evalúa.**

**Efectos comprobables** (en `efectos`): `sin_cambios`, `despensa_cantidad`, `despensa_agotado`, `despensa_aumenta`, `lista_contiene`, `preferencia`, `historial_contiene`.

**Juez (`--juez`).** Gemini puntúa de 1 a 5 con una rúbrica fija: adecuación, uso de ingredientes, variedad y realismo en las ideas; claridad, cantidades, coherencia y respeto de restricciones en las recetas; corrección, utilidad y concisión en el agente. Sirve para comparar configuraciones, no como nota absoluta. Si Gemini falla, se sigue sin juez.

**Resultados:** `data/eval/generacion_<fecha>.csv` (métricas por caso y fase) y `generacion_<fecha>_detalle.jsonl` (textos completos, para revisar a mano).

Que una receta **mencione** un alérgeno no siempre es un fallo: puede ser el aviso de una sustitución («sin gambas por tu alergia»). Revisa esos casos en el detalle o con el juez.

---

## 8. Flujo de trabajo recomendado

**Para elegir la configuración de búsqueda** (tras subir o cambiar documentos):

```bash
docker exec cookingaura-app-backend python -m metricas.evaluar generar --n 40
# revisa data/eval/consultas.jsonl
docker exec cookingaura-app-backend python -m metricas.evaluar recall
# pon la mejor configuración en el .env y reinicia el backend
```

**Para medir el rendimiento** (tras cambiar modelos, hardware o código):

```bash
docker exec cookingaura-app-backend python -m metricas.evaluar latencia --repeticiones 3
docker exec cookingaura-app-backend python -m metricas.evaluar preparar-voz      # solo la primera vez
docker exec cookingaura-app-backend python -m metricas.evaluar flujos --repeticiones 3 --receta
```

**Para documentar el sistema:**

```bash
docker exec cookingaura-app-backend python -m metricas.dibujarGrafo
```

Con Ollama en CPU, una ronda de `flujos` (6 casos por las dos vías, más recetas) puede tardar bastantes minutos. Para una primera medida basta con `--repeticiones 1`.

Para comparar dos configuraciones, cambia una sola cosa cada vez (por ejemplo, `AGENTE_MODELO`), repite la misma medida y compara los CSV.

---

## 9. Otras fuentes de métricas

Además de estos comandos, la aplicación mide continuamente:

| Dónde | Qué |
|---|---|
| **Chat** | Cada paso del progreso muestra su duración («🥫 Despensa revisada (1.2 s) → …») |
| **Logs del backend** (`docker logs -f cookingaura-app-backend`) | `⏱ nodo: X s` por cada nodo del grafo; `⏱ [RAG búsqueda]` y `⏱ [RAG indexación]` con cada fase del RAG; `🔧 herramienta(...)` por cada llamada del agente; `🎙️ Transcripción en X s` |
| **Subida de documentos** | Tiempos de cada fase de la indexación bajo el mensaje de éxito |
| **Evento `fin` de `/chat/stream`** | `metricas`: tiempo total, motor, tiempo de generación y tiempo hasta el primer token del LLM |
| **Ollama** | `docker exec ollama-server ollama ps` (si el modelo está en CPU o GPU) |
| **LangSmith** (opcional) | Con `LANGCHAIN_TRACING_V2=true` y una clave válida, trazas de cada llamada a LLM con prompts, tokens y tiempos |

Para medir la velocidad bruta de un modelo de Ollama (tokens por segundo):

```bash
curl -s http://localhost:11434/api/generate -d '{"model": "qwen2.5:3b", "prompt": "Escribe una receta de tortilla.", "stream": false, "options": {"num_predict": 300}}' \
  | python3 -c "import json,sys; r=json.load(sys.stdin); s=1e9; print(f'lectura {r[\"prompt_eval_count\"]/(r[\"prompt_eval_duration\"]/s):.0f} tok/s · generación {r[\"eval_count\"]/(r[\"eval_duration\"]/s):.1f} tok/s')"
```

---

## 10. Problemas habituales

| Síntoma | Solución |
|---|---|
| `No module named metricas` | La carpeta no está montada en el contenedor: añade `../metricas:/app/metricas` a los volúmenes del backend y recrea el contenedor con `docker compose up -d cookingaura-app-backend` |
| `No module named src` | El script se lanzó por ruta (`python metricas/evaluar.py`) o desde otra carpeta. Lánzalo con `python -m metricas.evaluar` (en local, con `PYTHONPATH=backend`) |
| `Relative module names not supported` o `No module named '.\metricas...'` | `-m` recibe el nombre del módulo con puntos, no una ruta: `python -m metricas.dibujarGrafo` |
| Los resultados no aparecen en `data/` | Las rutas de salida no suben el nivel correcto: revisa `parents[1]` en los dos scripts (sección «Rutas de salida») |
| `No existe .../consultas.jsonl` al lanzar `recall` | Ejecuta antes `generar` |
| `La colección 'recetas' no existe o está vacía` | Sube documentos desde la pestaña Recetas (o reindexa con `python -m src.rag.indexer --recrear --reindexar`) |
| `generar` omite muchos fragmentos | El LLM no devuelve JSON válido. Prueba `--motor gemini` o un modelo mayor |
| `latencia` o `flujos` fallan con errores de conexión | El backend no está en marcha o la `--url` no es correcta. Desde fuera del contenedor, usa `--url http://localhost:8080` con el puerto publicado |
| Errores por *timeout* | En CPU, sube `--timeout` o usa modelos más pequeños |
| `flujos` mide solo el modo texto | Faltan los WAV: ejecuta `preparar-voz` y añade los audios que indique |
| `preparar-voz` dice que faltan audios | No hay `espeak-ng`: instálalo en el contenedor o graba tú los WAV |
| Transcripción vacía en algún caso | Audio en silencio o demasiado bajo. Revisa el WAV |
| `PNG no generado` en `dibujarGrafo` | No hay acceso a mermaid.ink. Usa el HTML o pega el `.mmd` en mermaid.live |
| `ImportError` al lanzar `dibujarGrafo` en local | Al entorno le faltan dependencias del backend (o `mcp` y `langchain-mcp-adapters` son incompatibles). Ejecútalo dentro del contenedor |
| Los tiempos de la primera repetición son muy altos | Es la carga inicial de modelos: calienta el sistema antes o descarta esa repetición |