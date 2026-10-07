# LongToShort

Aplicación local para Windows que toma un video largo de YouTube, lo transcribe,
detecta automáticamente sus **mejores momentos** y los convierte en **Shorts
verticales 9:16** (MP4 1080x1920).

```
URL de YouTube → descarga → transcripción → análisis → candidatos → selección → clips → 9:16
```

Todo corre en tu PC: no usa servicios pagos ni APIs externas (salvo YouTube para descargar).

---

## Requisitos

| Herramienta | Versión | Para qué |
|---|---|---|
| Windows 10/11 | — | |
| Python | 3.10 o superior (probado con 3.12) | la aplicación |
| FFmpeg | cualquier versión reciente | cortar y convertir los clips, leer audio |
| Node.js o Deno | opcional pero recomendado | yt-dlp lo usa para resolver los formatos de YouTube |
| GPU NVIDIA | opcional | transcripción mucho más rápida (si no hay, usa la CPU) |

Espacio en disco: ~1,5 GB para los modelos + lo que ocupen los videos (un video de 40 min en 1080p ≈ 550 MB).

### Instalar FFmpeg

La forma más simple (PowerShell o CMD):

```
winget install Gyan.FFmpeg
```

Cerrá y volvé a abrir la terminal, y verificá con `ffmpeg -version`.
Alternativa manual: descargá un build de <https://www.gyan.dev/ffmpeg/builds/>, descomprimilo y agregá la carpeta `bin` al `PATH`.

### Instalar Node.js (recomendado)

```
winget install OpenJS.NodeJS.LTS
```

YouTube exige ejecutar JavaScript para obtener los formatos de video; sin Node/Deno algunas descargas pueden fallar.

---

## Instalación

La primera vez, simplemente ejecutá **`iniciar.bat`** (doble clic). El script:

1. crea el entorno virtual propio `LongToShort\.venv` (independiente de cualquier otro proyecto);
2. instala las dependencias de `requirements.txt`;
3. si detecta una GPU NVIDIA, instala las librerías CUDA (`requirements-gpu.txt`, ~1 GB, una sola vez);
4. inicia el servidor y abre el navegador en <http://127.0.0.1:8765>.

Instalación manual equivalente:

```
cd LongToShort
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-gpu.txt   :: solo con GPU NVIDIA
python -m app.main --open-browser
```

La primera vez que analices un video se descargan los modelos a `models/`
(Whisper large-v3-turbo ≈ 1,6 GB, embeddings ≈ 0,2 GB, detector de rostros ≈ 0,3 MB).

## Uso

1. Ejecutá `iniciar.bat`.
2. Pegá la URL de YouTube y tocá **Analizar video**.
3. Esperá la descarga, la transcripción y el análisis (la barra muestra el progreso de cada etapa).
4. Revisá los candidatos: cada uno tiene puntaje 0-100, tiempos, el texto, por qué fue elegido
   y el desglose por dimensión. Con **▶ Ver** se reproduce ese tramo del video original.
5. Marcá los que quieras y tocá **Generar Shorts**.
6. Los archivos quedan en `output\VIDEO_ID\short_01.mp4`, `short_02.mp4`, … (y en la página para verlos/descargarlos).

Para detener el servidor: cerrá la ventana de `iniciar.bat` o ejecutá **`detener.bat`**.

Si volvés a analizar el mismo video, se reutilizan la descarga y la transcripción (solo se recalcula el análisis, que tarda segundos).

Tiempos medidos con un video de 40 minutos en una PC con GTX 1050 (2 GB):

| Etapa | Tiempo |
|---|---|
| Descarga 1080p | ~1 min (según la conexión) |
| Transcripción (large-v3-turbo, GPU) | ~5,5 min |
| Análisis y detección | ~10 s (la primera vez también se descarga el modelo de embeddings) |
| Generación de un Short de 50 s | ~15 s |

---

## Estructura

```
LongToShort/
├── app/
│   ├── main.py               servidor FastAPI y endpoints
│   ├── pipeline.py           orquesta las etapas (análisis y generación)
│   ├── jobs.py               cola de trabajos en segundo plano con estados y progreso
│   ├── downloader.py         yt-dlp: metadata, selección de formato y de la pista de audio ORIGINAL
│   ├── transcription.py      faster-whisper: idioma real, timestamps por palabra, GPU/CPU
│   ├── analyzer.py           frases, rasgos semánticos/prosódicos, cambios de tema
│   ├── lexicon.py            señales lingüísticas y prototipos semánticos
│   ├── clip_scorer.py        puntaje de cada ventana en 7 dimensiones
│   ├── clip_detector.py      generación de ventanas, deduplicación y ranking
│   ├── vertical_converter.py plan de recorte 9:16 (rostros por plano / centro)
│   ├── clip_generator.py     corte + recorte + codificación con FFmpeg
│   ├── media.py              utilidades de FFmpeg/FFprobe
│   ├── config.py             rutas y lectura de config.json
│   ├── errors.py             errores con mensaje para el usuario
│   └── logging_setup.py      logs en logs/
├── templates/index.html      interfaz
├── static/css, static/js     estilos y lógica de la interfaz (JS vanilla)
├── downloads/VIDEO_ID/       original.mp4, metadata.json, transcript.json, analysis.json
├── output/VIDEO_ID/          short_XX.mp4 y shorts.json (qué momento es cada archivo)
├── models/  temp/  logs/
├── config.json               toda la configuración
├── requirements.txt          dependencias
├── requirements-gpu.txt      librerías CUDA opcionales
├── iniciar.bat / detener.bat
```

## Cómo funciona

### Descarga y audio original
yt-dlp lista todos los formatos. El video se elige explícitamente (≤1080p, prefiriendo H.264).
El audio **no** se toma "a ciegas": muchos videos tienen doblajes (incluso automáticos), así que
se identifica la pista original con, en orden: la marca *original* de YouTube, los subtítulos
automáticos `*-orig` (generados sobre el audio original), el idioma declarado del video y la pista por defecto.

Antes de transcribir, Whisper **detecta** el idioma del audio descargado. Si no coincide con el
idioma original (p. ej. sale árabe en un video en español) se considera que la pista elegida es
incorrecta: se descarga la pista del idioma original y se verifica de nuevo. El idioma nunca se
fuerza en Whisper para "tapar" una pista equivocada.

### Transcripción
faster-whisper (`large-v3-turbo` por defecto) con filtro de voz (VAD) y timestamps **por palabra**.
Usa la GPU NVIDIA si está disponible y si no la CPU (`cpu_model`, por defecto `small`, más rápido).

### Detección de momentos
1. **Frases**: se reconstruyen desde los timestamps por palabra (puntuación y pausas), y se recortan
   muletillas iniciales ("Eh, bueno, …"). Un clip siempre empieza y termina en un límite de frase.
2. **Rasgos por frase**:
   - *semánticos*: un modelo de embeddings multilingüe compara cada frase con prototipos de
     ganchos, remates/conclusiones, historias, consejos/datos, momentos intensos y frases que
     dependen de contexto previo. Esto captura el **sentido** de la frase, no palabras sueltas;
   - *estructura del tema*: detección de cambios de tema entre bloques de frases (estilo TextTiling)
     y pausas largas, para saber dónde empieza y termina una idea;
   - *prosodia*: energía de la voz (énfasis) relativa al propio video, ritmo, silencios;
   - *léxicos*: preguntas, afirmaciones fuertes, números, conectores. Solo suman como una señal más.
3. **Ventanas**: se evalúan todas las combinaciones frase-inicio → frase-fin de 20-60 s (se aceptan
   hasta `duration_flex` segundos de desvío solo si la estructura narrativa es fuerte).
4. **Puntaje** (0-10 por dimensión, 0-100 total):
   - **Hook**: la apertura plantea una pregunta, afirmación fuerte o algo intrigante.
   - **Impacto**: emoción, sorpresa, humor, opinión fuerte, picos de energía en la voz.
   - **Valor**: consejos, explicaciones, datos, historias; pocas muletillas.
   - **Storytelling**: hook → desarrollo → payoff: cierre conclusivo, coherencia interna, el final
     retoma la apertura, pregunta que se responde.
   - **Autonomía**: no empieza con "y entonces…", "eso…", "como te decía…"; empieza en un cambio de tema.
   - **Retención**: ritmo de habla, pocos silencios, voz con energía, duración adecuada.
   - **Inicio/final**: corta en pausas naturales, termina en una frase completa.
5. **Deduplicación**: supresión de no-máximos; si dos candidatos se superponen más de `max_overlap`
   (proporción del más corto) queda solo el mejor.
6. **Ranking**: hasta `max_candidates`, descartando los que quedan muy por debajo del mejor
   (`relative_cutoff`) o del mínimo absoluto (`min_score`). Mejor 8 excelentes que 30 mediocres.

### Conversión 9:16
Nunca se estira la imagen: se recorta una ventana 9:16 de la altura completa y se escala a 1080x1920.
Para ubicar la ventana se muestrean frames, se detectan los **cambios de plano** y en cada plano se
detectan rostros (YuNet) y se centra al sujeto principal (el más grande y persistente). Sin una
detección confiable se usa el centro. `vertical_converter.py` produce un *plan de recorte* por
tramos, así que se pueden sumar tracking de rostros/personas/objetos o un crop dinámico sin tocar el resto.

---

## Configuración (`config.json`)

Los cambios se aplican en el siguiente análisis, sin reiniciar.

| Sección | Clave | Descripción |
|---|---|---|
| `server` | `host`, `port` | dirección del servidor local |
| `video` | `preferred_height` / `fallback_height` | calidad buscada (1080) y mínimo esperado (720) |
| | `preferred_codecs` | orden de preferencia de códecs |
| `audio` | `prefer_original` | usar la pista de audio original |
| | `verify_language` | verificar el idioma del audio y corregir la pista si no coincide |
| `transcription` | `model` | modelo Whisper con GPU (`large-v3-turbo`, `medium`, `small`…) |
| | `cpu_model` | modelo si no hay GPU |
| | `device` | `auto`, `cuda` o `cpu` |
| | `compute_type` | `auto`, `int8`, `float16`… |
| `analysis` | `embedding_model` | modelo de embeddings multilingüe |
| | `sentence_max_pause` | pausa (s) que corta una frase |
| `clips` | `min_duration`, `max_duration` | duración buscada de los clips (20-60 s) |
| | `duration_flex` | desvío permitido si la narrativa lo justifica |
| | `max_candidates`, `min_candidates` | cuántos candidatos mostrar |
| | `min_score`, `relative_cutoff` | umbral absoluto y relativo al mejor candidato |
| | `max_overlap` | superposición máxima entre candidatos |
| | `start_padding`, `end_padding` | margen (s) antes y después del clip |
| `scoring` | `hook`, `impact`, `value`, `storytelling`, `autonomy`, `retention`, `boundaries` | **pesos** del puntaje total (único lugar donde se definen) |
| `youtube` | `cookies_file` | archivo de cookies de tu sesión (default `cookies.txt`), solo para videos con restricción de edad |
| | `cookies_from_browser` | alternativa: leer cookies de un navegador (`firefox`); Chrome/Edge en Windows no lo permiten |
| `crop` | `mode` | `smart` (rostros por plano) o `center` |
| | `sample_fps`, `face_min_confidence`, `shot_change_threshold`, `min_shot_duration` | ajustes del crop inteligente |
| `output` | `width`, `height` | resolución final (1080x1920) |
| | `video_codec`, `preset`, `crf`, `audio_bitrate` | calidad de codificación |

## Problemas frecuentes

- **"No se pudo descargar el video"**: revisá que sea público. Si YouTube cambió algo, `iniciar.bat`
  actualiza yt-dlp en cada inicio; también podés ejecutar `.venv\Scripts\pip install -U "yt-dlp[default]"`.
- **"El video tiene restricción de edad"**: YouTube solo los entrega a una cuenta logueada y mayor de 18.
  1. Instalá en Chrome la extensión **Get cookies.txt LOCALLY**.
  2. Abrí youtube.com con tu sesión iniciada y exportá las cookies (formato Netscape).
  3. Guardá el archivo como `cookies.txt` en la carpeta del proyecto (junto a `config.json`).

  La app solo usa las cookies cuando YouTube pide iniciar sesión; el resto de los videos se descargan
  sin sesión. Si después de un tiempo vuelve el error, las cookies vencieron: exportalas de nuevo.
  `cookies.txt` da acceso a tu cuenta: no lo compartas (ya está en `.gitignore`).
- **La transcripción es lenta**: sin GPU NVIDIA se usa la CPU. Podés usar un modelo más chico en
  `transcription.cpu_model` (`base`, `small`).
- **Errores técnicos**: el detalle completo queda en `logs/longtoshort.log`.

## Fuera del alcance de este MVP

Subtítulos, traducciones, música, efectos, zooms, B-roll, títulos/hashtags y publicación automática.
El código está separado por etapas (`pipeline.py`) para poder agregarlas después.
