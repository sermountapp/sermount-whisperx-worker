# sermount-whisperx-worker

RunPod Serverless worker with two modes on one image:

- **Align** (the live mode; any input without `"mode"`): tightens Sermount's word timestamps
  with WhisperX forced alignment (wav2vec2: `WAV2VEC2_ASR_BASE_960H` for English,
  `VOXPOPULI_ASR_BASE_10K_ES` for Spanish). **Timing only.** The words come from Sermount's
  stored transcript and go back out unchanged, in the same order, each with new start and end
  times.
- **Transcribe** (`"mode": "transcribe"`, added in v0.2.0 for the transcription bake-off):
  speech to text with word timings, through one of two engines on the same faster-whisper
  large-v3 weights. Nothing in the Sermount app calls it yet. See "Transcribe mode" below.

RunPod builds this repo from GitHub. Nothing is built or pushed from a laptop unless the
fallback below is needed.

## Contract (align mode)

Input (`POST /v2/{endpoint_id}/run`, body `{"input": ...}`):

```json
{"audio_url": "https://<short-lived signed URL>",
 "language": "en",
 "words": [{"i": 0, "w": "And", "s": 0.32, "e": 0.51}]}
```

Output (the handler never raises; failures come back as `aligned: false`):

```json
{"aligned": true, "reason": null,
 "words": [{"i": 0, "s": 0.29, "e": 0.47, "score": 0.91}],
 "model": "WAV2VEC2_ASR_BASE_960H", "image_tag": "v0.1.1", "exec_ms": 2140,
 "word_count": 1, "scored_count": 1, "segment_count": 1, "fallback_segments": 0,
 "audio_secs": 11.0}
```

- Output holds exactly the input's `i` values, in input order. No word is added, dropped or
  reordered, and the output carries no word text.
- A word the aligner cannot time keeps its incoming `s`/`e` with `score: null`. If WhisperX's
  words for a segment do not match the input one for one, the whole segment keeps its
  incoming times (counted in `fallback_segments`).
- Words are grouped into segments of at most 30 s of audio, cut at the widest pause.
  Alignment runs one segment at a time (batch size 1).
- `language` is `en` (the default when absent) or `es`; `model` in the output names the
  wav2vec2 model that ran. An input without `"mode"` behaves exactly as it did before v0.2.0
  (`"mode": "align"` is the same thing, spelled out).
- `reason` on failure, one word: `input`, `language` (anything but `en`/`es`),
  `download`, `audio`, `mapping`, `error`.

## Transcribe mode

Input:

```json
{"mode": "transcribe", "audio_url": "https://<short-lived signed URL>",
 "language": "en", "engine": "whisperx",
 "options": {"chunk_size": 30, "batch_size": 16, "vad_onset": 0.5, "vad_offset": 0.363}}
```

- `language` is required, `en` or `es`, and pinned (never detected).
- `engine` is required: `whisperx` or `faster-whisper`.
- `options` is optional and `whisperx` only. Each key overrides one default below, within
  bounds: `chunk_size` integer 5..30, `batch_size` integer 1..32, `vad_onset` and
  `vad_offset` 0.05..0.95. Anything else is `reason: "options"`.

Output (the handler never raises; failures come back as `transcribed: false`):

```json
{"transcribed": true, "reason": null, "engine": "whisperx",
 "engine_id": "whisperx/large-v3@v0.2.0", "model": "large-v3", "language": "en",
 "segments": [{"start": 0.0, "end": 4.1, "text": "And so, my fellow Americans,"}],
 "words": [{"w": "And", "start": 0.21, "end": 0.38, "score": 0.91}],
 "settings": {"path": "batched", "chunk_size": 30, "...": "every effective setting"},
 "versions": {"whisperx": "3.8.6", "faster-whisper": "1.2.1", "ctranslate2": "4.8.2", "torch": "2.8.0"},
 "image_tag": "v0.2.0", "exec_ms": 2140, "download_ms": 120, "transcribe_ms": 1500,
 "align_ms": 400, "segment_count": 1, "word_count": 22, "scored_count": 22,
 "words_synthesised": 0, "vad_chunks": 1, "audio_secs": 11.0}
```

`reason` on failure: `input`, `language`, `engine`, `options`, `download`, `audio`, `error`.
`engine_id` is the string Sermount stores as the transcript's provenance
(`sermons.transcript_engine`); `versions` is read from the installed packages at runtime.

### The two engines

Both load the same weights: `Systran/faster-whisper-large-v3` (MIT), revision
`edaa852ec7e145841d8ffdb056a99866b5f0a478`, `model.bin` sha256 `69f74147…d4f1`, baked at
build time into `/models/faster-whisper-large-v3`. float16 on GPU, int8 on CPU. A difference
between the two engines is the pipeline, not the model.

**`whisperx` is the BATCHED path** (whisperx 3.8.6, the path WHC's SermonVault runs):

1. Pyannote VAD (the segmentation model shipped in `whisperx/assets`) marks speech, onset
   0.500 / offset 0.363.
2. Speech is merged into chunks of at most `chunk_size` = 30 s. **A stretch of continuous
   speech longer than 30 s is cut inside the speech**, at the point of lowest voice activity
   (pyannote `Binarize(max_duration=chunk_size)`). This is where the "words skipped at chunk
   edges" concern lives: each chunk is decoded on its own, with no text from the chunk before.
3. Chunks are decoded in batches of 16, each exactly once: beam 5, **no temperature
   fallback and no compression-ratio check** (whisperx accepts those options and does not
   apply them on this path), `condition_on_previous_text` false.
4. wav2vec2 forced alignment (the same models as align mode) times every word; `score` is the
   aligner's. **When ANY word in a segment comes back untimed** (no character in the aligner's
   dictionary, e.g. "1961" or a symbol, or the whole segment fails), **every word in that
   segment** keeps its text but gets evenly spread times across the segment and `score: null`,
   counted in `words_synthesised` (ported from WHC). No word is dropped. For Part C: count
   score-null words as present (they are not missing) but exclude them from every timing
   statistic, and report `words_synthesised` beside the missing-word figure.

Chunk size, batch size, onset and offset are whisperx's own defaults (batch 16 is WHC's
production value; whisperx leaves it unset). `options` changes them per job. WHC's re-decode
of looped chunks (its "E2" fix) is NOT ported: this is the plain batched path.

**`faster-whisper` is the SEQUENTIAL path** (faster-whisper 1.2.1,
`WhisperModel.transcribe(word_timestamps=True, vad_filter=True)`):

1. Silero VAD (shipped in `faster_whisper/assets`) drops silence: threshold 0.5,
   min silence 2000 ms, speech pad 400 ms, no max speech length. The speech that remains is
   joined and decoded as one stream.
2. Decoding walks the stream in Whisper's 30 s windows, one after another, **with the
   previous window's text as context** (`condition_on_previous_text` true), beam 5, and the
   temperature fallback (0.0 → 1.0) whenever a window fails the compression-ratio (2.4) or
   log-prob (−1.0) checks.
3. Word times come from Whisper's own cross-attention; `score` is the word probability. No
   wav2vec2 pass.

Every decoding option there is faster-whisper's default except `vad_filter`, which is off by
default and on here. These settings are fixed; `options` is refused for this engine.

Part B makes both paths runnable. It decides nothing: the bake-off's Part C measures missing
words for each against hand-checked reference transcripts.

### Privacy (transcribe mode)

A transcribe result **carries the transcript text**: it is the point of the mode. Logs still
carry counts and timings only. RunPod keeps a job's output for a while after it finishes, so
a transcribe job's text sits in RunPod's `/status` result until it expires. The trial runs on
its own endpoint, never the live alignment endpoint.

## Privacy

- The audio is downloaded into a temp dir that is deleted in a `finally`, on success and on
  every failure.
- Logs carry counts and timings only: never words, text, or URLs. WhisperX's own logger
  (which prints segment text on an alignment miss) is silenced, and the RunPod SDK runs at
  `RUNPOD_LOG_LEVEL=INFO` (its default, DEBUG, logs every handler output).
- No network volume. Model weights (two wav2vec2 aligners, faster-whisper large-v3, the two
  VAD models inside their packages) and NLTK sentence data are baked into the image at build
  time; a worker never downloads anything except the audio it is given.
- RunPod itself keeps a job's input and output for a while after it finishes (the `/status`
  result). In align mode the input holds the word list and a signed URL that expires, and the
  output holds no text. In transcribe mode the output holds the transcript (see above).

## Endpoint settings (RunPod console)

New Endpoint → Serverless → **GitHub repo** `sermountapp/sermount-whisperx-worker`,
branch `main`, Dockerfile path `Dockerfile`.

| Setting | Value |
|---|---|
| GPU | cheapest 16 GB or 24 GB tier offered (e.g. 16 GB A4000/A4500 or 24 GB L4/A5000 class) |
| Min workers (active) | 0 |
| Max workers | 2 |
| Idle timeout | default |
| Execution timeout | 1200 s |
| FlashBoot | on |
| Network volume | none |
| Environment variables | none needed |

**Transcription trial endpoint (v0.2.0):** a second endpoint, never the live one. Same
settings, except a 24 GB GPU (large-v3 in float16 plus a wav2vec2 aligner) and an execution
timeout sized to a full sermon. A push to `main` builds and redeploys the endpoint that
tracks `main` (see "Shipping a new version"), so decide which branch each endpoint tracks
before v0.2.0 reaches `main`. The image grows by about 3.4 GB (large-v3 plus the Spanish
aligner), which every cold start and every build pays.

The trial endpoint exists: `sermount-whisperx-trial`, cloned from the live endpoint and built
from branch `feat/transcribe-mode` @ `761ae25`. The live endpoint still tracks `main`.
The RunPod GitHub build ran 2026-11-19 20:04-20:16 UTC: 12 minutes, inside the 30-minute cap
(**that open question is closed**), a 13.3 GB image, model bake 21 s. Baked:
`faster-whisper-large-v3`, `nltk_data`, and both wav2vec2 bundles.

After the endpoint exists, create an API key scoped to this endpoint. It goes into the
Render **worker** service as `RUNPOD_API_KEY` (`sync: false`) together with
`RUNPOD_ALIGN_ENDPOINT_ID`. Never into the web service, a file, or a chat.

## Shipping a new version

What starts a build, as far as it is known (2026-10-08):

- **The first build** starts on its own once the endpoint is connected to the repo. RunPod's
  docs say the image is built "automatically" on deploy; in practice the console showed "Push
  a commit to main to start a build" while the repo was empty, and the build started on the
  first push to `main`, before any release existed.
- **Later updates:** RunPod's docs say pushes do *not* update the endpoint ("they won't
  automatically be pushed to your endpoint... create a new release"), and that a new commit
  plus a release becomes the active build and supersedes a rollback. The console wording
  suggests a push to `main` builds. These disagree and the next change settles it: push the
  commit, then look at the endpoint's **Builds** tab *before* publishing a release. If a
  build started, a push deploys; if not, the release does. Record the answer here.
- **What a release does for us either way:** it is the version label. Its tag matches
  `IMAGE_TAG` in `handler.py`, which every response reports as `image_tag`, so the Owner
  Console shows which worker version aligned each sermon.

Steps:

1. Bump `IMAGE_TAG` in `handler.py` to the new tag (for example `v0.1.1`) and commit to `main`.
2. Push. Check Builds (see above).
3. GitHub → Releases → Draft a new release → tag `v0.1.1` on `main` → Publish.
4. RunPod console → the endpoint → Builds: wait for Building → Testing → Completed. Workers
   roll over to the new build on their own.
5. Run the console test (below) and check `image_tag` in the output.

**Live tag:** `v0.1.0` (release published 2026-10-08, `main` @ `6a05942`). Update this line
on every release.

## Rolling back

1. RunPod console → endpoint → Builds → the previous build's `⋯` menu → **Rollback**. The
   endpoint stays there until a newer build becomes active (see "Shipping a new version").
2. If old workers keep serving, scale workers to 0 (or delete the running workers) so they
   restart on the rolled-back build.
3. Fix forward with a new version; do not move or delete existing tags.

On the Sermount side the faster rollback is the app's `ALIGNMENT_ENABLED` flag on the
Render worker: off means no requests reach this endpoint at all.

## Console test

`test_input.json` aligns `samples/jfk.wav` (11 s, JFK's 1961 inaugural address, a US
government work in the public domain) from deliberately rough word times. Paste its
contents into the endpoint's **Requests** tab and run. Expect `aligned: true`, 22 words, and
times that move by tens to hundreds of milliseconds.

Result of the first console run (2026-10-08, build of `main` @ `6a05942`, `v0.1.0`):

```text
status: COMPLETED   delayTime: 15705 ms (cold start)   workerId: [redacted]
output: aligned: true, 22/22 words scored, image_tag: v0.1.0, exec_ms: 1305
```

First real sermon on staging (sermon 44, 2026-10-08): 2712 of 2734 word starts moved,
median shift 17 ms, p95 391 ms, execution 9.4 s, queue delay 1.1 s. It ran on the
24 GB PRO 6000 MIG tier because the 16 GB tier was low on supply.

## Local checks

```bash
python -m pytest -q tests          # shape tests for both modes; WhisperX, faster-whisper, torch and network are faked
```

Before the first release the real aligner ran (no mocks, `--network none`, so nothing was
downloaded at runtime) in a CPU container built from the same `requirements.txt`:

- `test_input.json` clip: 22/22 words scored, one segment, temp dir removed.
- JFK clip repeated 6 times as a 71 s 64 kbps MP3, every word start jittered by up to ±0.5 s,
  one start == end row: 132/132 scored in 3 segments, 0 fallback segments, median start error
  262 ms in, 7 ms out (p95 35 ms, max 77 ms), every `i` preserved, collapsed row re-timed.
  Align time on CPU was 7.3 s for 71 s of audio; GPU is expected to be far faster.

v0.2.0 ran for real (no mocks) on 2026-11-19, using the v0.1.1 image's own packages with
`download_models.py` baked into a volume and the jobs run with `--network none`. That was a
CPU under amd64 emulation on a Mac, int8, so the times only show that it runs, not how fast:

| Job | Result |
|---|---|
| align, en, `test_input.json` | 22/22 scored, `WAV2VEC2_ASR_BASE_960H` |
| align, es, 17 s synthetic Spanish clip, rough times | 28/28 scored, `VOXPOPULI_ASR_BASE_10K_ES` |
| transcribe `whisperx`, en, jfk.wav | 22 words, 22 scored, 1 VAD chunk, text exact |
| transcribe `whisperx`, en, `chunk_size: 5` | 22 words, 3 VAD chunks; a chunk edge put a full stop mid-sentence ("ask not. what") |
| transcribe `faster-whisper`, en, jfk.wav | 22 words, 22 scored, text exact |
| transcribe `whisperx` / `faster-whisper`, es clip | 29 / 28 words; both mishear the synthetic voice in different places |

The Spanish clip was made with macOS text-to-speech and is not committed. It only shows that
the Spanish models load and run; it says nothing about accuracy.

**First GPU run** (2026-11-19, trial endpoint, RTX 4090, `compute_type` float16,
`samples/jfk.wav`, 11 s). Versions reported: ctranslate2 4.8.2, faster-whisper 1.2.1,
torch 2.8.0, whisperx 3.8.6. CTranslate2 found cuDNN on the GPU (**the cuDNN open question
is closed**):

| Job | Result |
|---|---|
| transcribe `faster-whisper`, en | 22 words, text exact, transcribe_ms 641, exec_ms 7678 (first load), delayTime 13557 (cold start) |
| transcribe `whisperx`, en, defaults | 22 words, text exact, 1 VAD chunk, 2 segments, transcribe_ms 745, align_ms 169, exec_ms 8059 |
| transcribe `whisperx`, en, `chunk_size: 5` | 22 words, text exact, 3 VAD chunks, cut at "ask not \| what your country", no word dropped, align_ms 243 |

The stray full stop the CPU run put at that `chunk_size: 5` cut ("ask not. what") did not
appear on the GPU (float16, not int8). Punctuation at a chunk edge is not stable across
runs or compute types, so it is not evidence either way.

**Timings rule (user decision, bake-off brief).** Word times from the sequential path
(Whisper's own cross-attention) are not good enough for cuts. On jfk.wav it timed "ask" at
2.26-3.52 s (1.26 s, swallowing the pause before it), where the wav2vec2 aligner gave
3.548-3.788 s (0.24 s); and "And" at 0.0 where the aligner gave 0.421. So whichever engine
wins on text, Edit Mode's word timings keep coming from the aligner (align mode).

## Fallback: RunPod's builder rejects the image

The GitHub builder caps the docker build step at 30 minutes and images at 80 GB. If the
build times out or fails on size, build here and push to GHCR, then point the endpoint at
the image instead of the repo:

```bash
docker buildx build --platform linux/amd64 -t ghcr.io/sermountapp/sermount-whisperx-worker:v0.1.0 --push .
```

## Pins

- Base image: `runpod/base:1.4.0-cuda1281-ubuntu2204`, pinned by digest in the Dockerfile.
- Python 3.11 venv, every package pinned in `requirements.txt` and installed with
  `--no-deps`, then `pip check`. The lock was resolved for linux/amd64 + Python 3.11 from
  `whisperx==3.8.6 torch==2.8.0 torchaudio==2.8.0 runpod==1.12.0 requests`.
- To change a version, re-resolve the whole lock the same way (in a `python:3.11-slim`
  linux/amd64 container with `pip install --dry-run --report`), never by hand-editing one line.
