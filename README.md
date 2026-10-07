# sermount-whisperx-worker

RunPod Serverless worker that tightens Sermount's word timestamps with WhisperX forced
alignment (wav2vec2, `WAV2VEC2_ASR_BASE_960H`). **Timing only.** It never transcribes: the
words come from Sermount's stored whisper-1 transcript and go back out unchanged, in the same
order, each with new start and end times.

RunPod builds this repo from GitHub. Nothing is built or pushed from a laptop unless the
fallback below is needed.

## Contract

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
 "model": "WAV2VEC2_ASR_BASE_960H", "image_tag": "v0.1.0", "exec_ms": 2140,
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
- `reason` on failure, one word: `input`, `language` (anything but English in v1),
  `download`, `audio`, `mapping`, `error`.

## Privacy

- The audio is downloaded into a temp dir that is deleted in a `finally`, on success and on
  every failure.
- Logs carry counts and timings only: never words, text, or URLs. WhisperX's own logger
  (which prints segment text on an alignment miss) is silenced, and the RunPod SDK runs at
  `RUNPOD_LOG_LEVEL=INFO` (its default, DEBUG, logs every handler output).
- No network volume. Model weights and NLTK sentence data are baked into the image at build
  time; a worker never downloads anything except the audio it is given.
- RunPod itself keeps a job's input and output for a while after it finishes (the `/status`
  result). The input holds the word list and a signed URL that expires; the output holds no text.

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

After the endpoint exists, create an API key scoped to this endpoint. It goes into the
Render **worker** service as `RUNPOD_API_KEY` (`sync: false`) together with
`RUNPOD_ALIGN_ENDPOINT_ID`. Never into the web service, a file, or a chat.

## Cutting a release (this is how a new version goes live)

RunPod does not deploy on push. A **GitHub release** triggers a build.

1. Bump `IMAGE_TAG` in `handler.py` to the new tag (for example `v0.1.1`) and commit to `main`.
2. GitHub → Releases → Draft a new release → tag `v0.1.1` on `main` → Publish.
3. RunPod console → the endpoint → Builds: wait for Building → Testing → Completed. Workers
   roll over to the new build on their own.
4. Run the console test (below) and check `image_tag` in the output.

**Live tag:** none yet (first release will be `v0.1.0`). Update this line on every release.

## Rolling back

1. RunPod console → endpoint → Builds → the previous build's `⋯` menu → **Rollback**. The
   endpoint stays there until the next release is published.
2. If old workers keep serving, scale workers to 0 (or delete the running workers) so they
   restart on the rolled-back build.
3. Fix forward with a new release; do not move or delete existing tags.

On the Sermount side the faster rollback is the app's `ALIGNMENT_ENABLED` flag on the
Render worker: off means no requests reach this endpoint at all.

## Console test

`test_input.json` aligns `samples/jfk.wav` (11 s, JFK's 1961 inaugural address, a US
government work in the public domain) from deliberately rough word times. Paste its
contents into the endpoint's **Requests** tab and run. Expect `aligned: true`, 22 words, and
times that move by tens to hundreds of milliseconds.

Result of the first console run: _pending (paste the output here after the first build)._

## Local checks

```bash
python -m pytest -q tests          # handler shape tests; WhisperX, torch and network are faked
```

Before the first release the real aligner ran (no mocks, `--network none`, so nothing was
downloaded at runtime) in a CPU container built from the same `requirements.txt`:

- `test_input.json` clip: 22/22 words scored, one segment, temp dir removed.
- JFK clip repeated 6 times as a 71 s 64 kbps MP3, every word start jittered by up to ±0.5 s,
  one start == end row: 132/132 scored in 3 segments, 0 fallback segments, median start error
  262 ms in, 7 ms out (p95 35 ms, max 77 ms), every `i` preserved, collapsed row re-timed.
  Align time on CPU was 7.3 s for 71 s of audio; GPU is expected to be far faster.

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
