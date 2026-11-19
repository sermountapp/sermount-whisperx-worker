"""RunPod serverless handler. Two modes on one image.

ALIGN (an input without "mode", or "mode": "align"): WhisperX forced alignment of
an existing word list. This is the live mode Sermount's alignment poll calls.

Input:
    {"audio_url": "https://...", "language": "en",
     "words": [{"i": 0, "w": "And", "s": 0.32, "e": 0.51}, ...]}

Output (always; the handler never raises):
    {"aligned": true|false, "reason": null|"<word>",
     "words": [{"i": 0, "s": 0.29, "e": 0.47, "score": 0.91}, ...],
     "model": "...", "image_tag": "...", "exec_ms": 1234, ...counts}

Timing only. Alignment never adds, removes, renames or reorders a word: the
output holds exactly the input's "i" values in input order. A word the aligner
cannot time keeps its incoming times and gets "score": null.

TRANSCRIBE ("mode": "transcribe"): speech to text with word timings, for the
transcription bake-off. Two engines on the same baked faster-whisper large-v3
weights, so a difference between them is the pipeline, not the model:

    {"mode": "transcribe", "audio_url": "https://...", "language": "en"|"es",
     "engine": "whisperx"|"faster-whisper", "options": {...}}   # options: per engine, bounded

    whisperx       = BATCHED. Pyannote VAD cuts the audio into chunks of at most
                     `chunk_size` s, each chunk is decoded once (no temperature
                     fallback), then wav2vec2 alignment times the words.
    faster-whisper = SEQUENTIAL. WhisperModel.transcribe(word_timestamps=True,
                     vad_filter=True): Silero VAD drops silence, the remaining
                     speech is decoded window by window with the previous text as
                     context and the temperature fallback on; word times come from
                     Whisper's own cross-attention.

Output: {"transcribed": true|false, "reason": ..., "language": ...,
         "segments": [{"start","end","text"}], "words": [{"w","start","end","score"}],
         "engine": ..., "engine_id": "<engine>/large-v3@<image_tag>",
         "settings": {...every effective setting...}, "versions": {...}, counts}

Unlike alignment, a transcribe result CARRIES TEXT (that is what it is for).

Privacy: the audio lives in a temp dir that is deleted in a `finally`, and
nothing here logs words, text or URLs, in either mode. Only counts and timings
are logged.
"""

import logging
import math
import os
import shutil
import tempfile
import time

import requests

IMAGE_TAG = "v0.2.1"  # bump with every GitHub release; see README "Cutting a release"

# The torchaudio bundles whisperx 3.8.6 maps each language to
# (whisperx/alignment.py DEFAULT_ALIGN_MODELS_TORCH); both baked by download_models.py.
ALIGN_MODELS = {"en": "WAV2VEC2_ASR_BASE_960H", "es": "VOXPOPULI_ASR_BASE_10K_ES"}
ALIGN_MODEL = ALIGN_MODELS["en"]
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
SUPPORTED_LANGUAGES = set(ALIGN_MODELS)

# Transcribe mode. One set of weights for both engines, baked as a plain
# directory (download_models.py pins the Hub revision and sha256).
WHISPER_MODEL = "large-v3"
WHISPER_DIR = os.environ.get("WHISPER_DIR", os.path.join(MODEL_DIR, "faster-whisper-large-v3"))
ENGINES = ("whisperx", "faster-whisper")

# whisperx batched path. chunk_size / vad_onset / vad_offset are whisperx 3.8.6's
# own defaults (asr.py load_model default_vad_options, FasterWhisperPipeline.transcribe
# chunk_size=30); batch_size 16 is WHC's production value (whisperx leaves it unset).
# A job may override these within the bounds below, so the bake-off can test the
# "words lost at chunk edges" claim against a setting, not only the default.
WHISPERX_DEFAULTS = {"batch_size": 16, "chunk_size": 30, "vad_onset": 0.500, "vad_offset": 0.363}
WHISPERX_BOUNDS = {"batch_size": (int, 1, 32), "chunk_size": (int, 5, 30),
                   "vad_onset": (float, 0.05, 0.95), "vad_offset": (float, 0.05, 0.95)}

# faster-whisper sequential path. beam_size 5, word timestamps and VAD are fixed;
# every decoding option is faster-whisper 1.2.1's WhisperModel.transcribe default
# (temperature fallback 0.0..1.0, log_prob_threshold -1.0). vad_filter is OFF by
# default there and ON here; its VadOptions are the library defaults.
FASTER_WHISPER_SETTINGS = {
    "beam_size": 5, "word_timestamps": True, "vad_filter": True,
    "vad": {"threshold": 0.5, "min_silence_duration_ms": 2000, "speech_pad_ms": 400,
            "max_speech_duration_s": None, "min_speech_duration_ms": 0},
}
# v0.2.1: three per-job knobs against the sequential decoder's loops (sermon 44,
# 2026-11-19: "I want a car." x5). Defaults are the library's, so a job without
# options decodes exactly as v0.2.0 did. hallucination_silence_threshold (seconds)
# skips silent stretches longer than this when a segment looks hallucinated; it
# needs word_timestamps, which this path always has.
FASTER_WHISPER_DEFAULTS = {"condition_on_previous_text": True, "hallucination_silence_threshold": None,
                           "compression_ratio_threshold": 2.4}
FASTER_WHISPER_BOUNDS = {"condition_on_previous_text": (bool, None, None),
                         "hallucination_silence_threshold": ("float_or_none", 0.5, 10.0),
                         "compression_ratio_threshold": (float, 0.5, 5.0)}
OPTION_SPECS = {"whisperx": (WHISPERX_DEFAULTS, WHISPERX_BOUNDS),
                "faster-whisper": (FASTER_WHISPER_DEFAULTS, FASTER_WHISPER_BOUNDS)}
SAMPLE_RATE = 16000

MAX_SEGMENT_SECS = 30.0
EDGE_PAD_SECS = 0.5
MAX_AUDIO_BYTES = 1024 * 1024 * 1024
DOWNLOAD_TIMEOUT = (10, 120)  # connect, read (seconds between bytes)
DOWNLOAD_CHUNK = 1024 * 1024

# WhisperX logs "Failed to align segment ("<segment text>")". That is
# transcript text, so its logger is silenced before whisperx is imported.
# A handler being present also stops whisperx installing its own.
_wx_logger = logging.getLogger("whisperx")
_wx_logger.addHandler(logging.NullHandler())
_wx_logger.setLevel(logging.CRITICAL)
_wx_logger.propagate = False

# urllib3 (under requests) logs full request paths at DEBUG, and a signed
# URL's path carries its token. Pinned at WARNING whatever the root level is.
logging.getLogger("urllib3").setLevel(logging.WARNING)

log = logging.getLogger("aligner")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s aligner %(message)s"))
    log.addHandler(_h)
log.setLevel(logging.INFO)
log.propagate = False

_MODELS = {}  # language -> (align_model, metadata, device), loaded once per worker
_WHISPER = None  # (faster_whisper.WhisperModel, device, compute_type), loaded on first transcribe


class AlignError(Exception):
    """Expected failure; `reason` is the one-word code returned to the caller."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _failure(reason, started):
    return {
        "aligned": False,
        "reason": reason,
        "words": [],
        "model": ALIGN_MODEL,
        "image_tag": IMAGE_TAG,
        "exec_ms": int((time.monotonic() - started) * 1000),
    }


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_input(job_input):
    """Return (audio_url, language, words) or raise AlignError("input"/"language")."""
    if not isinstance(job_input, dict):
        raise AlignError("input")
    audio_url = job_input.get("audio_url")
    if not isinstance(audio_url, str) or not audio_url.startswith(("https://", "http://")):
        raise AlignError("input")
    language = job_input.get("language", "en")
    if not isinstance(language, str):
        raise AlignError("input")
    if language.lower() not in SUPPORTED_LANGUAGES:
        raise AlignError("language")
    raw = job_input.get("words")
    if not isinstance(raw, list) or not raw:
        raise AlignError("input")
    words, seen = [], set()
    for item in raw:
        if not isinstance(item, dict):
            raise AlignError("input")
        i, w, s, e = item.get("i"), item.get("w"), item.get("s"), item.get("e")
        if not isinstance(i, int) or isinstance(i, bool) or i in seen:
            raise AlignError("input")
        if not isinstance(w, str) or not _finite(s) or not _finite(e) or s < 0 or e < 0:
            raise AlignError("input")
        seen.add(i)
        words.append({"i": i, "w": w, "s": float(s), "e": float(max(s, e))})
    return audio_url, language.lower(), words


def token_for(word):
    """The text handed to the aligner for one word: no whitespace, never empty.

    WhisperX splits segment text on single spaces, so a token must not contain
    one. An empty token would be dropped by WhisperX and break the 1:1 mapping.
    """
    tok = "".join(word.split())
    return tok if tok else "-"


# --------------------------------------------------------------------------
# Segmenting
# --------------------------------------------------------------------------

def _split_point(words, lo, hi):
    """Index of the last word of the chunk words[lo..k], chosen at the widest
    pause in the second half of the window so a cut rarely lands mid-phrase."""
    start = words[lo]["s"]
    best_k, best_gap = hi, -1.0
    for k in range(lo, hi):
        if words[k]["e"] - start < MAX_SEGMENT_SECS / 2:
            continue
        gap = words[k + 1]["s"] - words[k]["e"] if k + 1 <= hi else 0.0
        if gap > best_gap:
            best_k, best_gap = k, gap
    return best_k


def build_segments(words, duration, max_secs=MAX_SEGMENT_SECS, pad=EDGE_PAD_SECS):
    """Group consecutive words into alignment segments of at most `max_secs`.

    Returns [{"start", "end", "first", "last"}] with word indexes inclusive.
    Windows never overlap: the edge between two segments sits in the pause
    between them, at most `pad` from either word.
    """
    if not words:
        return []
    chunks, lo, n = [], 0, len(words)
    while lo < n:
        hi = lo
        while hi + 1 < n and (words[hi + 1]["e"] + pad) - (words[lo]["s"] - pad) <= max_secs:
            hi += 1
        if hi + 1 < n and hi > lo:
            hi = _split_point(words, lo, hi)
        chunks.append((lo, hi))
        lo = hi + 1

    segments = []
    for idx, (lo, hi) in enumerate(chunks):
        first, last = words[lo], words[hi]
        if idx == 0:
            start = first["s"] - pad
        else:
            prev = words[chunks[idx - 1][1]]
            start = max((prev["e"] + first["s"]) / 2.0, first["s"] - pad)
        if idx == len(chunks) - 1:
            end = last["e"] + pad
        else:
            nxt = words[chunks[idx + 1][0]]
            end = min((last["e"] + nxt["s"]) / 2.0, last["e"] + pad)
        start = max(0.0, start)
        end = min(duration, end, start + max_secs)
        segments.append({"start": start, "end": end, "first": lo, "last": hi})
    return segments


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------

def _get_model(language="en"):
    if language not in _MODELS:
        import torch
        import whisperx

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, metadata = whisperx.load_align_model(
            language_code=language, device=device, model_name=ALIGN_MODELS[language], model_dir=MODEL_DIR
        )
        _MODELS[language] = (model, metadata, device)
    return _MODELS[language]


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def align_segment(words, seg, audio, model, metadata, device):
    """Align one segment. Returns (results, ok) with one result per input word.

    If WhisperX's words do not match our tokens one for one (count or text),
    the whole segment keeps its incoming times. Never guess a mapping.
    """
    import whisperx

    chunk = words[seg["first"]:seg["last"] + 1]
    tokens = [token_for(w["w"]) for w in chunk]
    fallback = [{"i": w["i"], "s": w["s"], "e": w["e"], "score": None} for w in chunk]
    if seg["end"] <= seg["start"]:
        return fallback, False

    result = whisperx.align(
        [{"start": seg["start"], "end": seg["end"], "text": " ".join(tokens)}],
        model, metadata, audio, device, return_char_alignments=False,
    )
    out = result.get("word_segments") or []
    if len(out) != len(chunk) or any(o.get("word") != t for o, t in zip(out, tokens)):
        return fallback, False

    results = []
    for w, o in zip(chunk, out):
        s, e, score = _num(o.get("start")), _num(o.get("end")), _num(o.get("score"))
        if s is None or e is None or score is None:
            results.append({"i": w["i"], "s": w["s"], "e": w["e"], "score": None})
        else:
            results.append({"i": w["i"], "s": round(s, 3), "e": round(e, 3), "score": round(score, 3)})
    return results, True


def _download(url, dest):
    try:
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as resp:
            if resp.status_code != 200:
                raise AlignError("download")
            total = 0
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(DOWNLOAD_CHUNK):
                    total += len(chunk)
                    if total > MAX_AUDIO_BYTES:
                        raise AlignError("download")
                    fh.write(chunk)
    except requests.RequestException:
        raise AlignError("download")
    if total == 0:
        raise AlignError("download")
    return total


def run(job_input):
    started = time.monotonic()
    tmp = None
    try:
        audio_url, language, words = validate_input(job_input)

        tmp = tempfile.mkdtemp(prefix="align-")
        audio_path = os.path.join(tmp, "audio")
        t0 = time.monotonic()
        size = _download(audio_url, audio_path)
        download_ms = int((time.monotonic() - t0) * 1000)

        import whisperx

        try:
            audio = whisperx.load_audio(audio_path)
        except Exception:
            raise AlignError("audio")
        duration = len(audio) / 16000.0
        if duration <= 0:
            raise AlignError("audio")

        model, metadata, device = _get_model(language)
        segments = build_segments(words, duration)
        t1 = time.monotonic()
        aligned, fallback_segments = [], 0
        for seg in segments:
            results, ok = align_segment(words, seg, audio, model, metadata, device)
            aligned.extend(results)
            if not ok:
                fallback_segments += 1
        align_ms = int((time.monotonic() - t1) * 1000)

        if [r["i"] for r in aligned] != [w["i"] for w in words]:
            raise AlignError("mapping")
        scored = sum(1 for r in aligned if r["score"] is not None)
        exec_ms = int((time.monotonic() - started) * 1000)
        log.info(
            "aligned words=%d scored=%d segments=%d fallback_segments=%d "
            "audio_s=%.1f bytes=%d download_ms=%d align_ms=%d exec_ms=%d device=%s",
            len(words), scored, len(segments), fallback_segments,
            duration, size, download_ms, align_ms, exec_ms, device,
        )
        return {
            "aligned": True,
            "reason": None,
            "words": aligned,
            "model": ALIGN_MODELS[language],
            "image_tag": IMAGE_TAG,
            "exec_ms": exec_ms,
            "word_count": len(words),
            "scored_count": scored,
            "segment_count": len(segments),
            "fallback_segments": fallback_segments,
            "audio_secs": round(duration, 3),
        }
    except AlignError as exc:
        log.info("not aligned reason=%s", exc.reason)
        return _failure(exc.reason, started)
    except Exception as exc:  # never raise to RunPod
        log.warning("not aligned reason=error type=%s", type(exc).__name__)
        return _failure("error", started)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# Transcribe mode
# --------------------------------------------------------------------------

class TranscribeError(Exception):
    """Expected transcribe failure; `reason` is the one-word code returned."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _transcribe_failure(reason, started, engine=None):
    return {
        "transcribed": False,
        "reason": reason,
        "engine": engine,
        "segments": [],
        "words": [],
        "model": WHISPER_MODEL,
        "image_tag": IMAGE_TAG,
        "exec_ms": int((time.monotonic() - started) * 1000),
    }


def validate_transcribe_input(job_input):
    """Return (audio_url, language, engine, options) or raise TranscribeError.

    language is required and pinned, never detected: detection costs a pass and
    can pick wrong on a quiet opening, and the bake-off scores each language.
    """
    audio_url = job_input.get("audio_url")
    if not isinstance(audio_url, str) or not audio_url.startswith(("https://", "http://")):
        raise TranscribeError("input")
    language = job_input.get("language")
    if not isinstance(language, str):
        raise TranscribeError("input")
    if language.lower() not in SUPPORTED_LANGUAGES:
        raise TranscribeError("language")
    engine = job_input.get("engine")
    if engine not in ENGINES:
        raise TranscribeError("engine")
    raw = job_input.get("options") or {}
    if not isinstance(raw, dict):
        raise TranscribeError("input")
    defaults, bounds = OPTION_SPECS[engine]
    options = dict(defaults)
    for key, value in raw.items():
        if key not in bounds:
            raise TranscribeError("options")
        options[key] = _bounded(value, *bounds[key])
    return audio_url, language.lower(), engine, options


def _bounded(value, kind, lo, hi):
    """One option value checked against its kind and bounds, or TranscribeError."""
    if kind is bool:
        if not isinstance(value, bool):
            raise TranscribeError("options")
        return value
    if kind == "float_or_none":
        if value is None:
            return None
        kind = float
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TranscribeError("options")
    if kind is int and value != int(value):
        raise TranscribeError("options")
    value = kind(value)
    if not math.isfinite(value) or not (lo <= value <= hi):
        raise TranscribeError("options")
    return value


def _get_whisper():
    """faster-whisper large-v3 from the baked directory, loaded on the first
    transcribe job only, so an alignment-only worker never holds it.

    whisperx.asr.WhisperModel, not faster_whisper.WhisperModel: whisperx's
    batched pipeline calls generate_segment_batched, which only the whisperx
    subclass has (a plain faster-whisper model fails with AttributeError, found
    on the first real run). The subclass's one other override, encode(), is
    faster-whisper 1.2.1's encode line for line, so its inherited transcribe()
    is the unmodified sequential path. One instance serves both engines."""
    global _WHISPER
    if _WHISPER is None:
        import torch
        from whisperx.asr import WhisperModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(WHISPER_DIR, device=device, compute_type=compute_type)
        _WHISPER = (model, device, compute_type)
    return _WHISPER


def _versions():
    from importlib import metadata

    out = {}
    for name in ("whisperx", "faster-whisper", "ctranslate2", "torch"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _segment_words(seg):
    """Words for one aligned whisperx segment, every word accounted for.

    whisperx keeps a segment's text but returns no (or partial) word timings when
    alignment fails for it (no character in the model's dictionary, a start past
    the audio). Dropping those words would understate what the engine heard,
    which is exactly what the bake-off measures, so the segment's span is spread
    evenly across its tokens with score None and the count is reported.
    Ported from WHC runpod_worker/transcribe_core.py _segment_words.
    """
    text = (seg.get("text") or "").strip()
    raw = [w for w in (seg.get("words") or []) if (w.get("word") or "").strip()]
    timed = [w for w in raw if _num(w.get("start")) is not None and _num(w.get("end")) is not None]
    if raw and len(timed) == len(raw):
        return [{"w": w["word"].strip(), "start": round(_num(w["start"]), 3), "end": round(_num(w["end"]), 3),
                 "score": None if _num(w.get("score")) is None else round(_num(w["score"]), 3)}
                for w in timed], 0
    tokens = text.split()
    if not tokens:
        return [], 0
    start = _num(seg.get("start")) or 0.0
    end = _num(seg.get("end")) or start
    if not end > start:
        end = start + 0.001 * len(tokens)
    step = (end - start) / len(tokens)
    return [{"w": t, "start": round(start + k * step, 3), "end": round(start + (k + 1) * step, 3), "score": None}
            for k, t in enumerate(tokens)], len(tokens)


def transcribe_whisperx(audio, language, options):
    """Batched: pyannote VAD chunks -> one decode per chunk -> wav2vec2 alignment."""
    import whisperx

    whisper, device, compute_type = _get_whisper()
    pipeline = whisperx.load_model(
        WHISPER_MODEL, device, compute_type=compute_type, language=language, model=whisper,
        vad_method="pyannote",
        vad_options={"chunk_size": options["chunk_size"], "vad_onset": options["vad_onset"],
                     "vad_offset": options["vad_offset"]},
    )
    t0 = time.monotonic()
    result = pipeline.transcribe(audio, batch_size=options["batch_size"], language=language,
                                 chunk_size=options["chunk_size"])
    raw_segments = result.get("segments") or []
    transcribe_ms = int((time.monotonic() - t0) * 1000)

    align_model, metadata, _ = _get_model(language)
    t1 = time.monotonic()
    aligned = whisperx.align(raw_segments, align_model, metadata, audio, device, return_char_alignments=False)
    align_ms = int((time.monotonic() - t1) * 1000)

    segments, words, synthesised = [], [], 0
    for seg in aligned.get("segments") or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        seg_words, synth = _segment_words(seg)
        synthesised += synth
        start = _num(seg.get("start"))
        end = _num(seg.get("end"))
        start = start if start is not None else (seg_words[0]["start"] if seg_words else 0.0)
        end = end if end is not None else (seg_words[-1]["end"] if seg_words else start)
        segments.append({"start": round(start, 3), "end": round(end, 3), "text": text})
        words.extend(seg_words)
    settings = {"path": "batched", "vad": "pyannote (whisperx/assets)", **options,
                "decode": "one pass per VAD chunk, no temperature fallback",
                "align_model": ALIGN_MODELS[language], "compute_type": compute_type}
    stats = {"vad_chunks": len(raw_segments), "words_synthesised": synthesised,
             "transcribe_ms": transcribe_ms, "align_ms": align_ms}
    return segments, words, settings, stats


def transcribe_faster_whisper(audio, language, options=None):
    """Sequential: Silero VAD drops silence, then window-by-window decoding with
    word timestamps from Whisper itself. No wav2vec2 pass. `options` are the
    validated FASTER_WHISPER_DEFAULTS keys; every one is passed explicitly."""
    options = dict(FASTER_WHISPER_DEFAULTS, **(options or {}))
    whisper, _device, compute_type = _get_whisper()
    t0 = time.monotonic()
    seg_iter, _info = whisper.transcribe(
        audio, language=language, beam_size=FASTER_WHISPER_SETTINGS["beam_size"],
        word_timestamps=True, vad_filter=True,
        condition_on_previous_text=options["condition_on_previous_text"],
        hallucination_silence_threshold=options["hallucination_silence_threshold"],
        compression_ratio_threshold=options["compression_ratio_threshold"],
    )
    segments, words = [], []
    for seg in seg_iter:  # a generator: decoding happens here
        text = (seg.text or "").strip()
        if not text:
            continue
        segments.append({"start": round(float(seg.start), 3), "end": round(float(seg.end), 3), "text": text})
        for w in seg.words or []:
            token = (w.word or "").strip()
            if not token:
                continue
            words.append({"w": token, "start": round(float(w.start), 3), "end": round(float(w.end), 3),
                          "score": None if _num(w.probability) is None else round(float(w.probability), 3)})
    transcribe_ms = int((time.monotonic() - t0) * 1000)
    settings = {"path": "sequential", "vad": "silero (faster_whisper/assets)",
                **FASTER_WHISPER_SETTINGS, **options, "compute_type": compute_type}
    stats = {"words_synthesised": 0, "transcribe_ms": transcribe_ms, "align_ms": 0}
    return segments, words, settings, stats


def run_transcribe(job_input):
    started = time.monotonic()
    tmp = None
    engine = job_input.get("engine") if isinstance(job_input, dict) else None
    engine = engine if engine in ENGINES else None
    try:
        audio_url, language, engine, options = validate_transcribe_input(job_input)

        tmp = tempfile.mkdtemp(prefix="transcribe-")
        audio_path = os.path.join(tmp, "audio")
        t0 = time.monotonic()
        try:
            size = _download(audio_url, audio_path)
        except AlignError as exc:
            raise TranscribeError(exc.reason)
        download_ms = int((time.monotonic() - t0) * 1000)

        import whisperx

        try:
            audio = whisperx.load_audio(audio_path)
        except Exception:
            raise TranscribeError("audio")
        duration = len(audio) / float(SAMPLE_RATE)
        if duration <= 0:
            raise TranscribeError("audio")

        if engine == "whisperx":
            segments, words, settings, stats = transcribe_whisperx(audio, language, options)
        else:
            segments, words, settings, stats = transcribe_faster_whisper(audio, language, options)

        scored = sum(1 for w in words if w["score"] is not None)
        exec_ms = int((time.monotonic() - started) * 1000)
        log.info(
            "transcribed engine=%s language=%s segments=%d words=%d scored=%d synthesised=%d "
            "audio_s=%.1f bytes=%d download_ms=%d transcribe_ms=%d align_ms=%d exec_ms=%d",
            engine, language, len(segments), len(words), scored, stats["words_synthesised"],
            duration, size, download_ms, stats["transcribe_ms"], stats["align_ms"], exec_ms,
        )
        return {
            "transcribed": True,
            "reason": None,
            "engine": engine,
            "engine_id": f"{engine}/{WHISPER_MODEL}@{IMAGE_TAG}",
            "model": WHISPER_MODEL,
            "language": language,
            "segments": segments,
            "words": words,
            "settings": settings,
            "versions": _versions(),
            "image_tag": IMAGE_TAG,
            "exec_ms": exec_ms,
            "download_ms": download_ms,
            "segment_count": len(segments),
            "word_count": len(words),
            "scored_count": scored,
            "audio_secs": round(duration, 3),
            **stats,
        }
    except TranscribeError as exc:
        log.info("not transcribed reason=%s", exc.reason)
        return _transcribe_failure(exc.reason, started, engine)
    except Exception as exc:  # never raise to RunPod
        log.warning("not transcribed reason=error type=%s", type(exc).__name__)
        return _transcribe_failure("error", started, engine)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def handler(job):
    job_input = (job or {}).get("input")
    if isinstance(job_input, dict) and job_input.get("mode") == "transcribe":
        return run_transcribe(job_input)
    # No "mode" (today's callers), "align", or anything else: the alignment
    # path, unchanged; it rejects what it cannot read.
    return run(job_input)


if __name__ == "__main__":
    import runpod

    _get_model()  # load before taking jobs, so FlashBoot snapshots a warm worker
    runpod.serverless.start({"handler": handler})
