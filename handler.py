"""RunPod serverless handler: WhisperX forced alignment of an existing word list.

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

Privacy: the audio lives in a temp dir that is deleted in a `finally`, and
nothing here logs words, text or URLs. Only counts and timings are logged.
"""

import logging
import math
import os
import shutil
import tempfile
import time

import requests

IMAGE_TAG = "v0.1.0"  # bump with every GitHub release; see README "Cutting a release"

ALIGN_MODEL = "WAV2VEC2_ASR_BASE_960H"
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
SUPPORTED_LANGUAGES = {"en"}

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

_MODEL = None  # (align_model, metadata, device), loaded once per worker


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

def _get_model():
    global _MODEL
    if _MODEL is None:
        import torch
        import whisperx

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, metadata = whisperx.load_align_model(
            language_code="en", device=device, model_name=ALIGN_MODEL, model_dir=MODEL_DIR
        )
        _MODEL = (model, metadata, device)
    return _MODEL


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
        audio_url, _language, words = validate_input(job_input)

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

        model, metadata, device = _get_model()
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
            "model": ALIGN_MODEL,
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


def handler(job):
    return run((job or {}).get("input"))


if __name__ == "__main__":
    import runpod

    _get_model()  # load before taking jobs, so FlashBoot snapshots a warm worker
    runpod.serverless.start({"handler": handler})
