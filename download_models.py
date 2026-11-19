"""Build-time only: bake every model the worker can use into the image so a cold
start never touches the network.

- Alignment (wav2vec2, torchaudio): English WAV2VEC2_ASR_BASE_960H and Spanish
  VOXPOPULI_ASR_BASE_10K_ES, the two names whisperx 3.8.6 maps "en" and "es" to
  (whisperx/alignment.py DEFAULT_ALIGN_MODELS_TORCH).
- NLTK punkt_tab (whisperx's sentence splitter; the package holds Spanish too).
- Transcription: faster-whisper large-v3 (Systran/faster-whisper-large-v3, MIT),
  pinned to one Hub revision and checked by sha256, written as a plain directory
  so WhisperModel loads it by path with no Hub lookup at runtime. Both transcribe
  engines (whisperx batched, faster-whisper sequential) use these same weights.
  The VAD models need no step: whisperx ships its pyannote segmentation model in
  whisperx/assets/pytorch_model.bin, faster-whisper ships Silero in
  faster_whisper/assets/silero_vad_v6.onnx.
"""

import hashlib
import os

# The runpod/base image sets HF_HUB_ENABLE_HF_TRANSFER=1 but the lock does not
# carry hf_transfer, so huggingface_hub refuses every download (found on the first
# real bake, 2026-11-19). Off for this script only, before huggingface_hub is
# imported (it reads the flag at import).
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import nltk  # noqa: E402
import torchaudio  # noqa: E402
from faster_whisper.utils import download_model  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
NLTK_DATA = os.environ.get("NLTK_DATA", os.path.join(MODEL_DIR, "nltk_data"))

ALIGN_BUNDLES = ("WAV2VEC2_ASR_BASE_960H", "VOXPOPULI_ASR_BASE_10K_ES")

WHISPER_REPO = "large-v3"  # faster_whisper.utils._MODELS -> Systran/faster-whisper-large-v3
WHISPER_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"  # pragma: allowlist secret (Hub commit id, public)
WHISPER_MODEL_BIN_SHA256 = "69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1"  # pragma: allowlist secret (public file hash)
WHISPER_DIR = os.environ.get("WHISPER_DIR", os.path.join(MODEL_DIR, "faster-whisper-large-v3"))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


for name in ALIGN_BUNDLES:
    getattr(torchaudio.pipelines, name).get_model(dl_kwargs={"model_dir": MODEL_DIR})

assert nltk.download("punkt_tab", download_dir=NLTK_DATA, quiet=True), "punkt_tab download failed"

download_model(WHISPER_REPO, output_dir=WHISPER_DIR, revision=WHISPER_REVISION)
got = _sha256(os.path.join(WHISPER_DIR, "model.bin"))
assert got == WHISPER_MODEL_BIN_SHA256, f"large-v3 model.bin sha256 {got} != pinned"

print("baked:", sorted(os.listdir(MODEL_DIR)))
