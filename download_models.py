"""Build-time only: bake the English alignment model and NLTK sentence data
into the image so a cold start never touches the network."""

import os

import nltk
import torchaudio

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
NLTK_DATA = os.environ.get("NLTK_DATA", os.path.join(MODEL_DIR, "nltk_data"))

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
bundle.get_model(dl_kwargs={"model_dir": MODEL_DIR})
assert nltk.download("punkt_tab", download_dir=NLTK_DATA, quiet=True), "punkt_tab download failed"
print("baked:", sorted(os.listdir(MODEL_DIR)))
