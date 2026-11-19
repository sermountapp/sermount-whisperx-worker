"""Transcribe-mode shape tests. WhisperX, faster-whisper, torch and the network are
faked; the real engines are exercised on samples/jfk.wav as described in the
README ("Local checks")."""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import handler  # noqa: E402

SR = 16000
SAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples", "jfk.wav")


class FakeAudio:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class FakePipeline:
    def __init__(self, fake):
        self.fake = fake

    def transcribe(self, audio, batch_size=None, language=None, chunk_size=30):
        self.fake.transcribe_calls.append({"batch_size": batch_size, "language": language,
                                           "chunk_size": chunk_size})
        if self.fake.mode == "raise":
            raise RuntimeError("cuda exploded")
        return {"segments": [{"start": 0.0, "end": 4.0, "text": " And so my fellow Americans,"},
                             {"start": 5.0, "end": 9.0, "text": " ask not what your country"}],
                "language": language}


class FakeWhisperX(types.ModuleType):
    def __init__(self):
        super().__init__("whisperx")
        self.mode = "ok"
        self.load_model_calls = []
        self.transcribe_calls = []
        self.align_calls = []

    def load_audio(self, path):
        assert os.path.exists(path)
        return FakeAudio(SR * 11)

    def load_model(self, arch, device, **kw):
        self.load_model_calls.append({"arch": arch, "device": device, **kw})
        return FakePipeline(self)

    def align(self, segments, model, metadata, audio, device, return_char_alignments=False):
        self.align_calls.append({"model": model, "language": metadata["language"]})
        out = []
        for k, seg in enumerate(segments):
            tokens = seg["text"].split()
            if self.mode == "unaligned" and k == 1:
                out.append({"start": seg["start"], "end": seg["end"], "text": seg["text"], "words": []})
                continue
            words = [{"word": t, "start": seg["start"] + i * 0.5, "end": seg["start"] + i * 0.5 + 0.4,
                      "score": 0.875} for i, t in enumerate(tokens)]
            out.append({"start": seg["start"], "end": seg["end"], "text": seg["text"], "words": words})
        return {"segments": out, "word_segments": [w for s in out for w in s["words"]]}


class FakeWord:
    def __init__(self, word, start, end, probability):
        self.word, self.start, self.end, self.probability = word, start, end, probability


class FakeSeg:
    def __init__(self, start, end, text, words):
        self.start, self.end, self.text, self.words = start, end, text, words


class FakeWhisperModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)

        def gen():
            yield FakeSeg(0.0, 2.1, " And so, my fellow",
                          [FakeWord(" And", 0.0, 0.3, 0.91), FakeWord(" so,", 0.3, 0.6, 0.88),
                           FakeWord(" my", 0.7, 0.9, 0.97), FakeWord(" fellow", 0.9, 1.4, 0.95)])
            yield FakeSeg(2.2, 2.3, "  ", [])
            yield FakeSeg(3.0, 4.0, " Americans.", [FakeWord(" Americans.", 3.0, 3.9, 0.99)])

        return gen(), types.SimpleNamespace(language=kw.get("language"))


@pytest.fixture
def env(monkeypatch, tmp_path):
    fake = FakeWhisperX()
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    whisper = FakeWhisperModel()
    monkeypatch.setattr(handler, "_get_whisper", lambda: (whisper, "cpu", "int8"))
    loaded = []

    def fake_get_model(language="en"):
        loaded.append(language)
        return (f"align-{language}", {"language": language}, "cpu")

    monkeypatch.setattr(handler, "_get_model", fake_get_model)
    downloads = []

    def fake_download(url, dest):
        downloads.append(url)
        with open(SAMPLE, "rb") as src, open(dest, "wb") as fh:
            data = src.read()
            fh.write(data)
        return len(data)

    monkeypatch.setattr(handler, "_download", fake_download)
    made = []
    real_mkdtemp = handler.tempfile.mkdtemp

    def mkdtemp(**kw):
        d = real_mkdtemp(dir=tmp_path, **kw)
        made.append(d)
        return d

    monkeypatch.setattr(handler.tempfile, "mkdtemp", mkdtemp)
    return types.SimpleNamespace(fake=fake, whisper=whisper, made=made, loaded=loaded, downloads=downloads)


def _job(engine, **extra):
    job_input = {"mode": "transcribe", "engine": engine, "language": "en",
                 "audio_url": "https://example.test/samples/jfk.wav"}
    job_input.update(extra)
    return {"id": "t", "input": job_input}


def test_sample_is_present():
    with open(SAMPLE, "rb") as fh:
        assert fh.read(4) == b"RIFF"


def test_whisperx_engine_is_batched_then_aligned(env):
    out = handler.handler(_job("whisperx"))
    assert out["transcribed"] is True and out["reason"] is None
    assert out["engine"] == "whisperx" and out["engine_id"] == f"whisperx/large-v3@{handler.IMAGE_TAG}"
    assert out["language"] == "en" and out["model"] == "large-v3"
    assert [s["text"] for s in out["segments"]] == ["And so my fellow Americans,", "ask not what your country"]
    assert all(set(w) == {"w", "start", "end", "score"} for w in out["words"])
    assert [w["w"] for w in out["words"]][:3] == ["And", "so", "my"]
    assert out["word_count"] == 10 == out["scored_count"] and out["words_synthesised"] == 0
    assert out["vad_chunks"] == 2 and out["audio_secs"] == 11.0
    # The library defaults reached whisperx, and the settings report says so.
    lm = env.fake.load_model_calls[0]
    assert lm["arch"] == "large-v3" and lm["language"] == "en" and lm["vad_method"] == "pyannote"
    assert lm["vad_options"] == {"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363}
    assert lm["model"] is env.whisper
    assert env.fake.transcribe_calls == [{"batch_size": 16, "language": "en", "chunk_size": 30}]
    assert env.fake.align_calls == [{"model": "align-en", "language": "en"}]
    st = out["settings"]
    assert st["path"] == "batched" and st["chunk_size"] == 30 and st["batch_size"] == 16
    assert st["align_model"] == "WAV2VEC2_ASR_BASE_960H"
    assert env.whisper.calls == []  # the sequential path never ran
    assert len(env.made) == 1 and not os.path.exists(env.made[0])


def test_whisperx_options_override_within_bounds(env):
    out = handler.handler(_job("whisperx", options={"chunk_size": 10, "vad_onset": 0.3, "batch_size": 4}))
    assert out["transcribed"] is True
    assert env.fake.load_model_calls[0]["vad_options"] == {"chunk_size": 10, "vad_onset": 0.3, "vad_offset": 0.363}
    assert env.fake.transcribe_calls[0]["chunk_size"] == 10 and env.fake.transcribe_calls[0]["batch_size"] == 4
    assert out["settings"]["chunk_size"] == 10 and out["settings"]["vad_onset"] == 0.3


def test_whisperx_unaligned_segment_keeps_its_words(env):
    env.fake.mode = "unaligned"
    out = handler.handler(_job("whisperx"))
    assert out["word_count"] == 10 and out["words_synthesised"] == 5 and out["scored_count"] == 5
    tail = out["words"][5:]
    assert [w["w"] for w in tail] == ["ask", "not", "what", "your", "country"]
    assert all(w["score"] is None for w in tail)
    assert tail[0]["start"] == 5.0 and tail[-1]["end"] == 9.0
    assert [s["text"] for s in out["segments"]][1] == "ask not what your country"


def test_faster_whisper_engine_is_sequential_with_vad(env):
    out = handler.handler(_job("faster-whisper", language="es"))
    assert out["transcribed"] is True and out["engine"] == "faster-whisper"
    assert out["engine_id"] == f"faster-whisper/large-v3@{handler.IMAGE_TAG}"
    assert env.whisper.calls == [{"language": "es", "beam_size": 5, "word_timestamps": True, "vad_filter": True}]
    assert env.fake.load_model_calls == [] and env.fake.align_calls == [] and env.loaded == []
    assert [s["text"] for s in out["segments"]] == ["And so, my fellow", "Americans."]  # blank segment dropped
    assert out["words"][1] == {"w": "so,", "start": 0.3, "end": 0.6, "score": 0.88}
    assert out["word_count"] == 5 == out["scored_count"]
    assert out["settings"]["path"] == "sequential" and out["settings"]["vad_filter"] is True
    assert out["settings"]["vad"]["min_silence_duration_ms"] == 2000


def test_spanish_whisperx_aligns_with_spanish_model(env):
    out = handler.handler(_job("whisperx", language="es"))
    assert out["language"] == "es" and out["settings"]["align_model"] == "VOXPOPULI_ASR_BASE_10K_ES"
    assert env.fake.align_calls == [{"model": "align-es", "language": "es"}]
    assert env.fake.load_model_calls[0]["language"] == "es"


@pytest.mark.parametrize("extra,reason", [
    ({"audio_url": "ftp://x"}, "input"),
    ({"audio_url": None}, "input"),
    ({"language": None}, "input"),
    ({"language": "fr"}, "language"),
    ({"engine": "parakeet"}, "engine"),
    ({"engine": None}, "engine"),
    ({"options": "wide"}, "input"),
    ({"options": {"chunk_size": 60}}, "options"),
    ({"options": {"chunk_size": 12.5}}, "options"),
    ({"options": {"batch_size": True}}, "options"),
    ({"options": {"vad_onset": float("nan")}}, "options"),
    ({"options": {"beam_size": 1}}, "options"),
    ({"engine": "faster-whisper", "options": {"chunk_size": 10}}, "options"),
])
def test_bad_transcribe_input(env, extra, reason):
    out = handler.handler(_job(**{"engine": "whisperx", **extra}))
    assert out["transcribed"] is False and out["reason"] == reason
    assert out["words"] == [] and out["segments"] == []
    assert env.made == [] and env.downloads == []


def test_download_failure_and_temp_dir_removed(env, monkeypatch):
    def boom(url, dest):
        raise handler.AlignError("download")

    monkeypatch.setattr(handler, "_download", boom)
    out = handler.handler(_job("faster-whisper"))
    assert out["transcribed"] is False and out["reason"] == "download" and out["engine"] == "faster-whisper"
    assert len(env.made) == 1 and not os.path.exists(env.made[0])


def test_engine_crash_is_a_failure_not_a_raise(env):
    env.fake.mode = "raise"
    out = handler.handler(_job("whisperx"))
    assert out["transcribed"] is False and out["reason"] == "error"
    assert len(env.made) == 1 and not os.path.exists(env.made[0])


def test_logs_carry_no_text_or_url(env, caplog):
    handler.log.propagate = True
    try:
        with caplog.at_level("INFO", logger="aligner"):
            handler.handler(_job("whisperx"))
            handler.handler(_job("faster-whisper"))
    finally:
        handler.log.propagate = False
    text = caplog.text
    assert "Americans" not in text and "fellow" not in text and "example.test" not in text
    assert "transcribed engine=whisperx language=en" in text
    assert "transcribed engine=faster-whisper language=en" in text


def test_one_whisperx_subclass_model_serves_both_engines(monkeypatch):
    """The batched pipeline needs generate_segment_batched, which only
    whisperx.asr.WhisperModel has; a plain faster_whisper.WhisperModel broke the
    first real run with AttributeError."""
    made = []

    class SubclassModel:
        def __init__(self, path, device, compute_type):
            made.append((path, device, compute_type))

    asr = types.ModuleType("whisperx.asr")
    asr.WhisperModel = SubclassModel
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "whisperx", types.ModuleType("whisperx"))
    monkeypatch.setitem(sys.modules, "whisperx.asr", asr)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # importing it would fail the test
    monkeypatch.setattr(handler, "_WHISPER", None)
    first = handler._get_whisper()
    assert isinstance(first[0], SubclassModel) and first[1:] == ("cpu", "int8")
    assert handler._get_whisper() is first and made == [(handler.WHISPER_DIR, "cpu", "int8")]
