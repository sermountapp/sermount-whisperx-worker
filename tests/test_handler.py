"""Handler shape tests. WhisperX, torch and the network are faked; the real
aligner is exercised by the RunPod console test (see README)."""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import handler  # noqa: E402

SR = 16000


def _words(n, step=0.5):
    return [{"i": 10 + k, "w": f"word{k}", "s": k * step, "e": k * step + 0.3} for k in range(n)]


class FakeAudio:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class FakeWhisperX(types.ModuleType):
    """Echoes every token back, shifted 40 ms, with a score."""

    def __init__(self, mode="ok"):
        super().__init__("whisperx")
        self.mode = mode
        self.calls = []

    def load_audio(self, path):
        assert os.path.exists(path)
        return FakeAudio(SR * 600)

    def align(self, segments, model, metadata, audio, device, return_char_alignments=False):
        seg = segments[0]
        self.calls.append(seg)
        tokens = seg["text"].split(" ")
        if self.mode == "raise":
            raise RuntimeError("cuda exploded")
        if self.mode == "drop":
            tokens = tokens[:-1]
        out = []
        for k, t in enumerate(tokens):
            w = {"word": t, "start": seg["start"] + 0.04 + k * 0.01, "end": seg["start"] + 0.2 + k * 0.01,
                 "score": 0.9}
            if self.mode == "unscored" and k == 0:
                del w["score"]
            out.append(w)
        return {"segments": [], "word_segments": out}


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Fake whisperx, a fake model, a fake download; record the temp dirs."""
    fake = FakeWhisperX()
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    monkeypatch.setattr(handler, "_get_model", lambda: ("model", {"language": "en"}, "cpu"))

    def fake_download(url, dest):
        with open(dest, "wb") as fh:
            fh.write(b"ID3fake")
        return 7

    monkeypatch.setattr(handler, "_download", fake_download)
    made = []
    real_mkdtemp = handler.tempfile.mkdtemp

    def mkdtemp(**kw):
        d = real_mkdtemp(dir=tmp_path, **kw)
        made.append(d)
        return d

    monkeypatch.setattr(handler.tempfile, "mkdtemp", mkdtemp)
    return types.SimpleNamespace(fake=fake, made=made)


def _job(words, **extra):
    return {"id": "t", "input": {"audio_url": "https://example.test/a.mp3", "words": words, "language": "en", **extra}}


def test_same_i_sequence_out_and_temp_dir_removed(env):
    words = _words(200)
    out = handler.handler(_job(words))
    assert out["aligned"] is True and out["reason"] is None
    assert [w["i"] for w in out["words"]] == [w["i"] for w in words]
    assert all(set(w) == {"i", "s", "e", "score"} for w in out["words"])
    assert out["image_tag"] == handler.IMAGE_TAG and out["model"] == handler.ALIGN_MODEL
    assert isinstance(out["exec_ms"], int)
    assert len(env.made) == 1 and not os.path.exists(env.made[0])


def test_output_carries_no_text(env):
    words = _words(20)
    out = handler.handler(_job(words))
    assert not any(w["w"] in repr(out) for w in words)
    assert "example.test" not in repr(out)


def test_temp_dir_removed_on_exception(env):
    env.fake.mode = "raise"
    out = handler.handler(_job(_words(20)))
    assert out == {**out, "aligned": False, "reason": "error", "words": []}
    assert len(env.made) == 1 and not os.path.exists(env.made[0])


def test_mismatched_segment_keeps_incoming_times(env):
    env.fake.mode = "drop"
    words = _words(10)
    out = handler.handler(_job(words))
    assert out["aligned"] is True and out["fallback_segments"] == 1
    assert out["words"] == [{"i": w["i"], "s": w["s"], "e": w["e"], "score": None} for w in words]


def test_unscored_word_keeps_incoming_times(env):
    env.fake.mode = "unscored"
    words = _words(3)
    out = handler.handler(_job(words))
    assert out["words"][0] == {"i": words[0]["i"], "s": words[0]["s"], "e": words[0]["e"], "score": None}
    assert out["words"][1]["score"] == 0.9


@pytest.mark.parametrize("job_input", [
    None,
    {},
    {"audio_url": "ftp://x", "words": _words(2)},
    {"audio_url": "https://x", "words": []},
    {"audio_url": "https://x", "words": [{"i": 1, "w": "a", "s": 0, "e": 1}, {"i": 1, "w": "b", "s": 1, "e": 2}]},
    {"audio_url": "https://x", "words": [{"i": 1, "w": "a", "s": float("nan"), "e": 1}]},
    {"audio_url": "https://x", "words": [{"i": True, "w": "a", "s": 0, "e": 1}]},
    {"audio_url": "https://x", "words": [{"i": 1, "w": None, "s": 0, "e": 1}]},
])
def test_bad_input_is_not_aligned(env, job_input):
    out = handler.handler({"id": "t", "input": job_input})
    assert out["aligned"] is False and out["reason"] == "input" and out["words"] == []
    assert env.made == []


def test_unsupported_language(env):
    out = handler.handler(_job(_words(3), language="es"))
    assert out["aligned"] is False and out["reason"] == "language"


def test_download_failure(env, monkeypatch):
    def boom(url, dest):
        raise handler.AlignError("download")

    monkeypatch.setattr(handler, "_download", boom)
    out = handler.handler(_job(_words(3)))
    assert out["reason"] == "download"
    assert len(env.made) == 1 and not os.path.exists(env.made[0])


def test_tokens_have_no_spaces_and_are_never_empty():
    assert handler.token_for("Lord,") == "Lord,"
    assert handler.token_for(" two words ") == "twowords"
    assert handler.token_for("   ") == "-"
    assert handler.token_for("") == "-"


def test_segments_cover_every_word_once_and_stay_under_cap():
    words = _words(400, step=0.4)  # 160 s of speech
    segs = handler.build_segments(words, duration=170.0)
    covered = [k for s in segs for k in range(s["first"], s["last"] + 1)]
    assert covered == list(range(len(words)))
    for a, b in zip(segs, segs[1:]):
        assert a["end"] <= b["start"]
    assert all(0 < s["end"] - s["start"] <= handler.MAX_SEGMENT_SECS for s in segs)


def test_segments_split_at_the_widest_pause():
    words = _words(80, step=0.5)
    for w in words[40:]:  # a 3 s pause before word 40 (at ~20 s)
        w["s"] += 3.0
        w["e"] += 3.0
    segs = handler.build_segments(words, duration=60.0)
    assert segs[0]["last"] == 39


def test_segment_windows_are_clamped_to_audio():
    words = [{"i": 0, "w": "a", "s": 0.1, "e": 0.4}, {"i": 1, "w": "b", "s": 0.5, "e": 0.9}]
    segs = handler.build_segments(words, duration=1.0)
    assert segs == [{"start": 0.0, "end": 1.0, "first": 0, "last": 1}]


def test_logs_carry_no_words_or_url(env, caplog):
    handler.log.propagate = True
    try:
        with caplog.at_level("INFO", logger="aligner"):
            handler.handler(_job([{"i": 0, "w": "Hallelujah", "s": 0.0, "e": 0.5}]))
    finally:
        handler.log.propagate = False
    text = caplog.text
    assert "Hallelujah" not in text and "example.test" not in text
    assert "aligned words=1" in text
