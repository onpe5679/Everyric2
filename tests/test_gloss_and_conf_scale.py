"""번역·독음 병기 시트 제외 + quality_score의 어댑터 스케일 오염 회귀 테스트.

두 항목 모두 코퍼스 73곡 전수 조사와 어댑터 스왑 실측이 하류로 넘긴 것이다.

항목 1 — 병기 시트: 사용자가 가사 사이트에서 「원문 / 한글 독음 / 한국어 번역」이 병기된
시트를 통째로 붙여넣으면 입력의 2/3가 비가창이 되고, 노래하지 않는 줄에 타이밍을 맞추려
들면 가창 줄 타이밍까지 망가진다. 실측 2곡:
    FxOfDVyITak — (가나, 한글, 한글) 74/74 완벽 반복, quality 0.0135
    ba7YbGO2aq4 — 한글 8줄이 8/8 직전 일본어 줄의 번역, quality 0.0072

항목 2 — conf 스케일: 같은 영어 곡(dQw4w9WgXcQ)에서 어댑터만 바꿨을 때 eng 0.1289 →
kor 0.0492 (2.6배 하락)인데 잔차는 동일하고 매칭률은 0.9979 → 1.0000으로 올랐다. vocab이
커지면 프레임당 posterior가 흩어지는 것뿐이다(eng 154 vs kor 1330).
"""
import json
import math
from pathlib import Path

import pytest

from everyric2.config.settings import AlignmentSettings
from everyric2.server.worker import (
    _ADAPTER_VOCAB_SIZE,
    _conf_alpha,
    _dual_align_prefers_original,
    _fold_gloss_into_segments,
    _line_script_class,
    _rescale_conf,
    _scale_free_quality,
    _split_gloss_lines,
    detect_gloss_lines,
)

# ── 실측 수치 (변경 시 근거 문서도 함께 갱신) ─────────────────────────
CONF_ENG = 0.1289  # dQw4w9WgXcQ, eng 어댑터 (vocab 154)
CONF_KOR = 0.0492  # 같은 곡, kor 어댑터 (vocab 1330), 잔차 동일


class _Line:
    """LyricLine 스텁 — 감지·분할은 .text만 본다."""

    def __init__(self, text: str, line_number: int) -> None:
        self.text = text
        self.line_number = line_number


def _lines(texts: list[str]) -> list[_Line]:
    return [_Line(t, i + 1) for i, t in enumerate(texts)]


# ─────────────────────── 항목 1: 스크립트 분류 ───────────────────────


def test_script_class_classifies_measured_lines():
    # 원문 판정은 tri-line.ts의 isJa와 같다 (가나/한자 + 한글 비율 < 0.15)
    assert _line_script_class("ゆらゆら numb numb") == "ja"
    assert _line_script_class("熱異常") == "ja"
    assert _line_script_class("揺らめく光の中で") == "ja"
    # 실측 번역 줄 — 한글 비율 0.38이라 tri-line.ts의 0.5 기준으로는 놓친다
    assert 0.3 < 5 / len("아스라이해numbnumb") < 0.5
    assert _line_script_class("아스라이해 numb numb") == "ko"
    assert _line_script_class("네츠이죠") == "ko"
    assert _line_script_class("never gonna give you up") == "other"
    assert _line_script_class("") == "other"
    assert _line_script_class("   ") == "other"
    # 가나가 섞인 줄은 ko가 아니다 (독음 줄에 가나가 남아 있으면 원문일 수 있다)
    assert _line_script_class("아스라이해 ゆら") != "ko"
    # 한자 줄에 한글 한 글자가 섞여도 한글 줄로는 보지 않는다 (한자 > 한글)
    assert _line_script_class("熱異常 열") != "ko"
    # 영어 줄에 한글 한 단어만 섞인 정도는 한글 줄로 보지 않는다
    assert _line_script_class("take me higher tonight forever 밤") == "other"


# ────────────── 항목 1: 실측 패턴 ① FxOfDVyITak (3줄 주기) ──────────────


def _fx_sheet(cycles: int = 74) -> list[str]:
    """FxOfDVyITak 형태: (일본어 원문, 한글 독음, 한국어 번역) 3줄 블록의 완벽한 반복."""
    out: list[str] = []
    for i in range(cycles):
        out += [f"揺らめく光の中で{i}", f"유라메쿠 히카리노 나카데{i}", f"흔들리는 빛 속에서{i}"]
    return out


def test_fx_pattern_excludes_two_thirds():
    texts = _fx_sheet()
    gloss = detect_gloss_lines(texts)
    # 74주기 * 2줄 = 148줄이 비가창
    assert len(gloss) == 148
    assert len(texts) - len(gloss) == 74
    # 역할·원문 인덱스가 (원문, 독음, 번역) 순서대로 매겨진다
    assert gloss[1] == ("pronunciation", 0)
    assert gloss[2] == ("translation", 0)
    assert len(texts) == 222
    assert gloss[221] == ("translation", 219)  # 마지막 주기
    # 원문 줄은 하나도 빠지지 않는다
    assert all(i % 3 != 0 for i in gloss)


def test_fx_pattern_survives_leading_stray_line():
    # 맨 앞에 제목/표기 한 줄이 붙어 주기가 밀려도 offset 탐색이 잡는다
    texts = ["【初音ミク】タイトル", *_fx_sheet(20)]
    gloss = detect_gloss_lines(texts)
    assert len(gloss) == 40
    assert 0 not in gloss  # 밀림 앞의 줄은 손대지 않는다
    assert gloss[2] == ("pronunciation", 1)


def test_fx_pattern_tolerates_one_broken_cycle():
    # 위키가 놓친 표기 등으로 한 주기가 깨져도(일치율 >= 0.9) 나머지는 잡고, 깨진 주기는 보존
    texts = _fx_sheet(20)
    texts[31] = "らららら"  # 주기 10의 독음 줄 자리를 원문 계열로 오염
    gloss = detect_gloss_lines(texts)
    assert len(gloss) == 38  # 19주기 * 2
    assert 30 not in gloss and 31 not in gloss and 32 not in gloss


# ─────── 항목 1: 실측 패턴 ② ba7YbGO2aq4 — 잡아서는 안 되는 형태 ───────


def _ba_sheet() -> list[str]:
    """ba7YbGO2aq4 형태: 일본어 줄이 다수고 한글이 **한 줄씩 흩어져** 섞인다.

    한때 이것을 "한글 8줄이 전부 직전 줄의 번역"으로 판정해 정렬 입력에서 뺐다. **그 판정이
    틀렸다** — 사용자가 그 곡을 직접 듣고 확인했다: 「아스라이해」「희미한」「미묘한」「좋아해」는
    실제로 노래에서 한국어로 불린다(그 곡은 일본어·영어·한국어가 실제 발성에 섞인다).
    """
    texts: list[str] = []
    pairs = 0
    for i in range(40):
        texts.append(f"ゆらゆら numb numb {i}")
        # 한국어 가창 줄이 한 줄씩 섞인다 (연달아 나오지 않는다)
        if i % 5 == 0 and pairs < 8:
            texts.append(f"아스라이해 numb numb {i}")
            pairs += 1
    return texts


def test_interleaved_sung_korean_lines_are_never_excluded():
    """한 줄씩 섞인 한국어 가창 줄은 **하나도** 빠지지 않는다.

    이 형태를 잡던 규칙(인접 종속)을 지운 이유는 코퍼스 68곡 전수 조사에서 그 규칙이 잡은
    곡이 ba7YbGO2aq4 하나였고 그 판정이 오판이었기 때문이다. 빠지면 두 가지를 동시에 잃는다:
    그 8줄의 가사와, 앵커로 지목된 앞줄의 실제 번역(덮어써진다).
    """
    texts = _ba_sheet()
    assert sum(1 for t in texts if _line_script_class(t) == "ko") == 8  # 표본 자체를 확인
    assert detect_gloss_lines(texts) == {}


def test_interleaved_korean_still_ignored_when_lines_run_together():
    # 한글 줄이 연달아 나오는 형태도 당연히 미발동 — 지운 규칙이 그것으로 자신을 정당화했다
    texts = _ba_sheet()
    idx = min(i for i, t in enumerate(texts) if _line_script_class(t) == "ko")
    texts.insert(idx + 1, "연달아 나오는 한국어 가창 줄")
    assert detect_gloss_lines(texts) == {}


# ─────────────── 항목 1: 정상 곡 오탐 없음 (핵심 보수성) ───────────────


def test_no_false_positive_on_plain_japanese_song():
    texts = [f"揺らめく光の中で{i}" for i in range(40)]
    assert detect_gloss_lines(texts) == {}


def test_no_false_positive_on_plain_korean_song():
    texts = [f"흔들리는 빛 속에서{i}" for i in range(40)]
    assert detect_gloss_lines(texts) == {}


def test_no_false_positive_on_korean_song_with_english_hooks():
    # (영어 훅, 한국어, 한국어)가 규칙적으로 반복되는 **실재하는** 한국어 곡 구조.
    # 원문 줄을 라틴까지 허용하면 3줄 주기에 걸려 곡의 2/3를 잃는다 → 원문은 ja만 인정한다.
    texts: list[str] = []
    for i in range(15):
        texts += [f"take me higher {i}", f"흔들리는 빛 속에서{i}", f"나를 부르는 소리{i}"]
    assert detect_gloss_lines(texts) == {}


def test_no_false_positive_on_english_song_with_interleaved_korean_lines():
    # 영어 줄 뒤에 한국어 줄이 1:1로 오는 형태(K-POP 병창)도 원문 계열 조건에서 탈락한다
    texts: list[str] = []
    for i in range(20):
        texts.append(f"take me higher {i}")
        if i % 3 == 0:
            texts.append(f"흔들리는 빛 속에서{i}")
    assert detect_gloss_lines(texts) == {}


def test_no_false_positive_on_alternating_bilingual_song():
    # (원문, 한글) 완전 교대는 한·일 병창과 구별 불가 → 의도적으로 통과시킨다
    texts: list[str] = []
    for i in range(20):
        texts += [f"揺らめく光の中で{i}", f"흔들리는 빛 속에서{i}"]
    assert detect_gloss_lines(texts) == {}


def test_no_false_positive_on_japanese_song_with_two_korean_lines():
    # 일본어 곡에 한국어 줄이 몇 개 섞인 형태 — 주기가 성립하지 않으므로 미발동
    texts = [f"揺らめく光の中で{i}" for i in range(30)]
    texts.insert(5, "흔들리는 빛 속에서")
    texts.insert(12, "나를 부르는 소리")
    assert detect_gloss_lines(texts) == {}


def test_no_false_positive_when_hangul_lines_follow_japanese_lines():
    """한글 줄이 전부 일어 줄 뒤에 오는 형태도 미발동 — 지운 규칙이 잡던 바로 그 형태다.

    앞줄 뒤에 붙어 있다는 것만으로 "번역"이라 단정할 수 없다. ba7YbGO2aq4가 정확히 이
    모양인데 실제로 불리는 한국어였다.
    """
    texts: list[str] = []
    for i in range(30):
        texts.append(f"揺らめく光の中で{i}")
        if i in (3, 9, 15, 21):
            texts.append(f"흔들리는 빛 속에서{i}")
    assert detect_gloss_lines(texts) == {}
    texts += ["take me higher", "나를 부르는 소리"]
    assert detect_gloss_lines(texts) == {}


def test_short_input_never_detected():
    # 3주기(9줄) 미달은 판정하지 않는다 — 우연 일치 배제
    texts = ["揺らめく光", "유라메쿠 히카리", "흔들리는 빛", "夜の底で", "요노 소코데", "밤의 바닥에서"]
    assert detect_gloss_lines(texts) == {}


def test_partial_periodic_run_inside_a_normal_song_is_ignored():
    # 3줄 주기가 곡 일부에만 있으면(커버리지 < 0.9) 발동하지 않는다
    texts = [f"揺らめく光の中で{i}" for i in range(40)]
    texts[10:10] = ["夜の底で", "요노 소코데", "밤의 바닥에서"] * 3
    assert detect_gloss_lines(texts) == {}


# ─────────── 항목 1: 정렬 입력 분리 + 표시용 유지 + 스위치 off ───────────


def test_split_removes_gloss_from_alignment_input_and_keeps_display_mapping():
    texts = _fx_sheet(20)
    kept, folded = _split_gloss_lines(_lines(texts), enabled=True)
    assert len(kept) == 20
    assert [ln.text for ln in kept] == [texts[i] for i in range(0, 60, 3)]
    # 원본 line_number는 유지된다 (빈 줄이 있는 가사와 동일한 gap 의미론)
    assert [ln.line_number for ln in kept] == list(range(1, 61, 3))
    # 정렬 인덱스(0..19) 기준으로 표시용 메타가 매핑된다
    assert folded[0] == {"pronunciation": texts[1], "translation": texts[2]}
    assert folded[19] == {"pronunciation": texts[58], "translation": texts[59]}


def test_split_is_noop_when_disabled():
    texts = _fx_sheet(20)
    lines = _lines(texts)
    kept, folded = _split_gloss_lines(lines, enabled=False)
    assert kept is lines
    assert folded == {}


def test_split_is_noop_on_normal_song():
    lines = _lines([f"揺らめく光の中で{i}" for i in range(40)])
    kept, folded = _split_gloss_lines(lines, enabled=True)
    assert kept is lines
    assert folded == {}


def test_fold_reattaches_gloss_for_display():
    timestamps = [{"text": "揺らめく光の中で0", "start": 1.0, "end": 2.0}]
    n = _fold_gloss_into_segments(timestamps, {0: {"pronunciation": "유라메쿠", "translation": "흔들리는"}})
    assert n == 2
    assert timestamps[0]["pronunciation"] == "유라메쿠"
    assert timestamps[0]["translation"] == "흔들리는"


def test_fold_never_overwrites_measured_pronunciation():
    # 독음 정렬 경로/위키 line_meta가 채운 값은 실측 타이밍을 동반하므로 우선한다
    timestamps = [
        {
            "text": "揺らめく光の中で0",
            "pronunciation": "위키 독음",
            "pron_segments": [{"text": "위", "start": 1.0, "end": 1.2}],
        }
    ]
    n = _fold_gloss_into_segments(timestamps, {0: {"pronunciation": "붙여넣기 독음", "translation": "번역"}})
    assert n == 1
    assert timestamps[0]["pronunciation"] == "위키 독음"
    assert timestamps[0]["pron_segments"]
    assert timestamps[0]["translation"] == "번역"


def test_fold_ignores_out_of_range_positions():
    # 정렬이 라인을 잃는 이상 상황에서도 예외를 내지 않는다
    timestamps = [{"text": "a"}]
    assert _fold_gloss_into_segments(timestamps, {5: {"translation": "x"}}) == 0


def test_setting_defaults_on_and_switchable():
    assert AlignmentSettings().exclude_gloss_lines is True
    assert AlignmentSettings(exclude_gloss_lines=False).exclude_gloss_lines is False


# ───── 항목 1+2: _run_alignment 배선 (GPU 없음 — 엔진·오디오만 스텁) ─────


class _FakeEngine:
    """CTC 엔진 대역 — 받은 라인 수만큼 결과를 만들고 어댑터 코드를 노출한다."""

    def __init__(self, adapter: str = "jpn", conf: float = 0.05) -> None:
        self._current_adapter = adapter
        self._current_lang = "ja"
        self._last_star_spans: list = []
        self._conf = conf
        self.seen: list[list[str]] = []

    def is_available(self) -> bool:
        return True

    @staticmethod
    def get_engine_type() -> str:
        # get_shared_ctc_engine 대역이라 실제 CTCEngine과 같은 값(2026-08-04, worker.py의
        # 결과 dict "engine" 키가 이 메서드로 채워진다).
        return "ctc"

    def align(self, audio, lyrics, language=None, progress_callback=None):
        from everyric2.inference.prompt import SyncResult, WordSegment

        self.seen.append([ln.text for ln in lyrics])
        out = []
        for k, ln in enumerate(lyrics):
            chars = [c for c in ln.text if not c.isspace()] or ["x"]
            step = 1.0 / len(chars)
            out.append(
                SyncResult(
                    text=ln.text,
                    start_time=float(k),
                    end_time=float(k) + 1.0,
                    confidence=self._conf,
                    line_number=ln.line_number,
                    word_segments=[
                        WordSegment(
                            word=c,
                            start=k + j * step,
                            end=k + (j + 1) * step,
                            confidence=self._conf,
                        )
                        for j, c in enumerate(chars)
                    ],
                )
            )
        return out


def _run_alignment_with_fake_engine(monkeypatch, tmp_path, lyrics: str, **align_overrides):
    """_run_alignment을 실제로 돌린다 — 오디오 로드/보컬 분리/멜로디/템포만 스텁."""
    import numpy as np

    from everyric2.alignment import ctc_engine as ctc_mod
    from everyric2.audio import loader as loader_mod
    from everyric2.config.settings import get_settings
    from everyric2.server import worker as worker_mod

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake")
    fake_audio = loader_mod.AudioData(
        waveform=np.zeros(16000, dtype="float32"), sample_rate=16000, duration=1.0
    )

    class _FakeLoader:
        def load(self, path):
            return fake_audio

    engine = _FakeEngine()
    monkeypatch.setattr(loader_mod, "AudioLoader", _FakeLoader)
    monkeypatch.setattr(ctc_mod, "get_shared_ctc_engine", lambda _s: engine)
    monkeypatch.setattr(worker_mod, "_separate_vocals", lambda _a: None)
    monkeypatch.setattr(worker_mod, "_estimate_tempo", lambda _a: None)

    settings = get_settings()
    saved_melody = settings.melody.enabled
    # gloss 처리/신뢰도 스케일은 레거시 CTC 엔진 전용 경로다(위 get_shared_ctc_engine 대역)
    # — 새 스택(기본값 owsm/omniasr)은 이 경로가 없으므로 레거시로 강제 고정한다.
    saved_engine = settings.alignment.engine
    object.__setattr__(settings.melody, "enabled", False)
    object.__setattr__(settings.alignment, "engine", "ctc")
    saved_align = {k: getattr(settings.alignment, k) for k in align_overrides}
    for k, v in align_overrides.items():
        object.__setattr__(settings.alignment, k, v)
    try:
        result = worker_mod._run_alignment(str(audio_file), lyrics, "ja")
    finally:
        object.__setattr__(settings.melody, "enabled", saved_melody)
        object.__setattr__(settings.alignment, "engine", saved_engine)
        for k, v in saved_align.items():
            object.__setattr__(settings.alignment, k, v)
    return result, engine


def test_run_alignment_feeds_only_sung_lines_and_keeps_gloss_for_display(monkeypatch, tmp_path):
    texts = _fx_sheet(20)
    result, engine = _run_alignment_with_fake_engine(monkeypatch, tmp_path, "\n".join(texts))

    # 엔진은 원문 줄만 봤다 (입력 60줄 → 정렬 20줄)
    assert engine.seen and len(engine.seen[0]) == 20
    assert engine.seen[0] == [texts[i] for i in range(0, 60, 3)]
    # 세그먼트도 원문 줄만, 그리고 걸러낸 줄은 표시용으로 붙어 있다
    ts = result["timestamps"]
    assert len(ts) == 20
    assert ts[0]["text"] == texts[0]
    assert ts[0]["pronunciation"] == texts[1]
    assert ts[0]["translation"] == texts[2]
    assert ts[19]["translation"] == texts[59]
    # 붙여넣은 60줄의 텍스트가 하나도 유실되지 않았다
    surfaced = {s["text"] for s in ts} | {s["pronunciation"] for s in ts} | {
        s["translation"] for s in ts
    }
    assert surfaced == set(texts)


def test_run_alignment_switch_off_feeds_every_pasted_line(monkeypatch, tmp_path):
    texts = _fx_sheet(20)
    result, engine = _run_alignment_with_fake_engine(
        monkeypatch, tmp_path, "\n".join(texts), exclude_gloss_lines=False
    )
    assert len(engine.seen[0]) == 60
    assert len(result["timestamps"]) == 60
    assert "translation" not in result["timestamps"][0]


def test_run_alignment_normal_song_untouched(monkeypatch, tmp_path):
    texts = [f"揺らめく光の中で{i}" for i in range(30)]
    result, engine = _run_alignment_with_fake_engine(monkeypatch, tmp_path, "\n".join(texts))
    assert len(engine.seen[0]) == 30
    assert len(result["timestamps"]) == 30


def test_run_alignment_reports_adapter_and_scale_free_quality(monkeypatch, tmp_path):
    texts = [f"揺らめく光の中で{i}" for i in range(30)]
    result, _ = _run_alignment_with_fake_engine(monkeypatch, tmp_path, "\n".join(texts))
    # quality_score는 원본 conf 그대로 (확장의 0.001 고정 임계 호환)
    assert result["quality_score"] == pytest.approx(0.05, rel=1e-6)
    debug = result["debug"]
    assert debug["quality_adapter"] == "jpn"
    assert debug["quality_norm"] == pytest.approx(
        _scale_free_quality(0.05, "jpn"), abs=1e-6
    )
    # 같은 품질을 vocab이 작은 어댑터로 재보고하면 raw는 달라지지만 norm은 유지된다
    assert _scale_free_quality(0.05, "jpn") != pytest.approx(_scale_free_quality(0.05, "kor"))


# ────────────── 항목 2: 어댑터 vocab 크기와 스케일 모델 ──────────────


def test_adapter_vocab_sizes_match_measured_census():
    census = json.loads(
        (Path(__file__).parent / "fixtures" / "mms_adapter_script_census.json").read_text(
            encoding="utf-8"
        )
    )
    for adapter, size in _ADAPTER_VOCAB_SIZE.items():
        assert census[adapter]["vocab_size"] == size, adapter


def test_log_scale_model_explains_the_measured_eng_kor_gap():
    # α = log_V(1/conf)는 어댑터에 무관해야 한다 — 실측 두 값의 α가 3% 안에서 일치
    a_eng = _conf_alpha(CONF_ENG, "eng")
    a_kor = _conf_alpha(CONF_KOR, "kor")
    assert abs(a_eng - a_kor) / a_kor < 0.03
    # 선형 모델(conf ∝ 1/V)은 3배 이상 틀린다 — 그래서 로그 스케일을 쓴다
    linear_pred = CONF_ENG * _ADAPTER_VOCAB_SIZE["eng"] / _ADAPTER_VOCAB_SIZE["kor"]
    assert linear_pred < CONF_KOR / 3


def test_scale_free_quality_is_comparable_across_adapters():
    q_eng = _scale_free_quality(CONF_ENG, "eng")
    q_kor = _scale_free_quality(CONF_KOR, "kor")
    # 2.6배 벌어져 있던 raw conf가 스케일 무관 지표에서는 3% 안으로 붙는다
    assert CONF_ENG / CONF_KOR > 2.5
    assert abs(q_eng - q_kor) / q_kor < 0.03
    # 0~1 범위, 클수록 좋다
    assert 0.0 < q_kor < 1.0
    assert _scale_free_quality(0.5, "kor") > _scale_free_quality(0.0005, "kor")


def test_scale_free_quality_none_on_unknown_or_missing():
    assert _scale_free_quality(None, "kor") is None
    assert _scale_free_quality(0.05, None) is None
    assert _scale_free_quality(0.05, "who-knows") is None
    assert _scale_free_quality(0.0, "kor") is None


def test_rescale_is_identity_for_same_adapter():
    # 영어 곡은 이제 ko/ja 양쪽이 kor 어댑터 → 보정이 개입하지 않는다 (회귀 0)
    assert _rescale_conf(0.0005, "kor", "kor") == 0.0005
    assert _rescale_conf(None, "jpn", "kor") is None
    assert _rescale_conf(0.0005, None, "kor") == 0.0005
    assert _rescale_conf(0.0005, "jpn", None) == 0.0005
    assert _rescale_conf(0.0005, "jpn", "nope") == 0.0005


def test_rescale_preserves_sharpness_across_adapters():
    # 같은 α를 가진 jpn conf를 kor 스케일로 옮기면 kor에서 같은 α로 측정된 값이 나온다
    alpha = 0.42
    jpn_conf = math.exp(-alpha * math.log(_ADAPTER_VOCAB_SIZE["jpn"]))
    scaled = _rescale_conf(jpn_conf, "jpn", "kor")
    assert abs(_conf_alpha(scaled, "kor") - alpha) < 1e-9
    # vocab이 더 큰 어댑터의 raw conf는 낮게 측정되므로 보정하면 올라간다
    assert scaled > jpn_conf


# ────────── 항목 2: 어댑터가 달라도 이중정렬 판정이 뒤집히지 않는다 ──────────


def _prefers_original(ko_conf, ko_adapter, ja_conf_raw, ja_adapter, min_ratio=1.5):
    """worker._run_alignment의 이중정렬 판정 경로 (스케일 보정 후 비율 비교)."""
    return _dual_align_prefers_original(
        ko_conf, _rescale_conf(ja_conf_raw, ja_adapter, ko_adapter), min_ratio
    )


def test_equally_collapsed_ja_never_wins_regardless_of_adapter():
    # 熱異常: ko(kor) 0.0005 = α 1.0567. ja가 **같은 정도로** 붕괴했을 때 어댑터별 raw conf는
    # 전부 다르지만(vocab이 크면 더 낮게 측정) 어느 경우에도 원문을 채택하면 안 된다.
    ko_conf = 0.0005
    alpha = _conf_alpha(ko_conf, "kor")
    for adapter in ("kor", "jpn", "cmn-script_simplified", "eng"):
        ja_raw = math.exp(-alpha * math.log(_ADAPTER_VOCAB_SIZE[adapter]))
        assert _prefers_original(ko_conf, "kor", ja_raw, adapter) is False, adapter


def test_genuinely_better_ja_wins_regardless_of_adapter():
    # ja가 실제로 훨씬 첨예하면(α가 작으면) 어느 어댑터로 측정됐든 채택된다
    ko_conf = 0.0005
    better_alpha = _conf_alpha(0.005, "kor")  # ko의 10배 수준
    for adapter in ("kor", "jpn", "cmn-script_simplified", "eng"):
        ja_raw = math.exp(-better_alpha * math.log(_ADAPTER_VOCAB_SIZE[adapter]))
        assert _prefers_original(ko_conf, "kor", ja_raw, adapter) is True, adapter


def test_eng_adapter_inflation_no_longer_flips_the_decision():
    # 스왑 전 회귀: 영어 곡의 원문 정렬이 eng(vocab 154)로 측정되면 raw conf가 2.6배 부풀어
    # 같은 품질인데도 1.5배 마진을 넘겨 원문이 채택됐다. 보정하면 넘지 않는다.
    ko_conf = CONF_KOR
    assert _dual_align_prefers_original(ko_conf, CONF_ENG, 1.5) is True  # 보정 전 = 오채택
    assert _prefers_original(ko_conf, "kor", CONF_ENG, "eng") is False  # 보정 후 = 유지


def test_client_low_conf_threshold_equivalents_are_pinned():
    """확장의 lowConfWarning 고정 임계 0.001(raw conf)이 어댑터별로 무엇을 뜻하는지 고정한다.

    서버는 quality_score를 원본 conf 그대로 내려보내므로 확장은 **지금 그대로 동작한다**.
    다만 그 한 숫자가 어댑터마다 다른 품질을 가리킨다는 것이 문제다 — 확장이 스케일 무관
    지표(extra.debug.quality_norm)로 옮길 때 쓸 등가 임계를 여기 못 박는다.
    """
    equiv = {a: _scale_free_quality(0.001, a) for a in _ADAPTER_VOCAB_SIZE}
    assert equiv["kor"] == pytest.approx(0.3828, abs=5e-4)
    assert equiv["jpn"] == pytest.approx(0.4089, abs=5e-4)
    assert equiv["eng"] == pytest.approx(0.2538, abs=5e-4)
    assert equiv["cmn-script_simplified"] == pytest.approx(0.4398, abs=5e-4)
    # 같은 0.001이 어댑터에 따라 1.7배 다른 품질을 가리킨다 = 고정 raw 임계의 오염
    assert max(equiv.values()) / min(equiv.values()) > 1.7
    # 실측 곡들이 kor 등가 임계(0.3828)의 어느 쪽에 놓이는지 — 현재 동작과 일치해야 한다
    assert _scale_free_quality(0.0005, "kor") < equiv["kor"]  # 熱異常 원곡: 경고 (raw도 경고)
    assert _scale_free_quality(0.0076, "kor") > equiv["kor"]  # 그 커버: 무경고 (raw도 무경고)
    assert _scale_free_quality(0.00106, "kor") > equiv["kor"]  # 消失: 무경고 (raw도 무경고)


def test_raw_threshold_semantics_unchanged_for_the_ko_floor_gate():
    # dual_align_conf는 ko(항상 kor 어댑터) conf에만 걸리므로 스케일 오염이 없다.
    # 熱異常 원곡/커버 경계가 그대로 유지되는지 고정한다.
    s = AlignmentSettings()
    assert s.dual_align_conf == 0.002
    assert _conf_alpha(0.0005, "kor") > _conf_alpha(0.0076, "kor")  # 원곡이 더 나쁘다
