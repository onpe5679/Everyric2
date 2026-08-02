"""표시 보정층 — 분리 스템 우세도 기반 라인/세그 이동 장치 + 추임새 탐지.

## 왜 별도 모듈인가

기존 프로드 보정은 두 갈래다: ``timing_postprocess.TimingPostProcessor``(1차 VAD 신축)와
``server/worker.py``의 ``_clamp_stretched_lines``/``_snap_silence_undershoot``/
``_snap_post_interlude_leak``(2차, 역시 VAD 기반). 이 모듈이 옮기는 장치들은 전부
``VocalActivityDetector``가 아니라 **보컬 우세도**(``star_prior.vocal_presence_from_stems``)
위에서만 판정한다 — VAD가 **분리된 vocals 스템** 위에서 죽기 때문이다(간주 블리드가
전 구간을 발성으로 채운다, ``star_prior.py`` 모듈 주석 실측 참고). 그래서 worker.py의
기존 함수들과 나란히 두지 않고 새 파일로 분리했다: 판정 신호(우세도 vs VAD)가 다르므로
섞으면 "이 장치가 왜 여기 있나"가 흐려진다.

포팅 원본은 ``scripts/bench_adapters/postprocess.py``(장치 4종 + 좌초 탐지기)와
``scripts/karaoke_review.py``의 ``adlib_candidates``(추임새 탐지)다. **문턱과 발동 조건은
벤치 실측으로 정해졌다 — 바꾸지 않았다.** 각 함수 docstring에 원본 실측 근거를 옮겨
적었으니 상세 사고 재현·기각된 대안은 그쪽을 봐라.

## 세그는 건드리지 않는다 (예외: 이 모듈이 세그를 옮기는 자리는 명시된 셋뿐)

라인이 움직여도 그 안의 CTC 실측 세그(``word_segments``)는 원칙적으로 그대로 둔다.
``_clamp_pathological``·``_stranded_sites``는 라인 경계만 만진다. 예외는
``_fold_stranded_repeats``(좌초 렌디션을 통째로 옮기므로 세그도 함께 재배치해야 의미가
있다)·``_pull_disconnected_tails``(끊긴 꼬리 세그 자체가 대상)·``_snap_silent_heads``
(무음 위에 걸친 세그의 시작을 미는 것이 장치의 본질)뿐이다 — 셋 다 "이 세그의 시작은
CTC 실측이 아니라 잔해/추정이다"를 우세도·절대 크기 두 축으로 확인한 뒤에만 움직인다.

이동으로 라인 끝이 줄어들어 세그가 새 끝을 넘어서는 일반적인 "넘친 세그 끝 잘라내기"는
**이 모듈의 책임이 아니다** — bench의 ``PostProcessedAligner._apply``는 그 클리핑을
장치별이 아니라 "라인이 움직였으면" 한 번에 하는 공용 단계로 둔다(원본 607~626행).
이 모듈을 worker.py에 배선할 때 그 공용 단계를 같이 옮겨야 한다.

## 실행 순서

``apply_stranded_corrections``가 강제하는 순서(접기 → 꼬리 되당김 → 머리 스냅 → 좌초
탐지)는 원본 ``_apply`` 말미(628~642행)와 같다. 접기를 가장 먼저 돌리는 이유: 좌초 라인이
제자리로 접히면 그 안의 끊긴 꼬리도 함께 옮겨지므로, 순서를 바꾸면 아직 안 접힌 라인의
꼬리를 꼬리-되당김이 먼저 뭉개 버린다. ``_stranded_sites``는 이동 장치가 아니라 사후
탐지기이므로 항상 맨 끝에 둬서 앞 세 장치가 옮긴 뒤에도 남는 좌초만 센다.
``_clamp_pathological``은 이 순서 밖이다 — 원본에서도 ``corrected`` 자체를 만드는
더 이른 단계(584~590행)라 별도로 호출한다.

## 우세도가 없으면 전부 조용히 건너뛴다

``dominance_activity_from_waveforms``가 ``None``을 돌려주면(분리 스템 부재·신호 열화)
이 모듈의 모든 장치는 호출하지 않는 것이 계약이다 — VAD로 물러서지 않는다. 물러서면
바로 위에서 설명한 "VAD가 분리 스템 위에서 죽는다"는 원래 문제가 재발한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from everyric2.alignment.star_prior import vocal_presence_from_stems
from everyric2.audio.vad import VocalRegion
from everyric2.inference.prompt import SyncResult

logger = logging.getLogger(__name__)

# ── 우세도로 만든 발성 구간 ──
# 0.35는 간주 우세도(~0.199)와 가창 우세도(0.36~0.68) 사이, 0.10초는 검출 구간의 최소
# 길이 — postprocess.py의 실측치 그대로(바꾸지 않았다).
_DOMINANCE_LEVEL = 0.35
_DOMINANCE_MIN_SEC = 0.10
# 무음 판정 절대 크기 하한(dBFS). 우세도는 비율이라 디지털 무음(0/0)에서 잡음으로 튄다 —
# 이 하한이 있어야 소리가 실제로 나는지를 우세도와 별도로 확인할 수 있다.
_SILENT_DB = -30.0


@dataclass
class DominanceActivity:
    """``postprocess.py``의 ``_RegionSet``에 대응 — 이동 장치들이 공유하는 우세도 신호 묶음.

    - ``regions``: 우세도 ≥0.35가 ``_DOMINANCE_MIN_SEC`` 이상 이어지는 구간(``VocalRegion``
      재사용 — ``.start``/``.end``만 쓰는 장치들이 그대로 돌게 하려고 별도 타입을 안 만들었다).
    - ``values``/``hop``: 우세도 원곡선(10ms 격자 기본) — ``_snap_silent_heads``·
      ``adlib_candidates``가 region보다 고운 판정에 쓴다.
    - ``db``: 보컬 절대 크기(dBFS) 곡선, 같은 hop. 우세도만으로는 디지털 무음(0/0 잡음)과
      진짜 저우세도 구간을 못 가른다 — 이 축이 그 구분을 담당한다.
    """

    regions: list[VocalRegion]
    values: np.ndarray
    hop: float
    db: np.ndarray


def dominance_activity_from_waveforms(
    vocals_waveform: np.ndarray,
    accomp_waveform: np.ndarray,
    sample_rate: int,
    *,
    smooth_sec: float = 0.2,
    hop_sec: float = 0.01,
) -> DominanceActivity | None:
    """보컬/반주 파형 → ``DominanceActivity``. 못 만들면(스템 부재·열화) ``None``.

    우세도 계산 자체(``vocal_presence_from_stems``)는 재구현하지 않고 그대로 가져다
    쓴다 — 이미 서버에 있고 실측으로 검증됐다(``star_prior.py``). 이 함수가 새로 하는
    일은 그 원곡선을 region 목록 + dB 곡선으로 감싸는 것뿐이다(``postprocess.py``의
    ``_dominance_activity``와 동일한 감싸기, 파일 I/O만 뺐다 — 서버는 이미 로드된 파형을
    받는다).
    """
    try:
        made = vocal_presence_from_stems(
            vocals_waveform, accomp_waveform, sample_rate,
            smooth_sec=smooth_sec, hop_sec=hop_sec,
        )
    except (ValueError, TypeError) as exc:
        # 이 함수는 파일 I/O도 임포트도 안 한다 — 남는 실패 경로는 파형 배열 자체가
        # 기대한 모양/타입이 아닌 경우뿐이다(``vocal_presence_from_stems``/``frame_rms``는
        # 순수 numpy 산술). 그 밖의 예외(예: 진짜 버그)는 여기서 "우세도 계산 실패"로
        # 위장되지 않고 그대로 올라간다 — 부수 기능(표시 보정층)의 정상적 가용성 부재와
        # 코드 결함을 같은 로그 한 줄로 구분 못 하게 만들면 안 된다(운영자 지시).
        logger.warning(
            "우세도 계산 실패(%s: %s) — 표시 보정층을 건너뛴다",
            type(exc).__name__, exc, exc_info=True,
        )
        return None
    if made is None:
        return None
    values = made[1]
    hop = hop_sec

    regions: list[VocalRegion] = []
    run: int | None = None
    for index in range(len(values) + 1):
        active = index < len(values) and float(values[index]) >= _DOMINANCE_LEVEL
        if active and run is None:
            run = index
        elif not active and run is not None:
            if (index - run) * hop >= _DOMINANCE_MIN_SEC:
                regions.append(VocalRegion(start=run * hop, end=index * hop, energy=0.0))
            run = None
    if not regions:
        return None

    step = max(1, int(hop * sample_rate))
    usable = (len(vocals_waveform) // step) * step
    if usable <= 0:
        db = np.zeros(0, dtype=np.float64)
    else:
        frames = np.asarray(vocals_waveform[:usable]).reshape(-1, step)
        db = 20 * np.log10(np.maximum(np.sqrt((frames**2).mean(axis=1)), 1e-9))
    return DominanceActivity(regions=regions, values=values, hop=hop, db=db)


def _coverage(regions: list[VocalRegion], start: float, end: float) -> float:
    duration = max(1e-6, end - start)
    return sum(
        max(0.0, min(reg.end, end) - max(reg.start, start)) for reg in regions
    ) / duration


def _silent_run(regions: list[VocalRegion], start: float, end: float) -> float:
    """[start, end] 안에서 발성 구간에 안 덮인 **최장 연속** 무음 길이."""
    cursor, best = start, 0.0
    for reg in sorted(regions, key=lambda r: r.start):
        if reg.end <= start or reg.start >= end:
            continue
        best = max(best, reg.start - cursor)
        cursor = max(cursor, reg.end)
    return max(best, end - cursor)


# ── 장치 1: 병적으로 늘어진 라인 절단 (우세도 기반) ──


def _clamp_pathological(results: list[SyncResult], regions: list[VocalRegion]) -> set[int]:
    """지속 8초 초과 + 발성 커버리지 50% 미만인 라인을 글자 질량이 실린 발성 구간 끝으로 자른다.

    ``worker._clamp_stretched_lines``의 첫 규칙과 조건·절단 지점이 같다 — 다만 판정에 쓰는
    ``regions``가 VAD가 아니라 **우세도 region**이라는 점이 다르다(모듈 docstring 참고).
    분리 스템에서 VAD가 못 가르는 자리를 우세도가 가른다. ``_line_body_region``(글자
    질량이 가장 큰 리전을 찾는 로직)은 재구현하지 않고 worker.py에서 그대로 가져다 쓴다.
    정상 라인은 건드리지 않는다. 반환: 클램프된 라인 인덱스 집합.
    """
    from everyric2.server.worker import _line_body_region

    clamped: set[int] = set()
    for index, result in enumerate(results):
        duration = result.end_time - result.start_time
        if duration <= 8.0:
            continue
        line_regions = [
            reg for reg in regions
            if reg.end > result.start_time and reg.start < result.end_time
        ]
        if not line_regions:
            continue
        vocal = sum(
            min(reg.end, result.end_time) - max(reg.start, result.start_time)
            for reg in line_regions
        )
        if vocal / duration >= 0.5:
            continue
        body = _line_body_region(result, line_regions)
        if body is None:
            continue
        new_end = min(result.end_time, max(body.end + 0.3, result.start_time + 1.5))
        if new_end < result.end_time:
            result.end_time = new_end
            clamped.add(index)
    return clamped


# ── 장치 2: 간주에 좌초한 반복 렌디션 접기 ──


def _fold_stranded_repeats(results: list[SyncResult], regions: list[VocalRegion]) -> int:
    """간주에 좌초한 **반복 렌디션**을 앞 렌디션 바로 뒤로 접는다.

    Madeon(The Prince) 라인29가 원형이다. 같은 가사가 두 번 연속 불리는데(PROD 실측
    1:25.27~1:26.74 · 1:26.74~1:30.19), 앵커가 두 번째 렌디션을 62초 간주 위(2:02~2:14)에
    앉혔다 — 신스 스탭이 가사와 호환되는 가짜 증거로 들려 DP가 속았다.

    꼬리 클램프(``_pull_disconnected_tails``)는 여기 무력하다 — 마지막 세그가 발성 위에
    있어 자를 무음 꼬리가 없다. 라인 전체가 잘못 놓인 것이므로 라인 단위로 옮기는
    수밖에 없다. 조건(지속 6초 이상 + 커버리지 0.30 미만 + 직전 라인과 같은 텍스트)은
    13곡 스캔에서 오검출 0으로 확인됐다 — 바꾸지 않았다.

    세그는 앞 렌디션의 리듬을 복사한다(같은 텍스트를 같은 곡조로 부르므로). 세그 수가
    다르면 비례 배치로 물러선다. 반환: 접힌 라인 수.
    """
    folded = 0
    for index in range(1, len(results)):
        line, prev = results[index], results[index - 1]
        start, end = float(line.start_time), float(line.end_time)
        segs = line.word_segments or []
        if end - start < 6.0 or not segs:
            continue
        if str(line.text or "").strip() != str(prev.text or "").strip():
            continue
        prev_segs = prev.word_segments or []
        if not prev_segs or float(prev.end_time) >= start:
            continue
        if _coverage(regions, start, end) >= 0.30:
            continue
        next_start = (
            float(results[index + 1].start_time) if index + 1 < len(results) else float("inf")
        )
        prev_start, prev_end = float(prev.start_time), float(prev.end_time)
        room = next_start - 0.05 - prev_end
        if room < 0.8:
            continue
        new_start = prev_end
        length = min(prev_end - prev_start, room)
        line.start_time = round(new_start, 3)
        line.end_time = round(new_start + length, 3)
        if len(segs) == len(prev_segs):
            scale = length / max(1e-6, prev_end - prev_start)
            for seg, ref in zip(segs, prev_segs):
                seg.start = round(new_start + (float(ref.start) - prev_start) * scale, 3)
                seg.end = round(new_start + (float(ref.end) - prev_start) * scale, 3)
        else:
            step = length / len(segs)
            for pos, seg in enumerate(segs):
                seg.start = round(new_start + pos * step, 3)
                seg.end = round(new_start + (pos + 1) * step, 3)
        folded += 1
    return folded


# ── 장치 3: 몸통과 끊긴 꼬리 세그 되당김 ──


def _pull_disconnected_tails(results: list[SyncResult], regions: list[VocalRegion]) -> int:
    """몸통과 무음으로 끊긴 라인 꼬리 세그를 몸통 끝으로 되당긴다.

    rookie 0:14가 원형이다 — 마지막 두 세그가 몸통(~0:11.0)에서 3.3초 떨어진 자리에
    앉았는데, 그 자리는 가사에 없는 캐치프레이즈였다. butcher 2:08도 같은 꼴. 판정: 라인
    안 인접 세그 간격 ≥2초이고 그 사이 연속 무음 ≥1초.

    **꼬리는 최대 2세그만** — 확정 결함 5건이 전부 1~2세그였다. 소절 반쪽(10세그+)이
    무음 뒤에 있으면 그건 잔해가 아니라 **긴 휴지 뒤의 진짜 뒷소절**이다. ロキ
    「はあ… 寝言は寝て言えベイビー」(2:36.8 뒤 8초 쉬고 2:45.8부터 가창)를 이 가드 없이
    당겼다가 진짜 가사 자리가 통째로 비어 추임새 띠로 둔갑했다 — **오검출 1호**.
    이 가드는 반드시 테스트로 고정한다(호출부 요구사항).

    꼬리 세그는 몸통 끝 +0.3초 지점으로 모으고 라인 끝도 거기로 줄인다. 반환: 되당긴
    라인 수.
    """
    pulled = 0
    for index, line in enumerate(results):
        segs = line.word_segments or []
        cut: int | None = None
        for k in range(len(segs) - 1):
            body_end = float(max(segs[k].end, segs[k].start))
            tail_start = float(segs[k + 1].start)
            if tail_start - body_end < 2.0:
                continue
            if _silent_run(regions, body_end, tail_start) < 1.0:
                continue
            # 꼬리 ≤2세그 가드 — 오검출 1호(ロキ) 방어. 이 조건을 완화하면 안 된다.
            if len(segs) - (k + 1) > 2:
                continue
            cut = k
            break
        if cut is None:
            continue
        body_end = float(max(segs[cut].end, segs[cut].start))
        next_start = (
            float(results[index + 1].start_time) if index + 1 < len(results) else float("inf")
        )
        target = round(min(body_end + 0.3, next_start - 0.01), 3)
        target = max(target, body_end)
        for seg in segs[cut + 1:]:
            seg.start = seg.end = target
        line.end_time = round(max(float(line.start_time), min(float(line.end_time), target)), 3)
        pulled += 1
    return pulled


# ── 장치 4: 무음 위에 앉은 라인 머리 스냅 ──


def _snap_silent_heads(results: list[SyncResult], activity: DominanceActivity) -> int:
    """«명백한 무음» 위에 앉은 라인 머리를 첫 가창 온셋으로 되민다.

    rookie 라인0이 원형 — 「Boo」가 0:00.00(우세도 0.00 · -180dBFS 디지털 무음)에서
    켜지는데 실제 소리는 1.22초 뒤에 난다.

    조건(14곡 스캔에서 3건 전부 실결함·오검출 0):
      · 시작 우세도 < 0.12(발성 바닥) · 시작부터 가창(≥0.35) 온셋까지 무음 런 ≥0.5초
      · 온셋이 라인 안에 있고 온셋 이후 커버리지 ≥0.4 — 통째로 무음에 앉은 좌초 라인은
        대상이 아니다(그건 접기/블록 계열의 문제).
      · 시작~온셋의 절대 크기 최대 < -30dBFS — **이 가드가 결정적이다.** 빼면 우세도만
        낮은 소프트 어택이 걸려 진짜 온셋을 잘라먹는다.

    온셋은 «우세도 ≥0.35 이면서 들리는(≥-30dBFS) 첫 프레임»이다. dB 축 없이 우세도만
    보면 디지털 무음의 0/0 잡음에 걸려 멈춘다. 반환: 스냅된 라인 수.
    """
    values, db, hop = activity.values, activity.db, activity.hop
    if values is None or db is None:
        return 0
    snapped = 0
    total = len(values)
    audible_total = min(total, len(db))
    for line in results:
        start, end = float(line.start_time), float(line.end_time)
        k = int(start / hop)
        if k >= total or float(values[k]) >= 0.12:
            continue
        onset = k
        while onset < audible_total and (
            float(values[onset]) < _DOMINANCE_LEVEL or float(db[onset]) < _SILENT_DB
        ):
            onset += 1
        if onset >= audible_total or (onset - k) * hop < 0.5:
            continue
        onset_t = onset * hop
        if onset_t >= end:
            continue
        hi = min(total, int(end / hop))
        covered = sum(1 for x in values[onset:hi] if float(x) >= _DOMINANCE_LEVEL)
        if hi <= onset or covered / (hi - onset) < 0.4:
            continue
        window = db[k:onset]
        if window.size and float(window.max()) >= _SILENT_DB:
            continue
        new_start = round(max(start, onset_t - 0.05), 3)
        shift = new_start - start
        segs = line.word_segments or []
        untouched = next(
            (float(s.start) for s in segs if float(s.start) >= new_start), end,
        )
        for seg in segs:
            if float(seg.start) >= new_start:
                continue
            duration = max(0.0, float(seg.end) - float(seg.start))
            seg.start = round(min(float(seg.start) + shift, untouched), 3)
            seg.end = round(min(end, float(seg.start) + duration), 3)
        line.start_time = new_start
        snapped += 1
    return snapped


# ── 좌초 시그니처 탐지기 (이동 장치가 아니다) ──


def _stranded_sites(results: list[SyncResult], regions: list[VocalRegion]) -> int:
    """좌초 시그니처 — «감당 안 되는 난이도»의 사후 판정 신호.

    조건: 직전 갭 ≥3초 · 갭 안 미설명 발성 ≥1.5초 · 다음 라인 **머리 1초** 커버리지 <0.2.
    라인 전체가 아니라 머리로 재는 이유: 제자리 라인은 머리부터 제 발성 위에 있으므로
    머리로 재는 것이 원리에도 맞고, 꼬리 가드(≤2세그) 이후 뒤로 뻗은 정상 라인이 전체
    커버리지로는 오검출되는 것을 막는다. 14곡 스캔 오검출 0.

    장치(라인을 옮기는 것)로는 기각됐다 — 옮기면 비워진 자리가 새 증상이 된다. 하지만
    **탐지기로는 유효하다**: 이 시그니처가 남아 있다는 것은 가사 어딘가가 제 발성을 못
    찾았다는 뜻이고, 상위 단계(재정렬 승급 판단 등)가 이 카운트를 신호로 쓸 수 있다.
    """
    count = 0
    for index in range(1, len(results)):
        prev, line = results[index - 1], results[index]
        gap0, gap1 = float(prev.end_time), float(line.start_time)
        if gap1 - gap0 < 3.0:
            continue
        start, end = float(line.start_time), float(line.end_time)
        if _coverage(regions, start, min(end, start + 1.0)) >= 0.2:
            continue
        inside = [reg for reg in regions if reg.end > gap0 + 0.05 and reg.start < gap1 - 0.05]
        voiced = sum(min(reg.end, gap1) - max(reg.start, gap0) for reg in inside)
        if voiced >= 1.5:
            count += 1
    return count


def apply_stranded_corrections(
    results: list[SyncResult], activity: DominanceActivity
) -> dict[str, int]:
    """접기 → 꼬리 되당김 → 머리 스냅 → 좌초 탐지, 원본 ``_apply`` 말미와 같은 순서.

    이 순서가 의미를 갖는다: 접기를 가장 먼저 돌려야 좌초 라인이 제자리로 가면서 그
    안의 끊긴 꼬리도 함께 옮겨진다 — 순서를 바꾸면 아직 안 접힌 라인의 꼬리를
    꼬리-되당김이 먼저 뭉개 버린다. ``_clamp_pathological``은 이 묶음 밖이다(모듈
    docstring "실행 순서" 참고) — worker.py 배선 시 더 이른 단계에서 별도로 호출한다.
    이 함수 자체는 아무 곳에도 배선돼 있지 않다 — 최종 배선은 별도 작업이다.
    """
    regions = activity.regions
    folded = _fold_stranded_repeats(results, regions)
    pulled = _pull_disconnected_tails(results, regions)
    snapped = _snap_silent_heads(results, activity)
    stranded = _stranded_sites(results, regions)
    return {
        "folded_lines": folded,
        "pulled_tails": pulled,
        "snapped_heads": snapped,
        "stranded_sites": stranded,
    }


# ── 추임새 탐지 ──
# 「가사가 주장하지 않았는데 부르고 있는 시간」. 두 신호만 쓴다 — 정렬 세그(주장한 시간)와
# 보컬 우세도(부르고 있는가). UST는 안 쓴다(karaoke_review.py 모듈 주석 참고).
#
# **이것은 후보이지 판정이 아니다** — 추임새인지 누락된 가사인지 늘임음인지는 자동으로
# 못 가른다. 화면에 띄워 귀로 판정하는 것이 원래 용도였다. 서버 반환 계약은 이 함수의
# docstring에 명시한다: 곡 단위 [(시작, 끝), ...] 리스트다. 세그먼트가 아니라 **응답
# 최상위**에 실릴 값이므로, 이 함수는 계산만 한다 — 응답에 싣는 배선은 별도 작업이다.
#
# bench 원본은 언어별로 다른 "채택 레인"(ja/en) 중 가사 표기(가나 유무)로 하나를 골라서
# 그 레인의 세그만 봤다 — 벤치가 **여러 정렬기 후보를 동시에 갖고 있기 때문**이다. 서버는
# 이미 확정된 정렬 결과 하나(``results``)만 가지므로 레인 선택 로직은 필요 없다 — 이
# 단순화는 동작 차이가 아니라 "벤치는 다중 후보, 서버는 단일 결과"라는 구조 차이의 반영이다.
#
# bench는 dB 곡선을 두 갈래로 따로 계산했다(postprocess.py의 ``_dominance_activity.db``와
# karaoke_review.py의 ``_vocal_db_curve`` — 공식은 동일: 20*log10(rms(vocals))). 여기서는
# ``DominanceActivity.db`` 하나로 합쳤다 — 같은 계산을 두 번 하지 않는다.
ADLIB_LEVEL = 0.35
ADLIB_MIN_SEC = 0.40
# 뒤 세그와 이만큼은 떨어져 있어야 한다 — 붙어 있으면 다음 라인의 실제 온셋을 정렬이
# 늦게 잡은 것이지 추임새가 아니다. 실측: 오검출 0.02초, 정검출 전부 0.12초 이상.
ADLIB_MIN_TAIL_GAP = 0.10
# 추임새 직전엔 우세도가 바닥까지 꺼졌다 올라온다(재-어택 흔적). 늘임음은 매끄럽게
# 이어진다. 21건 청취에서 추임새 0.000~0.208 · 늘임음 0.854~0.973으로 안 겹쳤다.
ADLIB_DIP_LEVEL = 0.25
ADLIB_DIP_WINDOW = 0.40
# 앞 음절 온셋까지 이만큼은 떨어져 있어야 한다 — 코앞이면 그 음절의 끝자락이다.
# 30건 청취: 추임새는 0.70초 이상, 실가사 끝자락은 0.06~0.09초.
ADLIB_MIN_ONSET_GAP = 0.35
# 보컬 절대 크기 하한(dBFS) — 우세도는 비율이라 디지털 무음(0/0)의 잡음 튐을 못 본다.
ADLIB_MIN_DB = -30.0


def adlib_candidates(
    results: list[SyncResult], activity: DominanceActivity
) -> list[tuple[float, float]] | None:
    """설명 안 된 가창 구간 [(시작, 끝), ...] — 곡 단위, 세그가 아니다.

    반환 계약: 라인이나 세그 배열에 섞이지 않는다. 호출자(최종 배선)가 응답 최상위에
    별도 필드(예: ``adlib_regions``)로 얹는다. 후보가 없으면 ``None``.
    """
    values, hop, db = activity.values, activity.hop, activity.db
    if values is None or hop is None:
        return None

    every = sorted(
        (seg for result in results for seg in (result.word_segments or [])),
        key=lambda seg: float(seg.start),
    )
    if not every:
        return None

    merged: list[list[float]] = []
    for seg in every:
        start, end = float(seg.start), float(max(seg.end, seg.start))
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    free: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            free.append((cursor, start))
        cursor = max(cursor, end)
    free.append((cursor, len(values) * hop))

    starts = [float(seg.start) for seg in every]  # every가 이미 start 오름차순

    def loud(t0: float, t1: float) -> bool:
        # 소리가 실제로 나는가 — 우세도는 비율이라 이걸 못 본다(디지털 무음 0/0 잡음 방어)
        if db is not None and len(db):
            lo = int(t0 / hop)
            hi = min(len(db), max(lo + 1, int(t1 / hop)))
            if hi > lo and float(db[lo:hi].max()) < ADLIB_MIN_DB:
                return False
        return True

    def keep(t0: float, t1: float) -> bool:
        # ⓪ 3초 넘게 이어지는 설명 안 된 발성은 늘임음도 온셋 끝자락도 아니다 — 온셋갭·딥
        #    규칙을 면제한다(절대 크기만 본다). ロキ 2:48~2:54 스크림이 원형.
        if t1 - t0 >= 3.0:
            return loud(t0, t1)
        # ① 다음 세그에 붙어 있으면 그 세그의 온셋이지 추임새가 아니다
        following = next((x for x in starts if x >= t1 - 0.01), None)
        if following is not None and following - t1 < ADLIB_MIN_TAIL_GAP:
            return False
        # ② 앞 음절 온셋이 코앞이면 그 음절의 끝자락이다
        preceding = max((x for x in starts if x <= t0 + 0.01), default=None)
        if preceding is not None and t0 - preceding < ADLIB_MIN_ONSET_GAP:
            return False
        # ③ 직전에 우세도가 꺼졌다 올라왔는가 — 늘임음이면 매끄럽게 이어진다
        low = max(0, int((t0 - ADLIB_DIP_WINDOW) / hop))
        high = min(len(values), int(t0 / hop) + 2)
        if high > low and min(values[low:high]) > ADLIB_DIP_LEVEL:
            return False
        # ④ 소리가 실제로 나는가
        return loud(t0, t1)

    out: list[tuple[float, float]] = []
    for start, end in free:
        lo, hi = int(start / hop), min(len(values), int(end / hop))
        run: int | None = None
        for index in range(lo, hi + 1):
            if index < hi and float(values[index]) >= ADLIB_LEVEL:
                if run is None:
                    run = index
            elif run is not None:
                t0, t1 = round(run * hop, 3), round(index * hop, 3)
                if t1 - t0 >= ADLIB_MIN_SEC and keep(t0, t1):
                    out.append((t0, t1))
                run = None
    return out or None
