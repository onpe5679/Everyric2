"""옥타브 접기 게이팅 회귀 고정 — extractor의 두 판정 함수(순수 수치).

수치 목표 자체는 scripts/bench_melody_ust.py가 잰다(realistic 프로파일 F1 0.941,
2026-08-03 실측). 여기서는 그 수치를 만드는 두 판정의 **성질**을 고정한다:
① 12의 배수 근처 이탈만 접는다(임의 도약은 12의 배수가 아니다),
② 앞뒤 문맥이 **합의**할 때만 접는다(진짜 옥타브 도약은 뒤 문맥이 새 음역과
   일치해 판정이 갈린다 — 이 비대칭이 서브하모닉과 도약을 가르는 유일한 신호).
구버전(잔차 게이팅 없음)은 잡음 없는 f0에서도 정답 노트 811개를 파괴했다.
"""

import numpy as np

from everyric2.melody.extractor import _agreed_octave_shift, octave_fold_shift


def test_exact_octave_deviations_fold_back():
    dev = np.array([12.0, -12.0, 24.0, -24.3, 11.4])
    shift = octave_fold_shift(dev)
    assert shift.tolist() == [-12.0, 12.0, -24.0, 24.0, -12.0]


def test_wide_but_non_octave_leaps_are_untouched():
    # 16·19반음(장6도+옥타브 등)은 실멜로디의 넓은 도약 — 12의 배수가 아니므로 접지 않는다
    dev = np.array([16.0, -19.0, 10.0, 14.2])
    assert octave_fold_shift(dev).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_small_deviations_below_min_dev_are_untouched():
    dev = np.array([0.4, -3.0, 7.9])
    assert octave_fold_shift(dev).tolist() == [0.0, 0.0, 0.0]


def test_agreed_shift_folds_only_when_both_contexts_agree():
    vals = np.array([60.0, 48.0, 72.0])
    prev_ref = np.full(3, 60.0)
    next_ref = np.full(3, 60.0)
    # 앞뒤 모두 기준 60 — 48(−12 이탈)만 +12로 접힌다, 72(+12 이탈)는 −12로
    assert _agreed_octave_shift(vals, prev_ref, next_ref).tolist() == [0.0, 12.0, -12.0]


def test_genuine_octave_jump_with_following_context_is_kept():
    # 멜로디가 진짜로 한 옥타브 올라가 머무르는 프레임: 앞 기준과는 +12 어긋나지만
    # 뒤 기준(새 음역)과는 일치 — 합의가 깨져 접지 않는다.
    vals = np.array([72.0])
    prev_ref = np.array([60.0])
    next_ref = np.array([72.0])
    assert _agreed_octave_shift(vals, prev_ref, next_ref).tolist() == [0.0]
