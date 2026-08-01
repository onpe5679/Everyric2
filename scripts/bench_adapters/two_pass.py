"""2패스 정렬 어댑터 — 앵커가 라인 경계를 잡고, 경량 모델이 그 창 안에서 음절만 다시 잡는다.

동기는 UST 준정답 채점의 두 축이 서로 다른 모델을 가리켰다는 것이다. **라인 축**(라인 시작
≤0.15s)에서는 OWSM-CTC가 붕괴곡을 하나도 만들지 않는 유일한 후보였고(극한 5곡 최악 71%),
**음절 축**(세그 온셋 ≤0.10s)에서는 OWSM이 63~69%로 처지는 대신 경량 단일 언어 후보들이
84% 대를 냈다 — 대신 그쪽은 어려운 곡에서 32~49%로 무너졌다. 두 축의 승자가 다른 이유는
구조적이다: OWSM은 50k SentencePiece라 한 토큰이 여러 글자를 덮고 그 내부는 **보간**이며,
라인 시작은 항상 토큰 경계라 그 손실이 라인 축에 안 잡힌다.

그래서 라인 경계는 앵커가 정하고, 각 라인의 창 안에서만 경량 모델이 음절을 다시 잡는다.
전역 DP를 창으로 쪼개면 경량 모델이 곡 전체에서 미끄러질 여지 자체가 사라지므로, 붕괴는
앵커가 막고 해상도는 경량 모델이 채우는 구조가 된다.

설계 불변식 넷:

* **라인 경계는 2패스가 절대 못 건드린다.** ``line["start"]``/``end``/``confidence``는 앵커
  값 그대로다. 앵커의 라인 강건성이 보존되는 지점이 정확히 여기다.
* **하한이 앵커 단독으로 고정된다.** 창이 토큰 수보다 짧거나, vocab에 없는 문자뿐이거나,
  DP가 실패하면 그 라인은 앵커 세그를 그대로 쓴다. 즉 이 구조는 앵커보다 나빠질 수 없다.
* **emission은 곡당 한 번.** 라인마다 forward를 다시 돌리는 게 아니라 전곡 emission
  (``HFCTCAligner.emission_for``)을 프레임 축에서 잘라 쓴다.
* **표기는 패스별 모국어.** 앵커는 가나(``--input-mode pron-kana``), 경량 모델이 ja면 가나
  그대로, ko면 결정론 음차로 한글로 바꿔 정렬한다. 세그의 ``t``는 어느 경로든 **원문(가나)**
  글자로 되돌려 붙이므로 하류(뷰어·음절 채점)에서 표기가 갈리지 않는다.

VRAM: 앵커(OWSM)는 별도 venv 서브프로세스라 정렬 시점에 이미 종료되어 있고, 경량 모델만
이 프로세스에 남는다. 그래서 보고 peak은 두 패스의 **최댓값**이고, 두 값 모두 meta에 남긴다.
프로드에서 둘을 동시에 상주시킬 때의 가중치 합은 별개 문제이므로 여기서 재지 않는다.
"""

from __future__ import annotations

import logging
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.benchmark_alignment import AlignerAdapter, AlignOut, VramProbe

logger = logging.getLogger(__name__)

# 창을 앵커 라인 경계에 딱 맞추면 2패스가 실제 발화의 앞뒤를 창 밖에 두고 잘라먹는다. 앵커의
# 라인 시작 오차 자체가 ≤0.15s 기준으로 재던 값이므로 그와 같은 자릿수의 여유를 준다.
WINDOW_PAD_SEC = 0.2


@dataclass(frozen=True)
class TwoPassConfig:
    """앵커 × 경량 정렬기 한 쌍."""

    name: str
    anchor: str
    refiner: str
    # ``native`` = 입력 표기를 그대로 먹인다(경량 모델의 모국어이거나 다국어라 변환이 없다).
    # ``hangul`` = 경량 모델이 ko 전용이라 결정론 음차(kana_hangul)로 한글 독음을 만들어 먹인다.
    #              한글 입력에는 이 변환이 항등이므로 ko 곡에서는 표기 손실이 없다.
    refiner_script: str
    note: str = ""
    # 오디오 심판 — 사전이 발음을 하나로 못 정하는 낱말(en 가사 출현의 36.35%)을 오디오에
    # 묻는다. 낱말마다 대체 발음으로 타깃을 바꿔 재정렬하고 span score가 오르면 채택한다.
    # 라인당 정렬 횟수가 후보 수만큼 늘어나므로 켤 때는 비용을 함께 본다.
    referee: bool = False
    # 음절 수는 **오디오 강도 봉우리**가 정한다(``_energy_nuclei``). 정렬 점수는 길이 편향이
    # 있어 음절 수를 공정하게 못 겨루기 때문이다. 후보끼리 음절 수가 같으면 이 신호는
    # 의견이 없고 정렬 점수가 그대로 결정한다 — 그래서 ja(표기만 갈림)에는 영향이 없다.
    energy_syllables: bool = False
    # 길이가 다른 후보도 견줄 것인가. 죽은 토큰 30%를 걷어낸 뒤 길이 편향이 실제로 남아 있는지
    # 다시 재기 위한 스위치다 — 켜면 ``our`` auer(2음절) vs aur(1음절)처럼 **음절 수가 진짜로
    # 다른** 후보가 심판에 올라온다.
    allow_length_change: bool = False
    # 한 시각에 뭉친 세그를 발성 구간에 펴 줄 것인가(``_spread_piled_segments``).
    spread_piles: bool = True
    # 반복 훅에서 렌디션을 건너뛴 자리를 되돌릴 것인가(``_respace_repeated_lines``).
    respace_repeats: bool = True
    # 세그 끝을 다음 세그 시작까지 늘릴 것인가(노래방 표시 규약). 프로드가 이미 하는 일이고
    # 하네스만 빠져 있었다 — ``_extend_segments`` 참조.
    extend_segments: bool = True
    # 혼합 경로에서 **라틴 낱말**도 심판에 올릴 것인가. 기본은 끔 — numb numb에서 ``color``를
    # 12번 전부 ``코러``로 바꿨는데 사용자 청취는 ``커러``였다(2026-08-02). 타이밍에는 영향이
    # 없고(두 후보가 같은 음절 구조라 세그가 동일) **발음 표기만** 갈리므로 지금 지표로는
    # 아예 안 잡힌다. 그 곡의 라틴 구간은 방출 신뢰도가 0.244로 무너져 있었다(가나 0.447,
    # rookie 라틴 0.627) — 국소 신뢰도 게이트가 답으로 보이나 오답 1건으로 문턱을 정할 수는 없다.
    latin_referee: bool = False


CONFIGS: tuple[TwoPassConfig, ...] = (
    # ja 네이티브 경로. reazon-hubert-base는 가중치 376MB(98M)로 후보 중 두 번째로 가볍고,
    # 음절 축 단독 84.8%로 측정된 최고값이다(대신 극한곡 최악 32% — 그 붕괴를 앵커가 막는다).
    # 가나를 그대로 먹이므로 표기 변환이 아예 없다. 단 ja 전용이라 다른 층에는 못 쓴다.
    TwoPassConfig(
        name="2pass-owsm-reazon",
        anchor="owsm-ctc-v4-1b-bf16",
        refiner="hf-reazon-hubert-base",
        refiner_script="native",
        note="OWSM 라인 창 + reazon-hubert 음절 (ja 전용, 표기 변환 없음)",
    ),
    # ★다국어 경로. omniasr는 ja 음절 84.7%로 reazon(84.8)과 동급이면서 **모든 층을 한 모델로**
    # 덮으므로 언어별 경량 모델 라우팅이 통째로 필요 없어진다. Apache-2.0 완전 클린이라
    # owsm의 CC-BY 2급 리스크 외에 라이선스 부담을 더하지도 않는다.
    TwoPassConfig(
        name="2pass-owsm-omniasr",
        anchor="owsm-ctc-v4-1b-bf16",
        refiner="omniasr-ctc",
        refiner_script="native",
        note="OWSM 라인 창 + omniASR 음절 (다국어 단일 경로, 표기 변환 없음)",
    ),
    # ko 경로. kkonjeong은 가중치 360MB(94M)로 후보 중 가장 가볍고 vocab이 54-way 자모라
    # 일본어 요음의 OOV 위험이 구조적으로 없다. ja 곡에서는 가나→한글 음차 한 단계가 붙어
    # 모라가 뭉치지만(なっ→낫), ko 곡에서는 그 변환이 항등이라 모국어 그대로 정렬한다.
    TwoPassConfig(
        name="2pass-owsm-kkonjeong",
        anchor="owsm-ctc-v4-1b-bf16",
        refiner="hf-kkonjeong",
        refiner_script="hangul",
        note="OWSM 라인 창 + kkonjeong 음절 (ko 전용, ja 곡은 한글 음차 경유)",
    ),
    # en 경로 두 갈래. 앵커는 **영어 원문**으로 라인을 잡고(한글 강제가 라인을 깎던 문제가
    # 구조적으로 차단된다) 경량 모델 몫에만 음차를 적용한다. 두 갈래의 트레이드오프:
    #   latin-kana   — 프로드 en 표시(latin_to_kana)와 «같은 표기»라 음절 수가 어긋날 위험이
    #                  없다. 대신 가나는 b/p를 구분 못 해 beautiful이 ピオティポル로 깨진다.
    #   latin-hangul — 음차가 더 정확하다(비어티펄). 대신 표시(가나)와 음절 수가 달라질 수
    #                  있어 매핑이 실패하면 그라데이션 폴백이다.
    # 어느 쪽이 실제로 나은지는 en UST 표본이 2곡뿐이라 수치로 못 가른다 — 청취 판정용이다.
    TwoPassConfig(
        name="2pass-en-kana",
        anchor="owsm-ctc-v4-1b-bf16",
        refiner="omniasr-ctc",
        refiner_script="latin-kana",
        note="OWSM 라인 창(영어 원문) + omniASR 음절 (가나 음차 — 프로드 표시와 일치)",
    ),
    TwoPassConfig(
        name="2pass-en-hangul",
        anchor="owsm-ctc-v4-1b-bf16",
        refiner="omniasr-ctc",
        refiner_script="latin-hangul",
        note="OWSM 라인 창(영어 원문) + omniASR 음절 (한글 음차 — 음차 정확도 우선)",
    ),
    # 발음 사전 경유 두 갈래. 위 둘은 **철자**에서 출발하는 규칙 음차라 영어의 철자-발음
    # 괴리를 못 넘는다(beautiful /ˈbjuːtɪfʊl/ → 비어티펄). CMU 사전은 음소를 주므로
    # 그 벽이 사라진다(→ 뷰터펄). 그 음소를 어디까지 가져갈지가 두 갈래의 차이다.
    TwoPassConfig(
        name="2pass-en-cmu",
        anchor="owsm-ctc-v4-1b-bf16",
        refiner="omniasr-ctc",
        refiner_script="latin-cmu",
        note="OWSM 라인 창 + omniASR 음절 (CMU 발음 사전 → 한글 조밀 음차)",
    ),
    TwoPassConfig(
        name="2pass-en-ipa",
        anchor="owsm-ctc-v4-1b-bf16",
        refiner="omniasr-ctc",
        refiner_script="latin-ipa",
        note="OWSM 라인 창 + omniASR 음절 (CMU 발음 사전 → IPA 직접 정렬)",
    ),
    # ★채택 후보 — 정렬은 IPA(가장 정확), 표시는 사람이 읽는 글자. 위 ``2pass-en-ipa``와
    # **정렬 결과가 완전히 같고** 세그의 텍스트·묶음만 다르다(음소 단위 → 음절 단위).
    TwoPassConfig(
        name="2pass-en-ipa-hangul",
        anchor="owsm-ctc-v4-1b-bf16",
        refiner="omniasr-ctc",
        refiner_script="latin-ipa-hangul",
        note="OWSM 라인 창 + omniASR 음절 (IPA 정렬 → 한글 음절 표시)",
    ),
    TwoPassConfig(
        name="2pass-en-ipa-kana",
        anchor="owsm-ctc-v4-1b-bf16",
        refiner="omniasr-ctc",
        refiner_script="latin-ipa-kana",
        note="OWSM 라인 창 + omniASR 음절 (IPA 정렬 → 가나 표시)",
    ),
    # ★앵커를 자기 자신으로 — owsm 없이 라인 창을 얻는다.
    #
    # 세 구성 대조에서 owsm 앵커는 값을 못 했다(정렬 22배 느린데 MAE는 0.08 → 0.14로 악화).
    # 그렇다고 창이 무용하다는 뜻은 아니다 — 창은 **음절 재정렬이 라인 밖으로 새는 것**을
    # 막는 장치이고, 그 값은 owsm이 아니라 창 자체에서 온다. 그렇다면 창을 asr이 스스로
    # 1패스로 잡으면 된다: 라틴 원문으로 라인 경계만 얻고(그 용도로는 라틴도 충분한지가
    # 이 실험의 물음이다), 2패스에서 IPA로 그 창 안에서만 음절을 나눈다.
    #
    # 비용은 같은 모델 두 번이라 ~1.6초 — owsm 18초 자리에 들어간다.
    # 심판은 **기본으로 켠다**. 사전 첫 발음을 고정하면 음절 수까지 사전이 정해 버리는데,
    # 어느 음절 수로 불렀는지는(our 1·2음절, fire 1·2음절) 곡마다 다르고 오디오만 안다.
    # 후보가 있던 라인의 76.3%에서 첫 발음이 뒤집혔다 — 고정은 틀린 기본값이었다.
    TwoPassConfig(
        name="2pass-asr-ipa-hangul",
        anchor="omniasr-ctc",
        refiner="omniasr-ctc",
        refiner_script="latin-ipa-hangul",
        referee=True,
        allow_length_change=True,
        note="omniASR 자기 라인 창 + ASCII 음소 음절 + 발음 오디오 심판",
    ),
    # 가나 음차는 결과가 한글과 사실상 같아 뷰어에서 뺐다(사용자 판정 2026-08-01). 대신
    # **타깃 자신**을 보여주는 레인을 둔다 — 표기 변환을 거치지 않으므로 정렬기가 실제로
    # 맞추는 것이 그대로 보이고, 심판이 발음을 바꾼 자리도 IPA로 직접 읽힌다.
    TwoPassConfig(
        name="2pass-asr-ipa-phonetic",
        anchor="omniasr-ctc",
        refiner="omniasr-ctc",
        refiner_script="latin-ipa-phonetic",
        referee=True,
        allow_length_change=True,
        note="위와 같은 정렬, 표시는 음소 전사 자체(음절 단위)",
    ),
    # ★원문 영어 층 — 위 둘과 **같은 정렬**을 원문 철자 음절로 묶는다. 한 번 정렬해 여러
    # 해상도를 내는 것이 요점이라, 이 레인의 세그 경계는 **한글 레인 음절 스팬의 부분합과
    # 정확히 일치해야 한다**(검증 대상). 한글은 CV 구조라 더 잘게 쪼개진다(1.32배).
    TwoPassConfig(
        name="2pass-asr-ipa-en",
        anchor="omniasr-ctc",
        refiner="omniasr-ctc",
        refiner_script="latin-ipa-en",
        referee=True,
        allow_length_change=True,
        note="위와 같은 정렬, 표시는 영어 원문 음절(beau-ti-ful)",
    ),
    # 강도 봉우리를 **켠** 대조군 — 음절 수를 오디오 강도에 맡기는 안이었다. 청취 정답 17건에서
    # 정렬 점수만 14/17, 봉우리를 켜면 11~12/17로 **떨어진다**(offering·suffering·victory를
    # 전부 깨고 every 하나를 얻는다, 2026-08-02). 그래서 기본은 끔이고 이 레인만 켜 둔다.
    TwoPassConfig(
        name="2pass-asr-ipa-en-energy",
        anchor="omniasr-ctc",
        refiner="omniasr-ctc",
        refiner_script="latin-ipa-en",
        referee=True,
        energy_syllables=True,
        allow_length_change=True,
        note="음절 수를 오디오 강도 봉우리가 정하는 대조군 — 실측에서 손해",
    ),
    # 심판을 끈 대조군 — 심판이 «값을 하는가»를 재는 자리다. 표시 표기마다 하나씩 둔다:
    # 청취 판정은 **원문 영어**가 읽기 쉽고(글자를 아니까), 한글은 발음층이 어떻게 갈리는지를
    # 본다. 짝과 나란히 놓고 «같은 구간에서 음절이 몇 개로 갈리는가»를 비교하는 용도다.
    TwoPassConfig(
        name="2pass-asr-ipa-hangul-noref",
        anchor="omniasr-ctc",
        refiner="omniasr-ctc",
        refiner_script="latin-ipa-hangul",
        note="위와 같은 구성에서 **심판만 끈** 대조군 — 사전 첫 발음 고정",
    ),
    TwoPassConfig(
        name="2pass-asr-ipa-en-noref",
        anchor="omniasr-ctc",
        refiner="omniasr-ctc",
        refiner_script="latin-ipa-en",
        note="심판 끈 대조군, 표시는 영어 원문 음절",
    ),
    # ★ja 독음 심판 — 기존 @kana 경로(--input-mode pron-kana)와 **정렬·표시가 같고** 심판만
    # 얹는다. 한자를 가나로 펼치는 것 자체는 이미 @kana가 하고 있으므로 그건 실험 대상이
    # 아니다. 물음은 하나다: 독음이 갈리는 자리(生=なま/せい/いき)를 오디오가 고르면 나아지나.
    #
    # ``-noref``는 @kana와 같은 결과를 내야 한다 — 그 일치가 곧 이 경로가 옳게 구현됐다는
    # 증거이고, 일치하지 않으면 심판 비교 자체가 성립하지 않는다.
    TwoPassConfig(
        name="2pass-owsm-reading",
        anchor="owsm-ctc-v4-1b-bf16",
        refiner="omniasr-ctc",
        refiner_script="ja-reading",
        referee=True,
        note="@kana와 같은 가나 독음 정렬 + MeCab N-best 오디오 심판",
    ),
    TwoPassConfig(
        name="2pass-owsm-reading-noref",
        anchor="owsm-ctc-v4-1b-bf16",
        refiner="omniasr-ctc",
        refiner_script="ja-reading",
        note="표기형 독음(@kana와 같은 타깃) — phonetic 대조군",
    ),
    # ★이번 실험의 본체 — 위와 **한 축만** 다르다(표기형 → 발음형). 심판은 꺼 둔다:
    # 두 축을 같이 켜면 어느 쪽이 이득을 냈는지 못 읽는다.
    TwoPassConfig(
        name="2pass-owsm-reading-phon",
        anchor="owsm-ctc-v4-1b-bf16",
        refiner="omniasr-ctc",
        refiner_script="ja-reading-phon",
        note="발음형 독음 — 조사 + 장음(장음 ー가 vocab에 없어 커버리지 95.6→92.3%)",
    ),
    TwoPassConfig(
        name="2pass-owsm-reading-joshi",
        anchor="owsm-ctc-v4-1b-bf16",
        refiner="omniasr-ctc",
        refiner_script="ja-reading-joshi",
        note="조사만 발음형(は→わ·を→お·へ→え) — 커버리지 손실 없이 563회 교정",
    ),
    # ★★심판 비교 본체 — 프로드가 서버에서 실제로 쓰는 독음·후보 생성기 위에서 심판만 갈린다.
    TwoPassConfig(
        name="2pass-owsm-prod",
        anchor="owsm-ctc-v4-1b-bf16",
        refiner="omniasr-ctc",
        refiner_script="ja-prod",
        referee=True,
        note="프로드 독음(phonetic+루비) + 프로드 후보(_AMBIGUOUS_WORDS) 오디오 심판",
    ),
    TwoPassConfig(
        name="2pass-owsm-prod-noref",
        anchor="owsm-ctc-v4-1b-bf16",
        refiner="omniasr-ctc",
        refiner_script="ja-prod",
        note="같은 독음, 심판만 끔 — 이 둘의 차이가 곧 심판의 값이다",
    ),
    # ★혼합 표기 — 위와 같은 ja 독음에 **라틴 낱말 음절화 + 장음 펴기**만 얹는다.
    #
    # 이름 주의: 평가셋의 stratum: mixed와 **다른 뜻**이다. 그 층에만 도는 레인이 아니라
    # ja 전곡에 도는 채택 스택이고, 라틴이 하나도 없는 곡(深海少女·Kikuo)에도 그대로 돈다.
    # 「표기가 섞인 가사를 계열별로 나눠 각자에게 넘긴다」는 뜻이다. ja·mixed
    # 45곡에서 라틴이 타깃의 12.4%이고 5%를 넘는 곡이 16곡이라(numb numb 53.7%), 그 곡들은
    # 지금 가사 절반이 `n|u|m|b`처럼 글자 단위로 쪼개져 나온다.
    TwoPassConfig(
        name="2pass-owsm-mixed",
        anchor="owsm-ctc-v4-1b-bf16",
        refiner="omniasr-ctc",
        refiner_script="ja-mixed",
        referee=True,
        note="프로드 독음 + 라틴 음절화 + 장음 펴기 + 오디오 심판",
    ),
    # ★줄 사이 star 앵커 — 추임새·애드립·반복 후렴을 와일드카드가 흡수하게 한다. 정렬·표시는
    # 위와 같고 **앵커만** 갈린다. ja 7곡에서 라인이 반복 가창 위로 늘어난 자리가 둘 남아
    # 있었고(ほら ほら ほら 9.30s·6음절, Boo boo booing 6.65s·8음절) VAD 클램프로는 못 잡는다.
    TwoPassConfig(
        name="2pass-owsm-mixed-star",
        anchor="owsm-ctc-v4-1b-bf16-star",
        refiner="omniasr-ctc",
        refiner_script="ja-mixed",
        referee=True,
        note="혼합 표기 + 줄 사이 star 앵커(추임새 흡수 실험)",
    ),
    # ★끊긴 자리에만 star — 「명백히 계속 부르는」 줄 사이에는 안 넣는다. 보컬 우세도로 가른다.
    TwoPassConfig(
        name="2pass-owsm-mixed-stargap",
        anchor="owsm-ctc-v4-1b-bf16-stargap",
        refiner="omniasr-ctc",
        refiner_script="ja-mixed",
        referee=True,
        note="혼합 표기 + 끊긴 줄 사이에만 star(우세도 판정)",
    ),
    # ★한글 표시층 — **위와 정렬이 완전히 같고 표시만** 한글이다(``ja-mixed-hangul``).
    # 한국어 사용자가 일본어 곡을 읽는 층이고 프로드의 실제 기능이다. 한글을 정렬 타깃으로
    # 쓰면 omniASR이 한글 자모를 거의 못 내 무너지지만(실측 자모 0/24), 표시로만 쓰면
    # 정렬은 가나가 하고 화면에는 읽을 수 있는 글자가 나간다 — 타깃/표시 분리의 요점이다.
    # 부수 효과로 분절이 **더 옳아진다**: 가나 레인은 ``しゅ``를 し|ゅ 두 칸으로 쪼개는데
    # 한글은 ``슈`` 한 칸이고, ``ん``·``っ``도 받침으로 앞 칸에 흡수된다(しんかい → 신카이).
    TwoPassConfig(
        name="2pass-owsm-mixed-hangul",
        anchor="owsm-ctc-v4-1b-bf16",
        refiner="omniasr-ctc",
        refiner_script="ja-mixed-hangul",
        referee=True,
        note="위와 같은 정렬, 표시는 한글 음절(신카이 쇼조)",
    ),
    # 위와 **장음 축만** 다른 대조군. 라틴 라우팅과 장음 펴기를 같이 켜면 어느 쪽이 이득을
    # 냈는지 못 읽는다 — 첫 측정에서 라틴 없는 곡까지 떨어져 원인을 착각했다.
    TwoPassConfig(
        name="2pass-owsm-mixed-nolong",
        anchor="owsm-ctc-v4-1b-bf16",
        refiner="omniasr-ctc",
        refiner_script="ja-mixed-nolong",
        referee=True,
        note="라틴 음절화만 — 장음 펴기 없음(축 분리 대조군)",
    ),
    # 라틴 낱말까지 심판에 올린 대조군. 타이밍은 안 바뀌고 **발음 표기**만 갈린다 —
    # ``color`` 커러/코러. 현 지표(UST 음절 타이밍)로는 원리적으로 못 재고 청취로만 판정된다.
    TwoPassConfig(
        name="2pass-owsm-mixed-en",
        anchor="owsm-ctc-v4-1b-bf16",
        refiner="omniasr-ctc",
        refiner_script="ja-mixed",
        referee=True,
        latin_referee=True,
        note="혼합 표기 + 라틴 낱말까지 심판 — 실측에서 color 12건 오답",
    ),
)


# 2패스 구성 요소를 찾을 모듈들. 한 모듈이 못 뜨는 환경(extras 미설치)에서도 나머지 조합은
# 살아야 하므로 개별 실패는 건너뛴다 — 정작 필요한 이름이 없으면 아래에서 명시적으로 죽는다.
_COMPONENT_MODULES = ("hf_ctc", "omni_ctc", "owsm_ctc")


def _resolve_adapter(name: str) -> Any:
    """어댑터 이름 하나를 인스턴스로 — 각 모듈의 ``register``를 그대로 재사용한다."""

    import importlib

    registry: dict[str, Any] = {}
    for module_name in _COMPONENT_MODULES:
        try:
            importlib.import_module(f"scripts.bench_adapters.{module_name}").register(registry)
        except Exception as exc:
            logger.debug("2패스 구성 요소 모듈 %s 배선 실패: %r", module_name, exc)
    adapter = registry.get(name)
    if adapter is None:
        raise ValueError(
            f"2패스 구성이 가리키는 어댑터가 없다: {name!r} "
            f"(탐색 모듈: {', '.join(_COMPONENT_MODULES)})"
        )
    return adapter()


def _native_units(source: str) -> tuple[str, list[str]]:
    """입력 표기를 그대로 쓰는 경로 — 정렬 텍스트와 문자별 원문 소유자가 동일하다."""

    return source, list(source)


def _hangul_units(source: str) -> tuple[str, list[str]]:
    """가나 한 줄을 한글 독음으로 바꾸고, 한글 글자마다 그것이 나온 **원문 가나**를 짝지어 준다.

    ``kana_to_hangul``을 접두사마다 다시 불러 출력 길이가 언제 늘어나는지로 귀속을 정한다.
    로직을 베껴 오는 대신 실제 변환 함수를 그대로 쓰므로 음차 규칙이 바뀌어도 여기가 따라간다
    (촉음 っ·발음 ん은 앞 음절에 종성으로 붙어 길이가 안 늘고, 요음 きゃ는 두 글자가 한 음절이
    되는데 둘 다 "길이가 안 늘면 직전 출력 글자의 소유"라는 같은 규칙으로 처리된다).
    줄 길이가 수십 자라 접두사 재호출 비용은 무시할 수 있다.
    """

    from everyric2.text.kana_hangul import kana_to_hangul

    text = kana_to_hangul(source)
    if not text:
        return "", []
    owners = [""] * len(text)
    for index, char in enumerate(source):
        produced = len(kana_to_hangul(source[: index + 1]))
        owners[min(max(produced - 1, 0), len(text) - 1)] += char
    return text, owners


def _latin_units(source: str, mode: str) -> tuple[str, list[str]]:
    """라틴 한 줄을 음차하고, **음차 글자 자신**을 소유자로 둔다.

    ja 경로(``_hangul_units``)와 달리 원문으로 되돌리지 않는다. ja 곡의 표시는 원문 가나지만
    **라틴 곡의 프로드 표시는 음차 결과 자체**(``worker._attach_latin_pron_variants`` →
    ``latin_to_kana``)이기 때문이다 — 되돌릴 대상이 애초에 없다.

    되돌리기를 포기하는 실질 이유도 있다. 라틴 음차는 단어 전체를 보고 규칙을 적용해서
    접두사 출력 길이가 단조 증가하지 않는다(``today``: t→2, to→2, tod→3, toda→2, today→3).
    ja에서 쓴 접두사 재호출 귀속이 여기서는 성립하지 않는다.

    음차가 곧 음절 분리라는 게 이 경로의 요점이다. 영어 철자에는 음절 경계가 없어서
    (``through``는 7글자 1음절) 철자 vocab CTC로는 음절을 못 딴다 — 묵음까지 스팬을 받는다.
    음차하면 ``투데이``/``トデイ``처럼 결과가 이미 음절 단위라 사전이 따로 필요 없다.
    대신 음차 음절 수가 실제 발음과 어긋날 수 있다(``through`` 1음절 → ``스라우`` 3음절).
    """
    if mode == "hangul":
        from everyric2.text.latin_hangul import transliterate_latin

        text = transliterate_latin(source)
    elif mode == "cmu":
        from scripts.bench_adapters.en_g2p import transliterate_cmu

        text = transliterate_cmu(source)
    elif mode == "ipa":
        from scripts.bench_adapters.en_g2p import transliterate_ipa

        text = transliterate_ipa(source)
    else:
        from everyric2.text.ko_reading import latin_to_kana

        text = latin_to_kana(source)
    return text, list(text)


_WORD_RE = re.compile(r"[A-Za-z']+")
# 앞 글자와 한 모라를 이루는 가나 — 요음·촉음·장음·소문자 모음.
_COMBINING_KANA = frozenset("ャュョァィゥェォッーゃゅょぁぃぅぇぉっ")


# 표기와 발음이 갈리는 조사 — 표기형 독음이 이 셋만 틀린다(を 280·は 267·へ 16회).
# 장음(よう→よー)과 달리 바꾼 결과가 vocab에 있으므로 순이득이다.
_JA_JOSHI_READING = {"は": "わ", "を": "お", "へ": "え"}


def _ja_reading_units(
    source: str,
    parse: list[Any] | None = None,
    *,
    phonetic: bool = False,
    joshi_only: bool = False,
) -> tuple[str, list[str]]:
    """일본어 한 줄 → (가나 독음 정렬 타깃, **가나 모라 소유자**).

    ``--input-mode pron-kana``(@kana 레인)와 **같은 결과를 내야 한다.** 그쪽도 같은
    ``tokenize_reading``으로 가사를 가나로 펼쳐 정렬기에 넣는다. 이 함수의 존재 이유는 표기가
    아니라 **심판**이다 — 어댑터가 원문을 들고 있어야 MeCab N-best로 대안 독음을 만들 수 있고,
    input-mode 경로는 가사를 미리 바꿔 넣으므로 어댑터에 원문이 남지 않는다.

    한때 표시를 원문 글자(한자)로 얹어 봤는데 해상도만 나빠졌다(深海少女 세그 634 → 429,
    UST 음절 97.7% → 97.3%). ``見た``를 ``み|た`` 두 칸으로 켜는 것과 ``見|た``로 켜는 것은
    카라오케에서 다르고, 한자 한 글자가 2~3모라를 통째로 덮으면 그만큼 타이밍을 덜 준다.
    그래서 표시도 모라 단위로 되돌렸다 — @kana와 같은 해상도라야 심판만 놓고 비교할 수 있다.
    """
    from everyric2.text.ja_reading import tokenize_reading

    tokens = parse if parse is not None else tokenize_reading(source, phonetic=phonetic)
    chars: list[str] = []
    owners: list[str] = []
    for token in tokens:
        reading = token.reading or token.surface
        if not reading:
            continue
        # 조사만 발음형으로. 품사가 助詞인 한 글자짜리만 바꾼다 — ``は``는 조사일 때만 /wa/이고
        # ``はな``(꽃)의 は는 그대로 /ha/다. 품사를 안 보면 그 구분이 무너진다.
        if joshi_only and len(reading) == 1 and token.pos.startswith("助詞"):
            reading = _JA_JOSHI_READING.get(reading, reading)
        chars.append(reading)
        owners.extend(reading)  # 모라마다 세그 하나
    return "".join(chars), owners


def _ja_prod_tokens(source: str) -> list[list[Any]]:
    """프로드 후보 생성기(``pron_style.candidate_token_sets``)의 토큰 열들. ``[0]``이 기본값.

    **MeCab N-best를 직접 쓰면 안 된다.** 그 구현은 실오디오에서 해로웠다고 기록돼 있다 —
    삭제가 섞이고 어댑터가 못 듣는 표기 변종이 신호를 익사시켰다(``pronunciation_candidates``
    docstring). 프로드는 후보를 ``_AMBIGUOUS_WORDS`` 표의 대립 읽기로 좁혀 8건 맞고 0건
    틀리는 지점까지 갔고, 서버에서 그 구성으로 돌고 있다. 벤치도 같은 것을 써야 비교가 된다.

    부수 효과로 기본값이 ``phonetic=True`` + 루비 채택이 되어 조사 は→わ·を→お까지 반영된다
    (표기형 독음은 ja 토큰의 11.1%를 틀린 음가로 준다).
    """
    from everyric2.text.pron_style import candidate_token_sets

    try:
        _, token_sets = candidate_token_sets(source)
    except Exception:
        logger.warning("ja 후보 생성 실패, 표기형으로 폴백 — %r", source[:24], exc_info=True)
        return []
    return list(token_sets or [])


def _ja_prod_units(source: str) -> tuple[str, list[str]]:
    """프로드 기본 독음(후보 [0])을 가나 타깃으로 — 심판 없이 쓰는 대조군."""
    token_sets = _ja_prod_tokens(source)
    if not token_sets:
        return _ja_reading_units(source, phonetic=True)
    return _ja_reading_units(source, token_sets[0])


def _ja_reading_variants(source: str) -> tuple[_Candidate, list[_Candidate]]:
    """(기본 독음, 대립 읽기 후보들). ja 후보는 **라인 전체** 단위다 — 프로드 심판과 같다."""
    token_sets = _ja_prod_tokens(source)
    if not token_sets:
        text, owners = _ja_reading_units(source, phonetic=True)
        return _Candidate("", text, owners), []
    text, owners = _ja_reading_units(source, token_sets[0])
    base = _Candidate("", text, owners)
    out: list[_Candidate] = []
    seen = {text}
    for rank, tokens in enumerate(token_sets[1:], start=1):
        alt_text, alt_owners = _ja_reading_units(source, tokens)
        if not alt_text or alt_text in seen:
            continue
        seen.add(alt_text)
        out.append(_Candidate(f"cand#{rank}", alt_text, alt_owners))
    return base, out


def _ipa_display_units(
    source: str, display: str, choices: dict[int, int] | None = None
) -> tuple[str, list[str]]:
    """``_ipa_display_full``의 얇은 래퍼 — ``_SCRIPTS``가 (타깃, 소유자) 두 값만 받는다."""
    text, owners, _ = _ipa_display_full(source, display, choices)
    return text, owners


def _ipa_display_full(
    source: str, display: str, choices: dict[int, int] | None = None
) -> tuple[str, list[str], list[tuple[int, int]]]:
    """라틴 한 줄 → (**IPA** 정렬 타깃, 표시 글자 소유권).

    앞의 경로들은 정렬 타깃과 표시가 같은 문자열이었다. 여기서는 **갈라진다**: 정렬은 IPA가
    하고 화면에는 한글(또는 가나)이 나간다. IPA가 정렬에서 압도하지만(무분리 span score
    −8.02 vs 한글 −15.01) 사람에게 ``bjutəfəl``을 보여줄 수는 없기 때문이다.

    가능한 이유는 ``en_g2p.Unit``이 «어느 IPA 조각이 어느 표시 글자에 속하는지»를 음소
    단계에서 이미 들고 있어서다. 유닛의 **첫 IPA 문자**에 표시 글자를 몰아주고 나머지는 빈
    소유자로 둔다 — 빈 소유자는 ``_refine``에서 앞 세그의 끝을 늘리는 데 쓰이므로, 결과적으로
    세그 하나가 음소 여럿을 덮는 **음절 스팬**이 된다(``쉿`` 한 글자 = swit 네 음소).

    OOV는 원문 라틴을 타깃에 그대로 둔다(``transliterate_ipa``와 같은 정책 — IPA 열에 다른
    문자 체계를 섞으면 무엇이 점수를 깎았는지 못 읽는다). 표시만 철자 음차로 채운다.

    ``display="en"``은 **원문 영어 층**이다. 영어 성악 악보가 ``beau-ti-ful``로 음표마다 음절을
    배치하듯 원문도 음절이 단위다 — 낱말로 묶으면 한 낱말 안에서 음높이·박자가 바뀌는 구조를
    통째로 버린다(사용자 지적, 2026-08-01).

    철자에 음절 경계 표시가 없다는 것이 유일한 장애인데, 음절 **수**는 CMU가 주므로 모음
    글자로 그 수만큼 가르면 된다(``en_g2p.syllabify_spelling``, 실제 가사 97.46%). 못 가른
    낱말은 통째로 둔다 — 「덜 쪼갠」 것이지 틀린 것이 아니다.

    음소 정밀도가 여기서 왜 필요한가 하면 **묵음** 때문이다. 철자 CTC는 ``make``의 e에도,
    공백·쉼표에도 스팬을 줘서(프로드 실측: 한 줄에 철자 세그 2,018개, 같은 구간 IPA 음절 749개)
    경계가 실제 소리보다 늦게 끝난다. IPA는 소리에만 스팬을 준다.
    """
    from scripts.bench_adapters.en_g2p import (
        syllabify_unknown,
        syllable_units_for_word,
        units_for_word,
    )

    if display in ("en", "ipa"):
        fallback_display = None  # 표시가 원문 철자(en)이거나 타깃 자신(ipa)이라 음차가 없다
    elif display == "kana":
        from everyric2.text.ko_reading import latin_to_kana as fallback_display
    else:
        from everyric2.text.latin_hangul import transliterate_latin as fallback_display

    chars: list[str] = []
    owners: list[str] = []
    shown_at = -1  # 표시를 마지막으로 얹은 owners 위치

    def emit(target: str, shown: str) -> None:
        """타깃 문자열 ``target``을 넣고 표시 ``shown``을 그 **첫 글자**에 몰아준다."""
        nonlocal shown_at
        if not target:
            return
        chars.append(target)
        shown_at = len(owners)
        owners.extend([shown] + [""] * (len(target) - 1))

    pos = 0
    # 낱말마다 «이 낱말이 타깃 문자열의 어디를 차지하는가». 심판이 그 구간의 점수만 보고
    # 판정하려면 이게 있어야 한다 — 라인 전체 평균으로 보면 긴 라인에서 낱말 하나의 신호가
    # 묻힌다(20음절 라인에서 2음절을 바꾸면 1/10로 희석된다).
    word_spans: list[tuple[int, int]] = []
    matches = list(_WORD_RE.finditer(source))
    words = [m.group(0) for m in matches]
    for word_index, match in enumerate(matches):
        for char in source[pos : match.start()]:  # 낱말 사이 공백·구두점
            emit(char, char)
        word_lo = len(owners)
        word = match.group(0)
        # 어느 사전 발음을 쓸지 — 기본은 첫 번째다. 오디오 심판이 이 자리를 바꿔가며 점수를
        # 비교한다(``_ipa_display_variants``).
        entry = (choices or {}).get(word_index, 0)
        if word.lower() in _CONTEXT_DETERMINED:
            entry = _the_entry(words, word_index)  # 문맥이 정한다 — 심판을 안 거친다
        units = units_for_word(word, entry)
        if display == "en":
            # 원문 철자를 음절 조각으로 갈라 각 조각의 IPA 첫 글자에 얹는다. 나머지는 빈
            # 소유자가 되어 ``_refine``의 merge_orphans가 그 음절 끝까지 스팬을 늘린다 —
            # 한글·가나 모드와 **같은 기계**이고 몰아주는 단위만 다르다.
            syllables = syllable_units_for_word(word, entry)
            if syllables:
                for piece, syl_units in syllables:
                    ipa = "".join(unit.ipa for unit in syl_units)
                    if ipa:
                        emit(ipa, piece)
                    elif shown_at >= 0:
                        owners[shown_at] += piece
            elif units:
                # 사전엔 있는데 철자를 못 갈랐다(가사 출현 2.5% — everything·sidewalk 등).
                # 타깃은 IPA 전체, 표시는 낱말 통째다.
                emit("".join(unit.ipa for unit in units), word)
            else:
                # OOV — 타깃이 원문 철자다. **여기서도 음절로 가른다**: 통째로 두면 그 낱말만
                # 세그 하나가 되어 그 구간에서 카라오케가 멈춘다(weathergirl의 weathervane이
                # 실제로 그랬다 — 한글 레인은 `위 더 베 인`인데 영어만 붙어 있었다).
                for piece in syllabify_unknown(word):
                    emit(piece, piece)
        elif units:
            for unit in units:
                # ``ipa``는 정렬 타깃을 그대로 표시한다 — 음절 단위로 묶인 IPA 전사가 되어
                # (bju|tə|fəl) 「무엇을 맞추고 있는가」를 표기 변환 없이 눈으로 볼 수 있다.
                shown = {"kana": unit.kana, "ipa": unit.ipa}.get(display, unit.hangul)
                # IPA가 빈 유닛은 타깃에 자리가 없어 스팬을 못 받는다. 표시만 앞 글자에 얹는다.
                if unit.ipa:
                    emit(unit.ipa, shown)
                elif shown_at >= 0:
                    owners[shown_at] += shown
        else:
            # OOV(전체 낱말의 0.3% — weathervane·unlaid 같은 복합어·파생어). 타깃은 원문 철자를
            # 그대로 두되, 표시 음차를 **철자 길이에 비례 배분**한다. 통째로 첫 글자에 몰면 그
            # 낱말이 세그 하나가 되어 그 구간만 카라오케가 멈춘다.
            shown = (fallback_display(word) if fallback_display else word) or word
            slots = [""] * len(word)
            cursor = -1
            for k, char in enumerate(shown):
                # 작은 가나·장음은 앞 글자와 **한 모라**다. 문자 단위로 균등 배분하면
                # ``ウィ``가 ``ウ``+``ィ`` 두 세그로 갈라진다.
                if char in _COMBINING_KANA and cursor >= 0:
                    slot = cursor
                else:
                    slot = max(min(k * len(word) // len(shown), len(word) - 1), cursor)
                slots[slot] += char
                cursor = slot
            chars.append(word)
            shown_at = len(owners)
            owners.extend(slots)
        word_spans.append((word_lo, len(owners)))
        pos = match.end()
    for char in source[pos:]:
        emit(char, char)
    return "".join(chars), owners, word_spans


# "kana"는 native의 옛 이름 — 기존 런 캐시의 meta와 호환을 위해 남긴다.
_SCRIPTS = {
    "native": _native_units,
    "kana": _native_units,
    "hangul": _hangul_units,
    "latin-hangul": lambda s: _latin_units(s, "hangul"),
    "latin-kana": lambda s: _latin_units(s, "kana"),
    # 발음 사전(CMU) 경유 — 철자가 아니라 «음소»에서 출발한다.
    "latin-cmu": lambda s: _latin_units(s, "cmu"),
    "latin-ipa": lambda s: _latin_units(s, "ipa"),
    # ★정렬은 IPA, 표시는 한글·가나 — 정렬 타깃과 표시가 갈라지는 유일한 경로.
    "latin-ipa-hangul": lambda s: _ipa_display_units(s, "hangul"),
    "latin-ipa-kana": lambda s: _ipa_display_units(s, "kana"),
    # 같은 IPA 정렬을 **원문 영어 음절** 해상도로 — 악보의 beau-ti-ful 그 단위.
    "latin-ipa-en": lambda s: _ipa_display_units(s, "en"),
    # 타깃을 그대로 표시 — 음절로 묶인 IPA 전사(bju|tə|fəl). 표기 변환을 거치지 않으므로
    # 「정렬기가 실제로 무엇을 맞추고 있는가」가 화면에 그대로 나온다.
    "latin-ipa-phonetic": lambda s: _ipa_display_units(s, "ipa"),
    # ja 가나 독음 — @kana(--input-mode pron-kana)와 같은 타깃이다(둘 다 tokenize_reading 기본값).
    "ja-reading": _ja_reading_units,
    # ★같은 독음을 **발음형**으로. 조사 は→わ·を→お, 장음 よう→よー가 반영된다.
    # ja 토큰의 11.1%가 여기서 갈리는데(を 280회·は 267회, 2026-08-01 실측) 지금까지 벤치는
    # 표기형만 써서 그만큼을 틀린 음가로 맞추고 있었다 — 조사 は를 주면 모델은 /ha/를 찾는다.
    # 프로드(pron_style)는 이미 phonetic=True를 쓴다. 벤치만 안 쓰고 있었다.
    "ja-reading-phon": lambda s: _ja_reading_units(s, phonetic=True),
    # ★조사만 — 발음형의 두 효과를 갈라 재려고 둔다. 장음 ー는 **vocab에 없어서**
    # (omniASR 실측) 발음형이 타깃 커버리지를 95.59% → 92.30%로 떨어뜨린다. 즉 발음형에는
    # 이득(조사 563회)과 손해(장음 ~795회)가 섞여 있어 그대로 재면 상쇄돼 아무것도 못 읽는다.
    "ja-reading-joshi": lambda s: _ja_reading_units(s, joshi_only=True),
    # ★프로드 독음 — 서버가 실제로 쓰는 후보 생성기의 기본값([0])이다. phonetic=True + 루비
    # 채택이라 조사·장음까지 반영된다. 심판 켬/끔이 **이 타깃 위에서** 갈려야 비교가 된다.
    "ja-prod": _ja_prod_units,
    # ★혼합 표기 — 위와 같은 ja 독음에 **라틴 낱말 음절화**와 **장음 펴기**를 얹는다.
    # 정의는 아래(``_mixed_units``)에 있고 여기서는 이름만 건다 — 등록 시점에 이미 정의돼
    # 있어야 하므로 실제 배선은 모듈 끝에서 한다.
}

# 오디오 심판이 붙는 스크립트 → 표시 종류. 심판은 낱말마다 사전의 대체 발음으로 바꿔 보고
# span score가 오르면 그 발음을 채택한다.
_REFEREE_DISPLAY = {
    "latin-ipa-hangul": "hangul",
    "latin-ipa-kana": "kana",
    "latin-ipa-en": "en",
    "latin-ipa-phonetic": "ipa",
}


# ── 오디오 강도 봉우리 — «몇 음절로 불렀는가»를 오디오가 직접 답한다 ──
#
# 정렬 점수(gain)는 «어느 발음이 그럴듯한가»는 답하지만 **음절 수**는 못 답한다. 짧은 타깃이
# 구조적으로 유리해서다(blank 확률이 늘 높다). 그래서 길이가 같은 후보만 견주게 막아 뒀는데,
# 영어 축약은 하필 그 안전장치를 그냥 통과한다 — ``our`` aʊɚ(2음절)와 aʊr(1음절)은 IPA
# 길이가 같고 음절 수만 다르다(ɚ는 음절성 r). 즉 gain이 음절 수를 정하고 있었다.
#
# 대신 오디오를 본다. 음절 경계에서 강도가 한 번 내려갔다 올라오므로 그 골을 세면 된다.
# 청취 판정 22건 실측(2026-08-02): 무조건 적은 쪽 18/22, gain 심판 18/22, **강도 2dB
# 19/22**(gain 오답 4건 중 2건 구제, 1건 파괴), 강도 1dB 19/22(4/4 구제, 3건 파괴).
# 파괴가 적은 2dB를 기본으로 둔다 — 되던 것을 깨는 대가가 안 되던 것을 고치는 이득보다 크다.
#
# CTC 방출의 **모음 사후확률로도 같은 것을 노렸으나 실패했다**: 방출 스파이크는 «모음 글자를
# 냈다»는 표시지 «모음 핵이 울렸다»가 아니라, 이중모음이면 둘·축약이면 하나가 되어 음절 수와
# 체계적으로 어긋난다(최대 16/22로 바닥선 18/22 미달). blank를 분모에서 빼면 곡선이 스파이크
# 대신 고원이 되는데(모음이 울리는 내내 «말한다면 모음»이 참이므로) 고원 개수로 세도 16/22가
# 한계였다. 오디오 강도는 그 텍스트 단계를 안 거치는 것이 요점이다.
_ENERGY_BAND_HZ = (300, 3000)   # 모음 포먼트 대역 — 호흡·베이스 누출과 마찰음 고역을 뺀다
_ENERGY_SMOOTH_FRAMES = 5       # 25ms 이동평균: 성문 주기(≤10ms)는 지우고 음절 골(≥80ms)은 남긴다
_ENERGY_MIN_NUCLEUS_SEC = 0.09  # 이보다 촘촘한 봉우리는 음절이 아니다
_ENERGY_HOP = 160               # 10ms


def _energy_envelope(vocals_path: Path) -> tuple[Any, float] | None:
    """(대역 제한 강도 dB 배열, 프레임 초). 못 만들면 None — 심판은 gain만으로 계속 돈다."""
    try:
        import librosa
        import numpy as np
        from scipy.signal import find_peaks  # noqa: F401  — 여기서 같이 확인해 둔다
    except ImportError:
        logger.warning("librosa/scipy 없음 — 강도 봉우리 없이 정렬 점수 심판만 쓴다")
        return None
    try:
        waveform, sample_rate = librosa.load(str(vocals_path), sr=16_000, mono=True)
        power = np.abs(librosa.stft(waveform, n_fft=512, hop_length=_ENERGY_HOP)) ** 2
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=512)
        lo, hi = _ENERGY_BAND_HZ
        db = 10 * np.log10(power[(freqs >= lo) & (freqs <= hi)].sum(axis=0) + 1e-12)
        kernel = np.ones(_ENERGY_SMOOTH_FRAMES) / _ENERGY_SMOOTH_FRAMES
        return np.convolve(db, kernel, mode="same"), _ENERGY_HOP / 16_000
    except Exception:
        logger.warning("강도 포락선 계산 실패 — 정렬 점수 심판만 쓴다", exc_info=True)
        return None


def _energy_nuclei(envelope, start: float, end: float, dip_db: float) -> int | None:
    """[start, end] 안의 음절 핵 개수. **셀 수 없으면 None**(판단 보류).

    봉우리 0개는 «0음절»이 아니라 **검출 실패**다 — 부른 낱말이 0음절일 수는 없다. 그런데
    0을 그대로 세면 1과의 거리가 2보다 가까워 자동으로 «적은 쪽»에 표를 던지게 되고, 그러면
    오디오를 안 보고 «무조건 짧은 후보»를 고르는 규칙과 구별되지 않는다. 실제로 그렇게 세면
    성적이 부풀려진다(청취 22건 중 13건이 0봉우리였고, 그 13건이 전부 «맞은» 것으로 셈됐다).

    짧은 낱말에서 자주 일어난다 — ``our``은 0.14~0.34초라 그 창 안에서 2dB 낙차를 만들
    여지가 없다. 하필 1↔2음절 물음이 사는 자리라, 이 신호의 실질 적용 범위는 좁다.
    """
    from scipy.signal import find_peaks

    db, frame_sec = envelope
    first = max(int(start / frame_sec) - 1, 0)
    last = min(int(math.ceil(end / frame_sec)) + 2, len(db))
    if last - first < 3:
        return None
    peaks, _ = find_peaks(
        db[first:last],
        prominence=dip_db,
        distance=max(int(_ENERGY_MIN_NUCLEUS_SEC / frame_sec), 1),
    )
    return len(peaks) or None


def _syllables_in(owners: list[str], span: tuple[int, int] | None) -> int | None:
    """후보의 그 낱말이 몇 음절인가 — 표시 글자가 붙은 타깃 문자의 개수."""
    if span is None:
        return None
    lo, hi = span
    return sum(1 for index in range(lo, min(hi, len(owners))) if owners[index])


# ── 혼합 표기 라우팅 ──
#
# ja 경로는 한자·가나만 다루고 **라틴을 글자 단위로 흘려보낸다**. ``numb``이 1음절인데
# ``n|u|m|b`` 네 세그로 나온다 — 카라오케 표시로는 깨진 것이다. ja·mixed 45곡에서 라틴이
# 타깃의 12.4%이고, **5% 넘는 곡이 16곡**이다(numb numb 53.7%, グラス 50.3%,
# DAYBREAK FRONTLINE 80.0%). 절반이 영어인 곡이 여럿이므로 그 곡들은 가사 절반이 깨진다.
#
# 부품은 이미 다 있다 — 라틴은 en 경로의 ``_ipa_display_full``이 CMU 음절로 묶어 준다.
# 여기서는 **문자 계열별로 나눠 각자에게 넘기기만** 한다. 한글은 손대지 않는다: 한글이 음절
# 문자라 글자 하나가 곧 음절이고, omniASR 어휘에도 음절이 그대로 있다(아·스·라·이·해 확인).
_LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")

# 장음 ``ー``가 이어받는 모음. omniASR 어휘에 ``ー``가 없어 그대로 두면 그 모라가 정렬
# 타깃에서 통째로 빠진다(ja 커버리지 95.6% → 92.3% 실측). 앞 가나의 모음으로 편다.
_KANA_VOWEL: dict[str, str] = {}
for _vowel, _row in (
    ("あ", "あかさたなはまやらわがざだばぱゃゎ"),
    ("い", "いきしちにひみりぎじぢびぴ"),
    ("う", "うくすつぬふむゆるぐずづぶぷゅ"),
    ("え", "えけせてねへめれげぜでべぺ"),
    ("お", "おこそとのほもよろをごぞどぼぽょ"),
):
    for _kana in _row:
        _KANA_VOWEL[_kana] = _vowel
        _KANA_VOWEL[chr(ord(_kana) + 0x60)] = chr(ord(_vowel) + 0x60)  # 가타카나도 같이


def _expand_choonpu(text: str, owners: list[str]) -> tuple[str, list[str]]:
    """``ー``를 앞 가나의 모음으로 펴되 **표시 소유자는 비운다**.

    정렬과 표시가 원하는 것이 다르다. CTC는 그 구간에 모음이 이어지는 걸 들으므로 타깃에
    모음이 있어야 맞고(``ー``는 omniASR 어휘에 없어 그 모라가 통째로 빠졌다 — ja 커버리지
    95.6% → 92.3%), 노래방 표시에서 ``しゅう``는 **한 칸**이지 두 칸이 아니다.

    소유자를 비우면 ``_refine``이 앞 세그의 끝을 여기까지 늘린다 — 세그 하나가 늘어난 모라
    전체를 덮는다. 처음엔 표시도 모음으로 채웠는데 세그가 늘면서 정확도가 떨어졌다
    (深海少女 97.85 → 96.55, 熱異常 23.94 → 21.46, 2026-08-02).
    """
    if "ー" not in text and "ｰ" not in text:
        return text, owners
    chars: list[str] = []
    out: list[str] = []
    for index, char in enumerate(text):
        owner = owners[index] if index < len(owners) else ""
        if char in ("ー", "ｰ"):
            vowel = _KANA_VOWEL.get(chars[-1]) if chars else None
            if vowel is None:
                continue  # 앞이 가나가 아니면 버린다 — 어차피 어휘 밖이라 정렬에서 빠진다
            chars.append(vowel)
            out.append("")
            continue
        chars.append(char)
        out.append(owner)
    return "".join(chars), out


def _route_latin(
    text: str, owners: list[str], display: str, choices: dict[int, int] | None = None
) -> tuple[str, list[str]]:
    """타깃 안의 **라틴 낱말**을 en 경로(ASCII 음소 + 음절 표시)로 갈아 끼운다.

    ``choices``는 «라틴 낱말 순번 → CMU 발음 번호»다. 라틴 낱말 심판이 이긴 발음을 여기로
    넘겨 조합 타깃을 만든다 — ja 독음 축과 라틴 축이 서로 다른 시간 구간이라 독립으로 반영된다.
    """
    chars: list[str] = []
    out: list[str] = []
    pos = 0
    for ordinal, match in enumerate(_LATIN_WORD_RE.finditer(text)):
        chars.append(text[pos : match.start()])
        out.extend(owners[pos : match.start()])
        word = match.group(0)
        entry = (choices or {}).get(ordinal, 0)
        word_text, word_owners, _ = _ipa_display_full(word, display, {0: entry} if entry else None)
        if word_text and len(word_owners) == len(word_text):
            chars.append(word_text)
            out.extend(word_owners)
        else:  # CMU에 없는 낱말 — 원문 철자를 그대로 둔다(en 경로와 같은 처리)
            chars.append(word)
            out.extend(owners[match.start() : match.end()])
        pos = match.end()
    chars.append(text[pos:])
    out.extend(owners[pos:])
    return "".join(chars), out


_KANA_RE = re.compile(r"[぀-ゟ゠-ヿ]+")


def _hangul_owners_for_kana(run: str) -> list[str]:
    """가나 한 덩이 → **글자별 한글 소유자**. 받침으로 흡수되는 글자는 빈 소유자다.

    ``kana_to_hangul``을 그대로 재사용한다 — 받침(ん·っ)과 장음 규칙이 이미 거기 있고, 표시가
    프로드와 갈리면 안 되기 때문이다. 글자를 하나씩 늘려 가며 «한글이 몇 글자가 됐는가»를
    보고, 안 늘어난 글자는 앞 글자에 먹힌 것이므로 소유자를 비운다(``しんかい`` = し·ん·か·い
    → 신··카·이). 빈 소유자는 ``_refine``이 앞 세그의 끝을 늘리는 데 쓴다.
    """
    from everyric2.text.kana_hangul import kana_to_hangul

    starts = []
    buffer = ""
    for char in run:
        starts.append(len(kana_to_hangul(buffer)))
        buffer += char
    final = kana_to_hangul(buffer)
    owners = []
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else len(final)
        owners.append(final[start:stop] if stop > start else "")
    return owners


def _route_hangul(text: str, owners: list[str]) -> tuple[str, list[str]]:
    """가나 구간의 **표시만** 한글로 바꾼다 — 정렬 타깃(``text``)은 건드리지 않는다.

    타깃과 표시를 가르는 것이 이 경로의 요점이다. 한글을 타깃으로 쓰면 omniASR이 한글 자모를
    거의 못 내서(실측 0/24) 정렬이 무너지지만, 표시로만 쓰면 정렬은 가나가 하고 화면에는
    한국어 사용자가 읽을 수 있는 글자가 나간다.
    """
    out = list(owners)
    for match in _KANA_RE.finditer(text):
        lo, hi = match.span()
        replaced = _hangul_owners_for_kana(text[lo:hi])
        # 원래 소유자가 비어 있던 자리(장음 등)는 그대로 비워 둔다 — 앞 세그의 몫이다.
        for offset, owner in enumerate(replaced):
            out[lo + offset] = owner if owners[lo + offset] else ""
    return text, out


def _mixed_stage1(
    source: str, parse: list[Any] | None, expand_long: bool
) -> tuple[str, list[str]]:
    """라틴 라우팅 **직전** 상태 — ja 독음 + 장음 펴기. 라틴 낱말은 아직 원문 철자다."""
    text, owners = (
        _ja_prod_units(source) if parse is None else _ja_reading_units(source, parse)
    )
    return _expand_choonpu(text, owners) if expand_long else (text, owners)


def _mixed_units(
    source: str,
    parse: list[Any] | None = None,
    display: str = "en",
    expand_long: bool = True,
    en_choices: dict[int, int] | None = None,
) -> tuple[str, list[str]]:
    """혼합 가사 한 줄 → (정렬 타깃, 표시 소유자). ja 독음 → 장음 펴기 → 라틴 라우팅.

    ``expand_long``은 **축 분리용**이다. 라틴 라우팅과 장음 펴기를 같이 켜면 어느 쪽이 이득을
    냈는지 못 읽는다 — 실제로 처음 측정에서 라틴 없는 곡까지 점수가 떨어져 원인을 착각했다.
    """
    text, owners = _mixed_stage1(source, parse, expand_long)
    if display == "hangul":
        text, owners = _route_hangul(text, owners)
    return _route_latin(text, owners, display, en_choices)


def _mixed_variants(
    source: str, expand_long: bool = True, latin: bool = False, display: str = "en"
) -> tuple[_Candidate, list[_Candidate]]:
    """혼합 경로의 심판 후보 — **두 축**을 함께 낸다.

    * ja 독음 축: 라인 전체 파스 후보(``word_index`` 없음). 파스마다 토큰 경계가 달라 조합이
      안 되므로 이긴 것 하나만 고른다.
    * 라틴 낱말 축: 낱말 하나씩 CMU 대체 발음(``word_index`` = 라틴 낱말 순번). 낱말끼리는
      서로 다른 시간 구간이라 독립 판정 후 한꺼번에 반영한다.

    두 축은 서로 다른 구간을 차지하므로 ``_refine``이 «이긴 파스 + 이긴 낱말들»로 조합
    타깃을 다시 만든다. 라틴 후보는 **기본 파스 위에서** 만든다 — 파스가 바뀌어도 라틴
    구간의 글자는 그대로라 판정이 그대로 옮겨진다.
    """
    from scripts.bench_adapters.en_g2p import pronunciations

    token_sets = _ja_prod_tokens(source)
    parses: list[Any] = list(token_sets) if token_sets else [None]
    base_parse = parses[0]
    base_text, base_owners = _mixed_units(source, base_parse, display, expand_long=expand_long)
    base = _Candidate("", base_text, base_owners, parse=base_parse)
    out: list[_Candidate] = []
    seen = {base_text}

    for rank, tokens in enumerate(parses[1:], start=1):
        alt_text, alt_owners = _mixed_units(source, tokens, display, expand_long=expand_long)
        if not alt_text or alt_text in seen:
            continue
        seen.add(alt_text)
        out.append(_Candidate(f"cand#{rank}", alt_text, alt_owners, parse=tokens))

    if not latin:
        return base, out
    stage1, _ = _mixed_stage1(source, base_parse, expand_long)
    for ordinal, word in enumerate(_LATIN_WORD_RE.findall(stage1)):
        if word.lower() in _CONTEXT_DETERMINED:
            continue  # 문맥이 이미 정한 낱말 — 오디오에 물을 것이 없다
        total = len(pronunciations(word))
        for entry in range(1, min(total, _MAX_ALTERNATES + 1)):
            alt_text, alt_owners = _mixed_units(
                source, base_parse, display, expand_long=expand_long,
                en_choices={ordinal: entry},
            )
            if not alt_text or alt_text in seen:
                continue
            seen.add(alt_text)
            out.append(
                _Candidate(
                    label=f"{word}#{entry}",
                    text=alt_text,
                    owners=alt_owners,
                    word_index=ordinal,
                    entry=entry,
                    parse=base_parse,
                )
            )
    return base, out


# ja 후보를 내는 스크립트. **두 곳에서 봐야 한다** — 후보 생성기(``_referee_variants``)와
# 심판을 켜는 게이트(``_refine``의 ``referee_on``). 한쪽에만 넣으면 심판이 조용히 안 돈다.
_JA_REFEREE_SCRIPTS = frozenset(
    {"ja-reading", "ja-prod", "ja-mixed", "ja-mixed-nolong", "ja-mixed-hangul"}
)

# ``_SCRIPTS``는 위에서 만들어 두고 배선만 여기서 한다 — ``_mixed_units``가 그 아래에 있어서다.
_SCRIPTS["ja-mixed"] = _mixed_units
_SCRIPTS["ja-mixed-nolong"] = lambda s, parse=None: _mixed_units(s, parse, expand_long=False)
# ★한글 표시 — **정렬은 같고 표시만** 한글이다. 한국어 사용자가 일본어 곡을 읽는 층이다.
_SCRIPTS["ja-mixed-hangul"] = lambda s, parse=None: _mixed_units(s, parse, "hangul")


def _referee_variants(
    script: str, source: str, allow_length_change: bool = False, latin_referee: bool = False
):
    """스크립트별 후보 생성. 심판을 지원하지 않는 스크립트면 None.

    en(CMU)은 **낱말 하나씩** 바꾼 후보를 내고, ja(MeCab)는 **라인 전체** 파스를 낸다 —
    사전의 단위가 다르기 때문이다(``_ja_reading_variants`` 참조).
    """
    if script.startswith("ja-mixed"):
        return _mixed_variants(
            source,
            expand_long=not script.endswith("-nolong"),
            latin=latin_referee,
            display="hangul" if script.endswith("-hangul") else "en",
        )
    if script in _JA_REFEREE_SCRIPTS:
        return _ja_reading_variants(source)
    display = _REFEREE_DISPLAY.get(script)
    if not display:
        return None
    return _ipa_display_variants(source, display, allow_length_change)
# 한 낱말에서 시험할 대체 발음 수 상한. CMU에 5개까지 있는 낱말이 있는데 뒤로 갈수록 희귀
# 이형이라(지명·강세 변이) 값을 못 하면서 정렬 횟수만 늘린다.
_MAX_ALTERNATES = 2

# **문맥이 정하는 낱말은 심판에 맡기지 않는다.** ``the``의 ðə/ði는 모호한 게 아니라 다음
# 낱말의 첫소리로 결정된다(모음 앞 ði, 자음 앞 ðə). 사전이 두 발음을 들고 있는 건 문맥
# 정보가 없어서일 뿐인데, 심판에 넘기면 자음 앞에서도 ði를 고른다 — weathergirl에서 `디
# 웨더`·`디 눗`·`디 선`처럼 전부 자음 앞이었고 사용자 청취로 오류 확정(2026-08-01).
_CONTEXT_DETERMINED = frozenset({"the"})
_ARPABET_VOWEL_PREFIX = frozenset("AEIOU")


def _the_entry(words: list[str], index: int) -> int:
    """``the``가 쓸 사전 발음 번호 — 다음 낱말이 모음으로 시작하면 ði, 아니면 ðə."""
    from scripts.bench_adapters.en_g2p import pronunciations

    following = words[index + 1] if index + 1 < len(words) else ""
    phones = pronunciations(following)
    if not phones or not phones[0]:
        return 0
    if phones[0][0][:1] not in _ARPABET_VOWEL_PREFIX:
        return 0
    # 모음 앞 — ði를 든 후보를 찾는다(CMU 표제어마다 순서가 달라 문자열로 고른다).
    entries = pronunciations("the")
    for position, entry in enumerate(entries):
        if entry and entry[-1].startswith("IY"):
            return position
    return 0


@dataclass
class _Candidate:
    """심판이 견줄 후보 하나. ``word_index``가 None이면 기본(사전 첫 발음) 후보다."""

    label: str
    text: str
    owners: list[str]
    word_index: int | None = None
    entry: int = 0
    # 이 후보에서 **그 낱말이 차지하는 타깃 문자 범위**. 심판은 이 구간의 점수만 본다.
    char_span: tuple[int, int] | None = None
    # 기본 후보에서 같은 낱말의 구간. 음절 수가 바뀌면 길이도 달라지므로 양쪽이 다 필요하다.
    base_span: tuple[int, int] | None = None
    # 혼합 경로에서 이 후보가 쓴 **ja 파스**. 라틴 낱말 심판과 조합할 때 필요하다 — 조합
    # 타깃을 다시 만들려면 «어느 독음 위에서» 낱말을 바꿨는지 알아야 한다.
    parse: Any = None
    # ``_refine``이 나중에 채워 넣는다 — 타깃 토큰 열과 «표시 글자 → 토큰 범위» 대응.
    tokens: list[int] = field(default_factory=list)
    ranges: list[Any] = field(default_factory=list)


def _ipa_display_variants(
    source: str, display: str, allow_length_change: bool = False
) -> tuple[_Candidate, list[_Candidate]]:
    """(기본 후보, 대체 후보들). 대체는 **낱말 하나씩만** 바꾼다.

    조합(2^k)은 만들지 않는다. 낱말끼리는 서로 다른 시간 구간을 차지하므로 독립으로 보고,
    이긴 것들을 나중에 **한꺼번에** 반영한다(``_refine``) — 그래서 라인당 정렬 횟수가 후보
    수에 선형으로만 늘고도 여러 낱말을 동시에 고칠 수 있다.
    """
    from scripts.bench_adapters.en_g2p import pronunciations

    text, owners, spans = _ipa_display_full(source, display)
    base = _Candidate("", text, owners)
    out: list[_Candidate] = []
    for index, match in enumerate(_WORD_RE.finditer(source)):
        word = match.group(0)
        if word.lower() in _CONTEXT_DETERMINED:
            continue  # 문맥이 이미 정한 낱말 — 오디오에 물을 것이 없다
        total = len(pronunciations(word))
        for entry in range(1, min(total, _MAX_ALTERNATES + 1)):
            alt_text, alt_owners, alt_spans = _ipa_display_full(source, display, {index: entry})
            if alt_text == text or index >= len(alt_spans) or index >= len(spans):
                continue
            # **길이가 같은 후보만 견준다** — 단 이 제약의 근거는 재검증이 필요했다.
            #
            # 원래 근거(2026-08-01): 타깃이 짧을수록 총 로그우도가 구조적으로 올라가고, 정규화를
            # 바꿔도 기울기가 안 없어졌다(길어지는 선택 15.3% → 3.8% → 0%). 그런데 그 실측은
            # **타깃 문자의 30.2%가 죽은 토큰이던 상태**에서 잰 값이다(``en_g2p`` 주석 참고).
            # 죽은 토큰은 어떤 프레임에서도 확률이 0이라 «길어질수록 손해»를 만드는 주범이었고,
            # 그 편향이 곧 길이 편향으로 보였을 가능성이 크다. ASCII 음소 전사로 옮겨 죽은
            # 토큰이 2.4%(문장부호뿐)로 떨어졌으므로 **다시 재야 한다** — ``allow_length_change``.
            if not allow_length_change and len(alt_text) != len(text):
                continue
            out.append(
                _Candidate(
                    label=f"{word}#{entry}",
                    text=alt_text,
                    owners=alt_owners,
                    word_index=index,
                    entry=entry,
                    char_span=alt_spans[index],
                    base_span=spans[index],
                )
            )
    return base, out


class TwoPassAligner(AlignerAdapter):
    """앵커 라인 창 안에서만 경량 CTC 모델로 음절을 재정렬하는 어댑터."""

    name: str = ""
    config: TwoPassConfig
    window_pad_sec: float = WINDOW_PAD_SEC
    # 오디오 심판이 사전 발음을 뒤집으려면 이만큼(프레임당 평균 로그확률, nats) 이겨야 한다.
    # 0이면 측정 노이즈로도 뒤집힌다. 값은 프로드가 실오디오로 보정한 것을 그대로 쓴다
    # (``AlignmentSettings.pron_referee_margin``, 2026-07-26): 맞는 후보가 이긴 최소 폭이
    # +0.0375, 틀린 후보가 진 최대 폭이 −0.056이라 그 사이에 0.03이 놓인다. 원래 값 0.15는
    # 추정치였고 실측에서 «맞는 후보 다섯 개가 전부 탈락»해 기각됐다.
    referee_margin: float = 0.03
    # 음절 골로 인정할 최소 강도 낙차(dB). 청취 22건에서 2dB가 «되던 것 파괴» 1건으로 가장
    # 얌전했고, 1dB는 gain 오답 4건을 전부 고치는 대신 3건을 깼다(2026-08-02).
    energy_dip_db: float = 2.0
    # 세그를 늘일 수 있는 최대 길이(초). UST 노트 15,503개에서 99.5퍼센타일 1.111s이고
    # 1.5s 초과는 0.29%뿐이라, 그보다 긴 공백은 늘임음이 아니라 쉼으로 본다.
    seg_hold_max_sec: float = 1.5

    def __init__(self, config: TwoPassConfig | None = None) -> None:
        if config is None:
            config = self.config
        self.config = config
        self.name = config.name
        self._anchor: Any | None = None
        self._refiner: Any | None = None

    def align(self, vocals_path: Path, lyrics: str, language: str) -> AlignOut:
        from scripts.bench_adapters.hf_ctc import _split_lyrics

        source_lines = _split_lyrics(lyrics)
        if not source_lines:
            raise ValueError("lyrics produced zero non-empty lines")

        started = time.perf_counter()
        anchor = self._ensure_anchor()
        anchor_out = anchor.align(vocals_path, lyrics, language)
        if len(anchor_out.lines) != len(source_lines):
            raise RuntimeError(
                f"{self.name} 앵커가 {len(source_lines)}줄 입력에 "
                f"{len(anchor_out.lines)}줄을 돌려줬다"
            )

        refiner = self._ensure_refiner()
        # 경량 모델 적재를 계측 **밖으로** 뺀다. 인스턴스는 스윕 내내 재사용되므로 로드는
        # 첫 곡 한 번뿐인데, 그게 emission 시간에 섞이면 「첫 곡만 6배 느린」 표가 나온다.
        # 프로드는 모델을 상주시키므로 이 비용 자체가 없다 — MMSBaselineAligner와 같은 처리다.
        refiner_load_sec = None
        if getattr(refiner, "_model", None) is None:
            load_started = time.perf_counter()
            refiner._ensure_model()
            _cuda_sync()
            refiner_load_sec = round(time.perf_counter() - load_started, 3)

        # 강도 포락선은 CPU 작업이라 VRAM 계측 밖에서 만든다 — 안에서 재면 카드가 노는
        # 시간이 refine_sec에 섞인다.
        envelope_started = time.perf_counter()
        envelope = (
            _energy_envelope(vocals_path)
            if self.config.referee and self.config.energy_syllables
            else None
        )
        envelope_sec = round(time.perf_counter() - envelope_started, 3)

        with VramProbe() as probe:
            # 두 구간을 따로 잰다. CUDA는 비동기라 동기화 없이 재면 커널 대기 시간이 다음
            # 구간으로 넘어가 「어디가 느린가」를 못 읽는다.
            emission_started = time.perf_counter()
            emission = refiner.emission_for(vocals_path)
            _cuda_sync()
            refine_started = time.perf_counter()
            stats = self._refine(anchor_out.lines, source_lines, refiner, emission, envelope)
            _cuda_sync()
            stats["emission_sec"] = round(refine_started - emission_started, 3)
            stats["refine_sec"] = round(time.perf_counter() - refine_started, 3)
            if self.config.energy_syllables:
                stats["energy_envelope_sec"] = envelope_sec

        elapsed = time.perf_counter() - started
        return AlignOut(
            lines=anchor_out.lines,
            elapsed_sec=round(elapsed, 2),
            # 두 패스는 순차이고 앵커는 별도 프로세스라 카드가 동시에 지는 부담은 최댓값이다.
            vram_peak_mb=_max_or_none(anchor_out.vram_peak_mb, probe.process_peak_mb),
            vram_device_peak_mb=_max_or_none(
                anchor_out.vram_device_peak_mb, probe.device_peak_mb
            ),
            # 라인 경계·신뢰도는 앵커 것이므로 품질 점수도 앵커 것을 그대로 쓴다.
            quality_score=anchor_out.quality_score,
            meta={
                "adapter": self.name,
                "model": f"{self.config.anchor} + {self.config.refiner}",
                "language": anchor_out.meta.get("language"),
                "vocab_unit": "2pass: 앵커 라인 경계 + 경량 모델 음절",
                "audio_sec": round(emission.audio_sec, 3),
                "sample_rate": 16_000,
                "preprocessing": (
                    f"{self.config.note}; 라인 창 ±{self.window_pad_sec}s 안에서만 재정렬, "
                    "실패한 라인은 앵커 세그 유지"
                ),
                "two_pass": {
                    "anchor": self.config.anchor,
                    "refiner": self.config.refiner,
                    "refiner_script": self.config.refiner_script,
                    "refiner_frame_sec": round(emission.frame_sec, 6),
                    "refiner_chunks": emission.chunks,
                    "window_pad_sec": self.window_pad_sec,
                    "referee_margin": self.referee_margin if self.config.referee else None,
                    "refiner_load_sec": refiner_load_sec,
                    **stats,
                },
                "anchor_meta": anchor_out.meta,
                "anchor_elapsed_sec": anchor_out.elapsed_sec,
                "anchor_vram_peak_mb": anchor_out.vram_peak_mb,
                "refiner_vram_peak_mb": probe.process_peak_mb,
                "refiner_vram_device_peak_mb": probe.device_peak_mb,
                "quality": anchor_out.meta.get("quality"),
            },
        )

    def _refine(
        self,
        lines: list[dict[str, Any]],
        source_lines: list[str],
        refiner: Any,
        emission: Any,
        envelope: tuple[Any, float] | None = None,
    ) -> dict[str, Any]:
        """라인마다 창 안에서 재정렬하고 ``segs``만 갈아 끼운다. 실패한 라인은 손대지 않는다."""

        import torch
        import torchaudio.functional as functional

        units = _SCRIPTS[self.config.refiner_script]
        vocab_width = int(emission.emission.shape[-1])
        total_frames = int(emission.emission.shape[1])
        frame_sec = emission.frame_sec
        device = emission.emission.device

        refined = 0
        stretched = 0
        spread = 0
        # 뭉침을 펼 때 쓸 프레임별 «지금 소리가 나고 있을 확률» — 방출에서 바로 얻는다.
        # VAD를 따로 돌릴 필요가 없고, 정렬이 본 것과 **같은 신호**라 판단이 어긋나지 않는다.
        presence = None
        if self.config.spread_piles:
            try:
                presence = (
                    (1 - emission.emission[0][:, emission.blank_id].exp()).float().cpu().numpy()
                )
            except Exception:
                logger.warning("%s: presence 계산 실패, 뭉침 펴기 생략", self.name, exc_info=True)
        converted = 0
        # 경량 모델이 자기 타깃을 얼마나 확신하는지. 라인 confidence는 **앵커** 값이라
        # 표기를 바꿔도 안 움직인다 — 표기 적합도(가나 vs 한글 vs IPA)를 비교하려면
        # 2패스 DP가 실제로 받은 점수를 따로 남겨야 한다.
        span_scores: list[float] = []
        fallbacks: dict[str, int] = {}
        # 표시 글자가 없는 타깃 문자를 앞 세그에 흡수할지. IPA 표시 경로만 그런 문자를
        # 의도적으로 만든다(음절의 둘째 음소 이후). 다른 경로에서 owner가 비는 것은 음차
        # 길이가 어긋난 사고에 가까우므로 기존대로 버린다 — 기존 실측값을 건드리지 않는다.
        # ja-reading도 같은 구조다 — 한자 한 글자가 가나 여럿을 덮으므로 뒤따르는 가나는
        # 소유자가 비고, 그 시간은 앞 글자(한자)의 몫이다.
        # ja-mixed도 같다 — 라틴 낱말은 첫 글자가 음절 전체를 소유하고(``numb`` = n·u·m 세
        # 토큰에 표시 하나), 늘어난 장음도 소유자가 빈다. 안 걸면 세그가 첫 음소 길이로 잘린다.
        merge_orphans = (
            self.config.refiner_script.startswith("latin-ipa-")
            or self.config.refiner_script.startswith("ja-mixed")
            or self.config.refiner_script == "ja-reading"
        )

        def skip(reason: str) -> None:
            fallbacks[reason] = fallbacks.get(reason, 0) + 1

        # 오디오 심판 — 켜져 있고 이 스크립트가 후보를 낼 수 있을 때만.
        script = self.config.refiner_script
        referee_on = bool(self.config.referee) and (
            script in _JA_REFEREE_SCRIPTS or script in _REFEREE_DISPLAY
        )
        referee_display = _REFEREE_DISPLAY.get(script) if referee_on else None
        referee_stats = {"lines": 0, "candidates": 0, "switched": 0}
        # 강도 봉우리가 «의견을 낸 횟수»와 «정렬 점수를 뒤집은 횟수». 뒤집은 적이 없으면
        # 두 신호가 같은 말을 한다는 뜻이라 이 장치를 켤 이유가 없다 — 채택 판단의 핵심 수치다.
        energy_stats = {"decided": 0, "overrode": 0}
        length_moves = {"shorter": 0, "longer": 0}
        if envelope is None and self.config.energy_syllables:
            logger.warning("%s: 강도 포락선이 없어 정렬 점수만으로 심판한다", self.name)
        # 어느 낱말을 어느 발음으로 뒤집었는지. span score는 «후보 중 최고»를 고르므로 오르는
        # 것이 당연하다(순환 논리) — 심판이 «옳은» 발음을 골랐는지는 이 목록을 사람이 봐야 안다.
        referee_picks: dict[str, int] = {}
        # 판정별 gain(프레임당 평균 로그확률 이득)과 시각. **마진 보정의 유일한 재료**다 —
        # 사람이 맞다/틀리다를 판정한 지점의 gain을 모아 그 사이에 문턱을 놓는다. 프로드가
        # 0.15 → 0.03으로 내린 것도 같은 방법이었다(맞은 최소 +0.0375, 틀린 최대 −0.056).
        referee_log: list[dict[str, Any]] = []

        # ── 1패스: 라인별 타깃을 먼저 전부 준비한다 (vocab 압축에 전체 토큰 집합이 필요) ──
        # 심판이 켜지면 라인마다 **후보 여럿**을 준비한다. 압축 열은 곡 전체에서 한 번 정하므로
        # 후보 토큰까지 여기서 모아 둬야 나중에 열이 없어 못 고르는 일이 없다.
        prepared: list[tuple[dict[str, Any], str, _Candidate, list[_Candidate]] | None] = []
        used: set[int] = set()

        def compile_candidate(candidate: _Candidate) -> bool:
            """타깃을 토큰으로 옮기고 vocab에 등록. 못 쓰면 False."""
            token_ids, ranges = refiner.prepare_line_targets(candidate.text)
            if not token_ids or max(token_ids) >= vocab_width:
                return False
            used.update(token_ids)
            candidate.tokens = token_ids
            candidate.ranges = ranges
            return True

        for line, source in zip(lines, source_lines):
            made = (
                _referee_variants(
                    script,
                    source,
                    self.config.allow_length_change,
                    self.config.latin_referee,
                )
                if referee_on
                else None
            )
            if made is not None:
                base, alternates = made
            else:
                text, owners = units(source)
                base, alternates = _Candidate("", text, owners), []
            if not base.text:
                skip("empty_refiner_text")
                prepared.append(None)
                continue
            # 표기 변환이 «실제로» 일어난 줄만 센다. hangul 경로도 입력이 이미 한글이면
            # 항등이므로(ko 곡), 설정값만 보고 "음차했다"고 표시하면 거짓이 된다.
            converted += base.text != source
            if not compile_candidate(base):
                skip("no_in_vocab_chars")
                prepared.append(None)
                continue
            alternates = [c for c in alternates if compile_candidate(c)]
            if alternates:
                referee_stats["lines"] += 1
                referee_stats["candidates"] += len(alternates)
            prepared.append((line, source, base, alternates))

        if not used:
            return _refine_stats(refined, converted, len(lines), fallbacks)

        # ── vocab 압축 — 곡 하나가 실제로 참조하는 열만 남긴다 ──
        # DP가 보는 열은 blank + 타깃 토큰뿐인데 emission 폭은 vocab 전체다(omniASR 9,812).
        # 라인마다 그 전체 폭을 슬라이스·복사하면 복사 비용이 DP 자체를 압도한다 — 단독 정렬
        # 1.5초짜리 모델이 2패스에서 7초를 먹던 원인이 이것이었다. owsm 워커가 이미 쓰는
        # 최적화와 같은 수법이다(owsm_ctc.py의 compact_tokens).
        blank = emission.blank_id
        columns = [blank] + sorted(t for t in used if t != blank)
        column_of = {token: position for position, token in enumerate(columns)}
        source_emission = emission.emission
        compact = torch.index_select(
            source_emission,
            2,
            torch.tensor(columns, dtype=torch.long, device=source_emission.device),
        ).contiguous()
        device = compact.device

        for item in prepared:
            if item is None:
                continue
            line, source, base, alternates = item
            start = float(line.get("start") or 0.0)
            end = float(line.get("end") or 0.0)
            if end <= start:
                skip("no_anchor_window")
                continue
            first = max(0, int((start - self.window_pad_sec) / frame_sec))
            last = min(total_frames, int(math.ceil((end + self.window_pad_sec) / frame_sec)))
            # 압축 후 blank는 언제나 0번 열이다.
            window = compact[:, first:last, :].contiguous()
            failure = ""

            def align(candidate: _Candidate) -> tuple[list[Any], Any] | None:
                """후보를 창 안에서 정렬 → (토큰 스팬, **프레임별 점수**). 실패하면 None.

                프레임별 점수를 함께 돌려주는 것이 요점이다 — 심판이 blank를 포함한 창
                전체로 정규화하려면 토큰 스팬만으로는 부족하다(``_frame_score`` 참조).
                """
                nonlocal failure
                tokens = candidate.tokens
                # CTC 경로는 토큰마다 최소 한 프레임이 필요하고, 같은 토큰이 연달아 오면 그 사이에
                # blank 프레임이 하나 더 든다. 창이 그보다 짧으면 DP에 경로가 아예 없다.
                repeats = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
                if last - first < len(tokens) + repeats:
                    failure = failure or "window_shorter_than_targets"
                    return None
                targets = torch.tensor(
                    [[column_of[t] for t in tokens]], dtype=torch.int32, device=device
                )
                try:
                    aligned, scores = functional.forced_align(window, targets, blank=0)
                    merged = functional.merge_tokens(aligned[0], scores[0], blank=0)
                except Exception as exc:
                    logger.warning("%s: 라인 재정렬 실패, 앵커 세그 유지 — %r", self.name, exc)
                    failure = failure or "forced_align_failed"
                    return None
                if len(merged) != len(tokens):
                    failure = failure or "span_count_mismatch"
                    return None
                return merged, scores[0]

            aligned_base = align(base)
            if aligned_base is None:
                skip(failure or "forced_align_failed")
                continue
            spans, base_frames = aligned_base
            owners, ranges = base.owners, base.ranges
            offset = first * frame_sec

            def word_window(candidate: _Candidate) -> tuple[float, float] | None:
                """기본 정렬에서 그 낱말이 차지한 시간 구간 — 강도 봉우리를 셀 창.

                후보 자신의 정렬이 아니라 **기본 정렬**을 쓴다. 후보마다 창이 달라지면 세는
                구간이 후보에 따라 흔들려 비교가 성립하지 않는다.
                """
                if candidate.base_span is None:
                    return None
                lo, hi = candidate.base_span
                token_lo = token_hi = None
                for index in range(lo, min(hi, len(ranges))):
                    token_range = ranges[index]
                    if token_range is None:
                        continue
                    if token_lo is None:
                        token_lo = token_range[0]
                    token_hi = token_range[1]
                if token_lo is None or token_hi is None or token_hi > len(spans):
                    return None
                return (
                    offset + float(spans[token_lo].start) * frame_sec,
                    offset + float(spans[token_hi - 1].end) * frame_sec,
                )

            # ── 심판: 낱말마다 «그 구간만» 견준다 ──
            # 라인 전체 평균으로 보면 긴 라인에서 낱말 하나의 신호가 희석된다(20음절 라인에서
            # 2음절을 바꾸면 1/10). 낱말끼리는 서로 다른 시간 구간을 차지하므로 독립으로
            # 판정하고, 이긴 것을 **한꺼번에** 반영해 마지막에 한 번 더 정렬한다.
            winners: dict[int, int] = {}
            # 라인 전체 후보(ja)는 조합할 수 없다 — 파스마다 토큰 경계가 달라서다. 이긴 것
            # **하나만** 고른다(프로드 심판과 같은 방식).
            line_best: tuple[float, _Candidate, list[Any]] | None = None
            for candidate in alternates:
                aligned_alt = align(candidate)
                if aligned_alt is None:
                    continue
                alt_spans, alt_frames = aligned_alt
                # 라인 창 **전체**로 견준다 — 낱말 구간만 보면 그 낱말에서 번 이득을 옆
                # 낱말에서 치르는 대가가 계산에서 빠진다(``_window_score`` ②).
                before = _window_score(base_frames)
                after = _window_score(alt_frames)
                if before is None or after is None:
                    continue
                gain = after - before

                # ── 음절 수는 오디오 강도가 정한다 ──
                # 후보끼리 음절 수가 다를 때만 의견을 낸다. 같으면(ja 독음처럼 «어느 글자냐»만
                # 갈릴 때) 봉우리는 답할 것이 없고 정렬 점수가 그대로 결정한다.
                energy_vote: str | None = None
                counted: int | None = None
                syl_base = syl_alt = None
                if envelope is not None and candidate.word_index is not None:
                    syl_base = _syllables_in(base.owners, candidate.base_span)
                    syl_alt = _syllables_in(candidate.owners, candidate.char_span)
                    window_sec = word_window(candidate)
                    if (
                        syl_base is not None
                        and syl_alt is not None
                        and syl_base != syl_alt
                        and window_sec is not None
                    ):
                        counted = _energy_nuclei(envelope, *window_sec, self.energy_dip_db)
                        if counted is not None:
                            near_alt, near_base = abs(counted - syl_alt), abs(counted - syl_base)
                            if near_alt != near_base:
                                energy_vote = "alt" if near_alt < near_base else "base"
                                energy_stats["decided"] += 1

                if energy_vote == "alt":
                    adopted = True          # 오디오가 대체 후보의 음절 수를 지지 — gain 무관
                elif energy_vote == "base":
                    adopted = False         # 오디오가 기본을 지지 — gain이 이겨도 안 바꾼다
                else:
                    adopted = gain >= self.referee_margin
                if energy_vote is not None and adopted != (gain >= self.referee_margin):
                    energy_stats["overrode"] += 1

                # 탈락한 후보의 gain도 남긴다 — 문턱을 어디에 놓아야 하는지는 «넘은 것»과
                # «못 넘은 것»을 함께 봐야 정할 수 있다.
                # 길이 변화는 **편향 감시용**이다. 채택이 한 방향(짧아짐)으로만 쏠리면 그건
                # 오디오 판정이 아니라 구조적 편향이라는 뜻이므로, 방향을 기록해 둔다.
                delta = len(candidate.text) - len(base.text)
                if adopted and delta:
                    length_moves["shorter" if delta < 0 else "longer"] += 1
                referee_log.append({
                    "at": round(start, 2),
                    "word": candidate.label,
                    "gain": round(gain, 5),
                    "adopted": adopted,
                    **({"dlen": delta} if delta else {}),
                    **({"nuclei": counted, "syl": [syl_base, syl_alt], "energy": energy_vote}
                       if energy_vote is not None else {}),
                })
                if not adopted:
                    continue
                if candidate.word_index is None:
                    if line_best is None or gain > line_best[0]:
                        line_best = (gain, candidate, alt_spans)
                    continue
                winners[candidate.word_index] = candidate.entry
                referee_picks[candidate.label] = referee_picks.get(candidate.label, 0) + 1
            if script.startswith("ja-mixed") and (line_best is not None or winners):
                # 혼합 경로는 **두 축이 함께** 이길 수 있다 — ja 독음(라인 전체)과 라틴 낱말.
                # 서로 다른 시간 구간이므로 이긴 파스 위에 이긴 낱말들을 얹어 다시 만든다.
                chosen_parse = line_best[1].parse if line_best is not None else base.parse
                text, merged_owners = _mixed_units(
                    source,
                    chosen_parse,
                    expand_long=(script == "ja-mixed"),
                    en_choices=winners or None,
                )
                combined = _Candidate("", text, merged_owners)
                if compile_candidate(combined) and all(t in column_of for t in combined.tokens):
                    aligned_combined = align(combined)
                    if aligned_combined is not None:
                        spans, _ = aligned_combined
                        owners, ranges = combined.owners, combined.ranges
                        referee_stats["switched"] += 1
                        if line_best is not None:
                            label = line_best[1].label
                            referee_picks[label] = referee_picks.get(label, 0) + 1
            elif line_best is not None:
                _, chosen, spans = line_best
                owners, ranges = chosen.owners, chosen.ranges
                referee_picks[chosen.label] = referee_picks.get(chosen.label, 0) + 1
                referee_stats["switched"] += 1
            elif winners:
                text, merged_owners, _ = _ipa_display_full(source, referee_display, winners)
                combined = _Candidate("", text, merged_owners)
                # 조합 타깃의 토큰은 각 후보 토큰의 부분집합이라 압축 열에 이미 다 들어 있다.
                if compile_candidate(combined) and all(t in column_of for t in combined.tokens):
                    aligned_combined = align(combined)
                    if aligned_combined is not None:
                        spans, _ = aligned_combined
                        owners, ranges = combined.owners, combined.ranges
                        referee_stats["switched"] += 1
            span_scores.extend(float(span.score) for span in spans)

            segs: list[dict[str, Any]] = []
            for index, token_range in enumerate(ranges):
                if token_range is None:
                    continue
                owner = owners[index] if index < len(owners) else ""
                lo, hi = token_range
                if not owner:
                    # 표시 글자가 없는 타깃 문자 = IPA 음절의 둘째 음소 이후(``쉿`` = s·w·i·t의
                    # w·i·t). 그 시간은 앞 표시 글자의 몫이므로 앞 세그의 끝을 여기까지 늘린다.
                    # 그래야 세그 하나가 **음절 전체**를 덮는다.
                    if merge_orphans and segs:
                        segs[-1]["end"] = round(offset + float(spans[hi - 1].end) * frame_sec, 3)
                    continue
                segs.append(
                    {
                        "t": owner,
                        "start": round(offset + float(spans[lo].start) * frame_sec, 3),
                        "end": round(offset + float(spans[hi - 1].end) * frame_sec, 3),
                    }
                )
            if not segs:
                skip("no_segments_produced")
                continue
            if self.config.extend_segments:
                stretched += _extend_segments(segs, line["end"], self.seg_hold_max_sec)
            line["segs"] = segs
            line.setdefault("meta", {})["refined_by"] = self.config.refiner
            refined += 1

        stats = _refine_stats(refined, converted, len(lines), fallbacks)
        if self.config.respace_repeats:
            stats["repeats_respaced"] = _respace_repeated_lines(lines, source_lines)
        stats["boundary_fixes"] = _enforce_monotonic(lines)
        # 뭉침 펴기는 **단조 보정 뒤에** 돈다. 뭉침을 만드는 것이 그 보정 자신이기 때문이다 —
        # 라인 창이 심하게 겹치면 «완전 역전» 처리가 앞뒤 세그를 같은 시각으로 눌러 버린다
        # (熱異常 boundary_fixes 104건 = 뭉친 세그 102개로 일치). 앞에서 돌리면 볼 것이 없다.
        if self.config.spread_piles and presence is not None:
            for line in lines:
                line_segs = line.get("segs") or []
                moved = _spread_piled_segments(line_segs, presence, frame_sec)
                spread += moved
                if moved and self.config.extend_segments:
                    _extend_segments(line_segs, line["end"], self.seg_hold_max_sec)
        # 뭉침 펴기는 **단조 보정 뒤에** 돈다. 뭉침을 만드는 것이 그 보정이기 때문이다 —
        # 라인 창이 심하게 겹치면 «완전 역전» 처리가 앞뒤 세그를 같은 시각으로 눌러 버린다
        # (熱異常 boundary_fixes 104건 = 뭉친 세그 102개). 앞에서 돌리면 볼 것이 없다.
        if self.config.spread_piles and presence is not None:
            for line in lines:
                line_segs = line.get("segs") or []
                spread += _spread_piled_segments(line_segs, presence, frame_sec)
                if self.config.extend_segments and spread:
                    _extend_segments(line_segs, line["end"], self.seg_hold_max_sec)
        if self.config.spread_piles:
            stats["segments_spread"] = spread
        if self.config.extend_segments:
            stats["segments_stretched"] = stretched
            stats["seg_hold_max_sec"] = self.seg_hold_max_sec
        stats["compact_vocab_size"] = len(columns)
        stats["full_vocab_size"] = vocab_width
        if span_scores:
            ordered = sorted(span_scores)
            stats["refiner_span_score_median"] = round(statistics.median(ordered), 4)
            stats["refiner_span_score_mean"] = round(statistics.fmean(ordered), 4)
            stats["refiner_span_count"] = len(ordered)
        if referee_on:
            # 심판이 «얼마나 자주 사전을 뒤집었는가». 0에 가까우면 사전 첫 발음이 이미 맞다는
            # 뜻이라 심판 비용을 지불할 이유가 없다 — 채택 판단의 핵심 수치다.
            stats["referee_lines_with_choice"] = referee_stats["lines"]
            stats["referee_candidates"] = referee_stats["candidates"]
            stats["referee_switched_lines"] = referee_stats["switched"]
            stats["referee_picks"] = dict(
                sorted(referee_picks.items(), key=lambda kv: -kv[1])[:40]
            )
            # 시각 순으로 — 사람이 뷰어에서 그 지점을 듣고 판정과 대조하는 순서 그대로다.
            stats["referee_log"] = sorted(referee_log, key=lambda r: r["at"])
            stats["referee_adopted_shorter"] = length_moves["shorter"]
            stats["referee_adopted_longer"] = length_moves["longer"]
        if self.config.energy_syllables:
            stats["energy_dip_db"] = self.energy_dip_db
            stats["energy_decided"] = energy_stats["decided"]
            stats["energy_overrode_score"] = energy_stats["overrode"]
            stats["energy_envelope"] = envelope is not None
        return stats

    def _ensure_anchor(self) -> Any:
        if self._anchor is None:
            self._anchor = _resolve_adapter(self.config.anchor)
        return self._anchor

    def _ensure_refiner(self) -> Any:
        if self._refiner is None:
            refiner = _resolve_adapter(self.config.refiner)
            if not hasattr(refiner, "emission_for"):
                raise TypeError(
                    f"{self.config.refiner}는 emission_for를 제공하지 않아 2패스 경량 모델로 "
                    "쓸 수 없다 (hf_ctc 후보여야 한다)"
                )
            self._refiner = refiner
        return self._refiner


def _window_score(frame_scores: Any) -> float | None:
    """**창 전체**의 프레임당 평균 로그확률 — 심판이 견주는 값. 프로드와 같은 정의다
    (``ctc_engine._score_tokens``: ``scores.sum() / n_frames``).

    세 가지가 이 정의에 걸려 있다.

    ① **삭제가 공짜가 되면 안 된다.** 토큰이 차지한 프레임만 평균 내면 짧은 후보가 자신 없는
       구간을 blank로 넘기고 그 비용을 전혀 내지 않는다. 프로드가 실오디오에서 겪은 파국이
       그것이다(134줄 중 53줄 교체, 그중 11줄이 글자가 사라진 것: 私 와타쿠시오→시오).
       blank 프레임의 로그확률을 합에 넣으면, 발성된 프레임을 버린 후보가 그만큼 벌점을 받는다.

    ② **낱말 구간만 봐도 안 된다.** 재정렬은 라인 전체에 걸리므로, 짧은 후보가 그 낱말에서
       번 이득을 옆 낱말에서 치를 수 있다. 구간만 보면 그 대가가 계산에서 빠진다 —
       2026-08-01에 실제로 그렇게 재서 길이 편향이 오히려 커졌다(짧아짐 8.6% → 19.8%).

    ③ **분모가 상수라야 한다.** 후보들이 같은 창에서 채점되므로 이 비교는 사실상 경로 총
       로그우도 비교이고, 길이가 다른 후보끼리도 같은 잣대에 놓인다.
    """
    if frame_scores is None or len(frame_scores) == 0:
        return None
    return float(frame_scores.sum()) / len(frame_scores)


def _spread_piled_segments(segs: list[dict[str, Any]], presence, frame_sec: float) -> int:
    """한 시각에 **뭉친** 세그를 그 앞 발성 구간에 펴 준다.

    CTC가 같은 가사를 여러 렌디션에 걸쳐 흘릴 때, 앞쪽 글자들이 스팬 길이 0으로 무너져
    한 프레임에 쌓인다 — 熱異常에서 세그의 **8.7%**가 앞 세그와 시작 시각이 같았다.
    화면에서는 그 음절들이 존재하지 않는 것처럼 스쳐 지나가고, 다음 실제 세그까지의
    구간은 통째로 비어 있다.

    ``UST`` 실측(熱異常 3:16 `編んだ名誉で`):

        정답      あ196.95 ん197.15 だ197.28 め197.60
        뭉친 상태  あ196.79 ん196.79 だ196.79 め196.79

    뭉친 덩이를 «앞 세그 시작 ~ 다음 실제 세그 시작» 사이에 **방출의 비-blank 확률로
    가중해** 편다. 균등 분배가 아니라 가중인 이유: 그 구간에 쉼이 섞여 있으면 균등
    분배는 음절을 무음 위에 놓는다. 가중하면 소리가 난 자리로 모인다.

    시작 시각만 고친다 — 끝은 ``_extend_segments``가 다시 잡는다.
    """
    import numpy as np

    fixed = 0
    index = 0
    while index < len(segs) - 1:
        if abs(segs[index + 1]["start"] - segs[index]["start"]) > 1e-6:
            index += 1
            continue
        # 같은 시각에 쌓인 덩이 [index .. stop)
        stop = index + 1
        while stop < len(segs) and abs(segs[stop]["start"] - segs[index]["start"]) <= 1e-6:
            stop += 1
        origin = segs[index]["start"]
        limit = segs[stop]["start"] if stop < len(segs) else segs[-1].get("end", origin)
        span = limit - origin
        count = stop - index
        if span > 1e-3 and count > 1:
            lo = max(int(origin / frame_sec), 0)
            hi = min(int(limit / frame_sec) + 1, len(presence))
            weights = presence[lo:hi] if hi > lo else None
            if weights is not None and len(weights) >= count and float(weights.sum()) > 0:
                cumulative = np.cumsum(weights) / float(weights.sum())
                # k번째 음절은 누적 발성량의 k/count 지점에서 시작한다.
                for offset in range(1, count):
                    position = int(np.searchsorted(cumulative, offset / count))
                    segs[index + offset]["start"] = round((lo + position) * frame_sec, 3)
            else:
                step = span / count
                for offset in range(1, count):
                    segs[index + offset]["start"] = round(origin + offset * step, 3)
            for offset in range(count):
                segs[index + offset]["end"] = max(
                    segs[index + offset]["end"], segs[index + offset]["start"]
                )
            fixed += count - 1
        index = stop
    return fixed


def _extend_segments(segs: list[dict[str, Any]], line_end: float, hold_max: float) -> int:
    """세그 끝을 **다음 세그 시작까지** 늘린다 — 프로드 ``segmentation._extend_to_next_start``.

    CTC 스팬은 본래 뾰족하다. 실측 세그 길이 중앙값이 20ms인데 UST 노트는 116~219ms이고
    사이가 100~200ms씩 비어 있다(2026-08-02). 그대로 노래방에 쓰면 이어 부르는 구간에서도
    음절마다 20ms만 켜졌다 꺼진다 — 「쭉 이어 부르는데 발음이 끊겨 보인다」는 지적이 이것이다.
    프로드는 문자 모드에서 이 늘이기를 이미 하는데(``segmentation.py``) 하네스만 빠져 있었다.

    **시작은 손대지 않는다.** 시작은 CTC 실측이고, 늘이는 것은 «언제까지 켜 둘 것인가»라는
    표시 규약이다. 추정치에 맞춰 실측을 옮기면 정보가 순손실이라는 것은 프로드 VAD 층 이식
    때 이미 겪었다(세그 리스케일로 음절 정확도 88.8% → 43.5%).

    ``hold_max``는 «간주에 흩어진 음절이 화면에서 쭉 늘어나는» 반대 증상을 막는 한도다.
    UST 노트 15,503개 실측에서 99.5퍼센타일이 1.111s, 1.5s를 넘는 것은 0.29%뿐이라 —
    그보다 긴 공백은 늘임음이 아니라 쉼이다. 한도를 넘으면 거기서 끊고 어둠을 남긴다.
    """
    stretched = 0
    for current, following in zip(segs, segs[1:]):
        target = min(following["start"], current["start"] + hold_max)
        if target > current["end"]:
            current["end"] = round(target, 3)
            stretched += 1
    if segs:
        target = min(line_end, segs[-1]["start"] + hold_max)
        if target > segs[-1]["end"]:
            segs[-1]["end"] = round(target, 3)
            stretched += 1
    return stretched


def _shift_line(line: dict[str, Any], delta: float) -> None:
    """라인과 그 안의 세그를 **통째로** 옮긴다(강체 이동)."""
    line["start"] = round(line["start"] + delta, 3)
    line["end"] = round(line["end"] + delta, 3)
    for seg in line.get("segs") or []:
        seg["start"] = round(seg["start"] + delta, 3)
        seg["end"] = round(seg["end"] + delta, 3)


def _respace_repeated_lines(
    lines: list[dict[str, Any]], sources: list[str], min_run: int = 3, factor: float = 1.5
) -> int:
    """같은 가사가 연속 반복될 때 **한 렌디션을 건너뛴 자리**를 되돌린다.

    반복 훅은 박자 위에 있어 시작 간격이 거의 일정하다 — 熱異常 ``黒い星が`` 16회의 UST
    정답 간격은 **0.49초로 완전히 균일**했다. 그런데 정렬은 렌디션을 하나 놓치는 일이 있고,
    그러면 그 자리 간격만 두 배가 되고 **뒤쪽 형제가 통째로 한 박자 밀린다**:

        정답   0.49 0.49 0.49 0.49 0.49 0.49 0.49
        실측   0.56 0.48 0.40 0.48 **1.12** 0.40 0.48   ← 라인 90~92가 +0.5초씩 밀림

    간격 중앙값의 ``factor``배를 넘는 자리를 찾아 **초과분만큼 뒤쪽을 당긴다**. 등간격으로
    전부 재배치하지 않는 이유: 앞쪽 형제들은 이미 맞아 있어(오차 0.03~0.07초) 건드리면
    손해다. 틀어진 곳만 고친다.

    세그는 라인과 함께 **강체 이동**한다. 라인 추정치에 맞춰 세그를 리스케일하면 CTC 실측이
    손상된다는 것은 이미 겪었지만(음절 88.8 → 43.5%), 여기서는 스팬 구조를 그대로 두고
    통째로 옮기는 것이고, 반복 훅은 렌디션끼리 음향이 같아 옮긴 자리도 같은 소리 위다.

    ``_clamp_repeated_outliers``(프로드)와 다른 규칙이다 — 그쪽은 형제 중앙값의 2.5배를 넘는
    **길이**를 자르는데, 여기서 틀린 것은 길이가 아니라 **위치**다(라인 90 길이 0.40초로 형제와
    같고 시작만 0.59초 늦다). 그래서 그 규칙은 이 자리에 발동하지 않는다.
    """
    fixed = 0
    index = 0
    while index < len(lines):
        text = sources[index].strip() if index < len(sources) else ""
        stop = index + 1
        while stop < len(lines) and text and sources[stop].strip() == text:
            stop += 1
        if text and stop - index >= min_run:
            for _ in range(stop - index):  # 한 구간에 건너뛴 자리가 여럿일 수 있다
                starts = [lines[k]["start"] for k in range(index, stop)]
                gaps = [b - a for a, b in zip(starts, starts[1:])]
                median = statistics.median(gaps) if gaps else 0.0
                if median <= 0.05:
                    break
                worst = max(range(len(gaps)), key=lambda k: gaps[k])
                excess = gaps[worst] - median
                if gaps[worst] <= median * factor or excess <= 0.10:
                    break
                for k in range(index + worst + 1, stop):
                    _shift_line(lines[k], -excess)
                fixed += 1
        index = max(stop, index + 1)
    return fixed


def _enforce_monotonic(lines: list[dict[str, Any]]) -> int:
    """라인 경계에서 세그가 뒤로 밀리는 것을 막는다. 고친 자리 수를 돌려준다.

    CTC forced align은 단조라 **라인 안**은 절대 뒤집히지 않는다(실측 역전 0건). 그런데
    ``_refine``은 라인마다 ``[start-pad, end+pad]`` 창에서 따로 정렬하고, 그 pad가 인접
    라인과 겹치기 때문에 앞 라인의 마지막 세그가 다음 라인 첫 세그보다 늦게 끝날 수 있다.
    이어 붙이면 시간이 거꾸로 흐르고, 카라오케 하이라이트가 뒤로 튀어 보인다(실측 곡당 1~8건).

    pad를 없애면 원천 차단되지만 그건 앵커 라인 경계의 오차를 흡수하는 장치라 포기할 수 없다.
    그래서 창은 그대로 두고 **결과만** 손본다: 겹친 구간을 반으로 갈라 양쪽에 나눠 준다.
    어느 쪽 정렬이 옳은지는 판단하지 않는다 — 판단할 근거가 없기 때문이다.
    """
    flat = [seg for line in lines for seg in (line.get("segs") or [])]
    fixed = 0
    for previous, current in zip(flat, flat[1:]):
        if current["start"] >= previous["end"]:
            continue
        middle = (previous["end"] + current["start"]) / 2
        # 앞 세그를 먼저 지킨다 — 경계가 그 시작보다 앞이면 세그 자체가 뒤집힌다.
        # 겹침이 심해 앞 세그 시작이 뒤 세그 끝보다도 늦은 «완전 역전»에서는 반으로 나눌
        # 구간이 아예 없다. 그때는 뒤 세그가 0길이로 눌린다 — 두 정렬이 모순인 자리라
        # 어느 쪽도 온전히 살릴 수 없고, 시간이 거꾸로 흐르는 것보다는 낫다.
        middle = max(middle, previous["start"])
        previous["end"] = round(middle, 3)
        current["start"] = round(middle, 3)
        if current["end"] < current["start"]:
            current["end"] = current["start"]
        fixed += 1
    return fixed


def _cuda_sync() -> None:
    """CUDA 큐를 비운다 — 구간별 시간 계측이 의미를 가지려면 필요하다."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # 계측 보조라 실패해도 정렬은 계속돼야 한다
        pass


def _refine_stats(
    refined: int, converted: int, total: int, fallbacks: dict[str, int]
) -> dict[str, Any]:
    return {
        "refined_lines": refined,
        "total_lines": total,
        "refined_ratio": round(refined / total, 4) if total else 0.0,
        "script_converted_lines": converted,
        "fallbacks": fallbacks,
    }


def _max_or_none(*values: float | None) -> float | None:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def register(aligner_registry: dict[str, type[AlignerAdapter]]) -> None:
    """2패스 조합을 하네스 레지스트리에 배선한다."""

    for spec in CONFIGS:
        aligner_registry[spec.name] = _config_class(spec)


def _config_class(spec: TwoPassConfig) -> type[TwoPassAligner]:
    class ConfiguredTwoPassAligner(TwoPassAligner):
        name = spec.name
        config = spec

    ConfiguredTwoPassAligner.__name__ = "TwoPass_" + spec.name.replace("-", "_")
    ConfiguredTwoPassAligner.__qualname__ = ConfiguredTwoPassAligner.__name__
    return ConfiguredTwoPassAligner
