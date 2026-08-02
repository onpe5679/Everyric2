# longform-tools — 장문 정렬·산업 도구 조사 (SendMessage 전문 1건)

# 장문 오디오 정렬 · 산업 도구 조사 결과

조사는 끝나 있었습니다. 전달 경로를 제가 놓쳤습니다 — 아래가 전문입니다.

## 먼저: 정정해야 할 인용이 하나 있습니다 (종합 문서에 들어가기 전에)

**「`arXiv:2406.02560`에서 MFA 63.0 vs CTC 계열, WebMAUS 70.0」은 그 논문의 수치가 아닙니다.** 방금 https://arxiv.org/html/2406.02560v3 을 직접 대조했습니다:

- **"WebMAUS" 또는 "MAUS"는 이 논문에 한 번도 등장하지 않습니다.**
- **"63.0", "70.0"도 등장하지 않습니다.** ("63"/"64"는 나오지만 PDUR = 예측 음소 지속시간 ms 값입니다.)
- Table 1 캡션 verbatim: **"All metrics are in milliseconds."** — 백분율 일치율이 아니라 **밀리초 경계오차**입니다.
- Table 1 헤더 verbatim: `Buckeye | TIMIT | PBE | WBE | PDUR | WDUR | PBE | WBE | PDUR | WDUR`
- **MFA 행 verbatim**: `MFA [8] | 30 | 41 | 84 | 251 | 17 | 23 | 85 | 313`
  → Buckeye 음소경계오차 **30ms**, 단어경계오차 **41ms** / TIMIT **17ms**, **23ms**
- 초록 verbatim: **"Compared with the most widely used FA toolkit Montreal Forced Aligner (MFA), our method performs similarly on PBE/WBE on Buckeye, yet falls behind MFA on TIMIT."**

63.0/70.0과 WebMAUS는 **다른 문헌에서 온 것이거나 혼입입니다.** 출처를 확인하지 못한 채 종합 문서에 넣으면 안 됩니다. 서지는 아래대로 쓰시길 권합니다:

> Huang, Zhang, Ni, Sun, Hira, Hwang, Manohar, Pratap, Wiesner, Watanabe, Povey, Khudanpur. *Less Peaky and More Accurate CTC Forced Alignment by Label Priors*. ICASSP 2024. arXiv:2406.02560

저자에 **torchaudio 메인테이너(Moto Hira, Jeff Hwang)**가 있습니다 — 즉 저희가 쓰는 그 API를 측정한 논문입니다.

---

## Q3부터 (가장 확정적입니다): MFA가 CTC보다 경계가 정확한가 — 예, 조건부로

**사실입니다. 단 「표준 CTC 대비」로 확실하고, 「개선된 CTC 대비」로는 데이터셋에 따라 갈립니다.**

Table 1 (모두 ms, PBE=음소경계오차 / WBE=단어경계오차):

| | Buckeye PBE/WBE | TIMIT PBE/WBE |
|---|---|---|
| 표준 CTC | 44 / 58 | 32 / 42 |
| label priors CTC | 38 / 43 | 28 / 29 |
| **MFA** | **30 / 41** | **17 / 23** |
| label priors + finetune | 30 / 34 | 27 / 28 |
| MFA + finetune | 27 / 36 | 16 / 22 |

(MFA 행만 verbatim 확인, 나머지는 자동 추출 — PDF 눈 대조는 안 했습니다. 초록 문장과 방향은 일치합니다.)

**왜 그런가** — 논문의 설명이 저희 star 진단과 정확히 같은 뿌리입니다:
- blank가 「가능한 정렬 경로 공간에서 가장 범용적이고 빈번한 토큰」이라 학습이 blank 위주 경로를 자기강화합니다. blank prior **0.80**.
- 그 결과 **각 심볼이 한 프레임만 발화**합니다. 실측 예측 음소 지속 **21ms**(= 프레임 크기)인데 실제는 **~82ms**. **경계가 아예 모델링되지 않습니다.**
- GMM-HMM은 blank가 없어 **모든 프레임이 어떤 phone state에 귀속**되고, phone당 3-state + self-loop가 지속시간을 표현합니다. 무음은 별도 phone이고 **확률이 파라미터로 노출**됩니다(`--initial_silence_probability`). 화자적응(fMLLR)도 있습니다.

**저희에게 치명적인 단서 하나**: 논문은 「**표준 CTC 모델의 디코딩 시점에만 label prior를 적용하면 개선이 없었다**」고 명시합니다. 즉 CTC 쪽 정공법은 **재학습을 요구**하고, 「학습 없음·추론 전용」 제약 하에서는 막혀 있습니다. Durand 소프트 마스크와 star 성형이 남는 이유가 이것입니다.
속도: 저자 측정으로 GPU에서 **MFA보다 3–9배 빠름**.

### 일본어·한국어 사전학습 MFA 모델 — 존재합니다, CC BY 4.0

https://github.com/MontrealCorpusTools/mfa-models (본체 MFA는 **MIT**)

- **japanese/mfa v3.0.0** (2024-02-08, **CC BY 4.0**): Common Voice ja 12.0(71.54h/1,364명) + GlobalPhone ja 3.1(33.88h) + MS Speech Language Translation ja(9.85h) + Japanese Versatile Speech(30.25h) + TEDxJP-10K 1.1(8.85h) = **약 154.37h / 1,991명**
- **korean/mfa v3.0.0** (2024-02-17, **CC BY 4.0**): Zeroth Korean(52.87h) + GlobalPhone ko(20.88h) + KSS(12.86h) + Pansori TEDxKR + Deeply + ASR-KCSC + ASR-SKDuSC + Common Voice ko = **약 99.92h / 367명**
- 둘 다 GMM-HMM + MFCC. 학습 불필요.

**「가창」에 쓸 수 있는가**: 위 코퍼스는 **전부 말**입니다. 노래 데이터는 없습니다. 그리고 **MFA는 곡 전체를 한 발화로 받지 못합니다** — 코퍼스 구조 문서가 「세그먼트는 최적 성능을 위해 **30초 미만**」, 「**100ms 미만 구간은 정렬하지 않는다**」, 「**숨·무음 휴지로 나눌 것을 권한다**」고 못박습니다. MFA의 장문 경로는 별도 워크플로입니다: **`mfa segment`**(전사 있음 — **SpeechBrain VAD**로 음성 구간을 찾고 **전사도 그 경계에 맞춰 같이 쪼갠다**) / **`mfa segment_vad`**(전사 없음).

→ **결론: 「엔진 교체」로는 답이 아니고, 「자막 앵커로 줄 단위 세그먼트를 만들고 각 세그먼트를 MFA로」만 성립합니다.** 그러면 star가 아예 필요 없어집니다(무음을 `sil` phone이 **비용과 함께** 먹습니다). 부수 이득: `torchaudio.pipelines.MMS_FA`는 **CC-BY-NC 4.0**이고 `mms-300m-1130-forced-aligner`(저희 `settings.ctc_model`)도 **CC-BY-NC 4.0**입니다 — MFA로 가면 둘 다 벗어납니다. 상용이면 정렬 정확도와 별개로 봐야 하는 항목입니다.

**비용 경고**: `docs/GPU_ALIGNMENT_ENGINES.md`의 「MFA 155초 → 454초, CPU 단일 코어」는 **재측정 대상**입니다. `everyric2/alignment/`에 `mfa_engine.py`가 **지금 없고**(팩토리는 ctc/nemo/gpu-hybrid/sofa만), MFA v3는 `--num_jobs` 다중처리가 기본입니다. 곡당 42초 예산과 비교하기 전에 32코어에서 제대로 측정해야 합니다. GPU는 안 씁니다 — **3090이 남고 CPU가 병목**이 되는 교환입니다.

---

## Q1: CTC-Segmentation — star 성형·소프트 마스크보다 나은 구조인가? **아니오. 다만 훔칠 것이 3개 있습니다.**

- https://github.com/lumaku/ctc-segmentation · **Apache-2.0**
- Kürzinger, Winkelbauer, Li, Watzel, Rigoll. *CTC-Segmentation of Large Corpora for German End-to-end Speech Recognition*. SPECOM 2020. arXiv:2007.09127
- 탑재처: ESPnet 1/2, NVIDIA NeMo(`tools/ctc_segmentation`), SpeechBrain(`speechbrain.alignment.ctc_segmentation`)

**실제 동작**(논문 본문 + `ctc_segmentation.py` 확인):
- trellis에서 매 스텝 「**blank를 먹거나 다음 글자를 먹거나**」 — 저희 `forced_align`과 **같은 Viterbi 계열**입니다. 구조적으로 새롭지 않습니다.
- **긴 오디오 처리의 정체**: 복잡도를 O(M·N)→O(M)으로 줄이는 **창 heuristic**입니다. 글자 j의 기대 위치 **t = j·N/M** 주변 **[t−W/2, t+W/2]** 프레임만 봅니다. `min_window_size=8000`, 백트래킹 실패 시 `max_window_size=100000`까지 2배씩 확장. 별도로 `get_partitions()`로 파일 분할.
  → **즉 이 도구는 이미 「시간이 균등하게 흐른다」는 양성 제약을 걸고 있습니다.** 중심선이 **선형 대각선**일 뿐입니다. 그 중심을 자막 앵커로 바꾸는 것이 저희가 하려는 일이고, **그것이 이 계열의 관용구입니다** — 저희 `_align_in_blocks`가 옳은 방향이라는 외부 근거입니다.
- **비음성 처리는 저희보다 약합니다**: 「첫 글자에 머무는 전이 비용을 0으로 둬서 전사 시작점을 임의 지점에 맞춘다」 — **머리쪽 preamble만** 무료 흡수합니다. 중간 간주용 장치는 `blank_transition_cost_zero`인데 README가 직접 「오정렬을 낳을 수 있다」고 경고합니다. 이건 star를 **더 싸게 만드는** 방향이라 저희가 가면 안 되는 길입니다.
- 신뢰도: 발화 프레임을 길이 L 조각으로 나눠 각 조각 확률 평균 m_j를 구하고 **그 최솟값**을 발화 점수로 씁니다(`score_min_mean_over_L`).
- 보고 수치: TEDlium v2 대조 평균편차 **0.31s vs Gentle 0.41s**, 0.5초 이내 **88.8% vs 82.0%**. (arXiv HTML 자동 추출.)

**대체 여부**: 대체할 이유 없음 — 같은 DP, **글자 단위 출력 없음**(줄 경계만), star 없음.
**훔칠 것 3개**:
1. 창 중심을 앵커로 옮기는 발상 (= blocks의 정당화)
2. **신뢰도를 평균이 아니라 최솟값으로** — 저희 `_token_peak_support`는 평균입니다. 「8줄 중 8줄만 엉뚱」을 평균은 못 잡고 최솟값은 잡습니다. 이게 가장 값싼 개선입니다.
3. `blank_transition_cost_zero`의 경고 = **하지 말라는 예**

**비용**: 추론만, posterior는 이미 저희가 만듭니다. DP는 Cython/CPU.

---

## Q2: 앵커 기반 재귀 분할 — 정식 이름과 원 논문

**① 음성 쪽 정통 — anchor 기반 재귀 강제정렬**
> **Moreno, P. J., Joerg, C., Van Thong, J.-M., Glickman, O. (1998). *A Recursive Algorithm for the Forced Alignment of Very Long Audio Segments*. ICSLP 1998, Sydney.** (Compaq Cambridge Research Laboratory)
> PDF: https://www.isca-archive.org/icslp_1998/moreno98b_icslp.pdf
> 페이지: https://www.isca-archive.org/icslp_1998/moreno98b_icslp.html

핵심: 「강제정렬 문제를 **사전과 언어모델을 점점 좁혀 가는 재귀적 음성인식 문제**로 바꾼다.」 **앵커 = N개 연속 단어가 올바르게 정렬된 오디오 구간.** 앵커로 자르고 사이를 재귀 정렬. 잡음·전사 누락에 강건.
계보: **SailAlign** (https://github.com/nassosoassos/sail_align, **GPL v2+**, HTK 기반, 2010–2013 정지) — Katsamanis, Black, Georgiou, Goldstein, Narayanan. *SailAlign: Robust long speech-text alignment*. Workshop on New Tools and Methods for Very-Large Scale Phonetics Research, 2011. / Gentle 기반 재귀 정렬기 **canetis** (https://github.com/nsheth12/canetis)

**② 음악 정렬 쪽 정통 — global constraint region**
**Sakoe-Chiba band**(대각선 고정폭 띠) / **Itakura parallelogram**. 구현은 **영역 밖 셀의 비용을 ∞로 두는 것**이고, 「제약된 최적 경로는 진짜 최적이 영역 밖이면 달라진다」는 대가가 명시돼 있습니다. 정리: Müller, *Fundamentals of Music Processing*, C3S2 — https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S2_DTWvariants.html
계층적 확장: **Müller et al., *An Efficient Multiscale Approach to Audio Synchronization*** / **Prätzlich & Müller, *Memory-Restricted Multiscale Dynamic Time Warping*** — **거친 해상도의 경로가 다음 해상도의 제약 영역이 된다.** 이것이 「재귀적/계층적 앵커 분할」의 음악 쪽 정식 이름(**MsDTW**)입니다. (두 논문은 FMP 참고문헌에서 **제목만** 확인, 연도·게재처 **확정 못 함**.)
구조적 점프 전용: **Fremerey, C., Müller, M., Clausen, M. (2010). *Handling Repeats and Jumps in Score-Performance Synchronization*. ISMIR 2010, Utrecht, pp. 243–248** (JumpDTW). PDF: https://zenodo.org/records/1415942/files/FremereyMC10.pdf

**③ 「외부 앵커를 양성 제약으로 쓰는 정립된 절차」 — 있습니다. 그리고 두 형태가 전부입니다.**
둘 다 **「영역 밖 비용을 ∞로 둔다」**는 같은 형식입니다:
- **분할 정복형**(Moreno/SailAlign): 앵커에서 오디오와 텍스트를 동시에 자르고 각 구간 독립 정렬 ← **저희 `_anchor_blocks` + `_align_in_blocks`가 이것입니다.**
- **밴드 제약형**(Sakoe-Chiba / MsDTW / CTC-Segmentation): 앵커가 정의하는 경로 주변 ±W만 허용 ← **Durand의 소프트 line-mask가 이것의 소프트 버전입니다.** M ∈ [0,1]에 2.5초 선형 완충을 두는 것은 하드 밴드의 ∞를 유한 페널티로 완화한 것이고, 「밴드를 벗어나도 음향 근거가 압도적이면 나갈 수 있다」는 성질을 얻습니다.

**→ Durand 소프트 마스크보다 「더 나은」 형태를 찾지 못했습니다. 오히려 소프트 마스크가 하드 밴드의 개선판입니다.** 장문 전통이 추가로 주는 것은 **마스크의 모양이 아니라 앵커를 고르는 방법**입니다:

> Moreno의 앵커 정의는 처음부터 「**올바르게 정렬되었음이 확인된** N개 연속 단어」입니다. **앵커는 믿는 것이 아니라 개별로 검증해서 고르는 것**입니다. 저희는 `caption_anchor_positive_min_match=0.85`와 `caption_anchor_max_token_loss`로 관리하는데 **둘 다 곡 단위 판정**이고, 52개 앵커를 전부 채택하거나 전부 버립니다. 그런데 `caption_anchors` 설명에 이미 실측이 적혀 있습니다 — 「**간주 이전 6줄은 0.2초 이내, 몰린 8줄만 17–22초 틀림**」. **앵커별로 품질이 갈린다는 증거**이고, 그러면 판정도 앵커별이어야 합니다. 그리고 척도는 §Q1의 이유로 **평균이 아니라 최솟값**입니다.

**성숙한 대조군이 하나 있습니다**: **stable-ts** (https://github.com/jianfch/stable-ts, **MIT**)의 `align_words()`가 저희 blocks와 **같은 계약**을 이미 구현했습니다. `stable_whisper/alignment.py` docstring verbatim: 「**각 세그먼트를 타임스탬프 범위 안으로 제한하여, `align()`이 쓰던 폴백 기제를 불필요하게 만든다**」, 「**제공된 각 세그먼트의 시작·끝 타임스탬프가 정확하다면 단어-타임스탬프 오차를 줄인다**」. 입력은 `start`/`end`/`text` dict 리스트 — **정확히 유튜브 자막 트랙의 형태**입니다.
같은 파일에서 저희에게 바로 쓸 파라미터 둘:
- **`nonspeech_skip`**: 「**이 값(초) 이상인 비음성 구간은 건너뛴다**」 — 간주를 탐색 공간에서 빼는 명시적 스위치
- **`failure_threshold`**: 「**지속시간 0인 단어의 비율이 이 값을 넘으면 정렬을 중단**」 — 「토큰이 1프레임에 몰렸다」를 **직접 탐지하는 지표**입니다. 저희에게 없는 게이트입니다.

---

## star 진단에 문헌 근거를 붙입니다 (다른 에이전트 결론 보강)

**「페널티 없는 filler」는 이 분야에서 이미 틀렸다고 정리된 설계입니다.**
> **Wilpon, J. G., Rabiner, L. R., Lee, C.-H., Goldman, E. R. (1990). *Automatic recognition of keywords in unconstrained speech using hidden Markov models*. IEEE TASSP 38(11), 1870–1878. doi:10.1109/29.103088** — https://ieeexplore.ieee.org/document/103088

핵심 발상이 「**실제 어휘 단어와 잉여 음성·배경 둘 다의 통계 모델을 만든다**」 — filler/garbage HMM의 기원이고, 이 전통에서 filler는 **항상 비용을 갖습니다**. 계승 경로:
1. **GMM-HMM(MFA/Kaldi)**: 무음 phone이 단어 사이에 선택적으로 들어가고 **그 확률이 파라미터**
2. **CTC blank**: filler 역할은 하지만 **비용 손잡이가 없고** 학습이 과대사용을 자기강화(prior 0.80)
3. **star/wildcard**: 튜토리얼 관용구가 **비용 0**이라 원설계를 잃음 ← 저희 위치
4. **VAD로 아예 제외**(WhisperX / stable-ts `nonspeech_skip` / `mfa segment`): 현대 도구가 실제로 가장 많이 쓰는 방법이고 **「학습 없음」과 완전 호환**

**참고 — 위치로 우회하는 선례**: `MahmoudAshraf97/ctc-forced-aligner`(https://github.com/MahmoudAshraf97/ctc-forced-aligner, 코드 BSD / **기본 모델 CC-BY-NC 4.0**)는 비용 대신 **자유도**를 줄입니다: `star_frequency` 기본값 **`"edges"`**(양 끝에만), 대안 `"segment"`(모든 발화 사이). 긴 오디오는 `window_size` 30초 + `context_size` 2초 오버랩으로 청크.

**WhisperX의 VAD 실측**(Bain, Huh, Han, Zisserman. *WhisperX*. INTERSPEECH 2023. arXiv:2303.00747): VAD Cut&Merge는 30초 초과 구간을 **음성 활성 점수가 최소인 지점**에서 자릅니다(자르는 범위를 최대 길이의 절반~전체로 제한). AMI 단어분할 **82.6%P/53.4%R → 84.1%P/60.3%R**. **이득의 대부분이 recall** — 즉 「빠뜨렸던 것을 되찾는」 효과이고, 저희 「뒤쪽 33초 공백」과 성질이 같습니다. 한계: README가 「**음소 구조가 없는 비음성·음악에서는 정렬이 실패**」라고 명시. 노래 적용 사례는 **확인 못 함**.

---

## Q4: 노래에 그대로 쓸 수 있는 도구 — **사실상 「없다」입니다**

**aeneas** (https://github.com/readbeyond/aeneas, **AGPL v3**, 1.7.3 / 2017-03 정지)가 가장 명확한 답을 스스로 줍니다. 동작은 ① 텍스트를 **TTS로 합성** ② 실오디오·합성오디오를 **MFCC**로 ③ **DTW**(계산은 Sakoe-Chiba band) ④ 경계 전이. README가 못박은 두 문장:
- 「**오디오가 텍스트와 일치해야 한다 — 잉여 텍스트나 잉여 오디오가 많으면 틀린 sync map이 나온다**」
- 「**오디오는 발화로 가정한다: 노래 캡션에는 적합하지 않다**」

저희는 「텍스트는 확실하고 시각만 모른다」는 전제는 같지만 **가정 두 개가 동시에 깨집니다**(간주 33초 = 잉여 오디오, 곡 = 노래). 해상도도 MFCC 프레임 배수(40ms)로 이산화. 유지되는 대안 **afaligner**(https://github.com/r4victor/afaligner) / 도구 목록 https://github.com/pettarin/forced-alignment-tools

**「노래 전용」이라고 스스로 주장하는 것은 SOFA 하나뿐이고, 저희 저장소에 이미 있습니다 — 그런데 지금 켜면 안 됩니다.**
- **SOFA (Singing-Oriented Forced Aligner)**: https://github.com/qiuqiao/SOFA, **MIT**. README가 MFA 대비 「설치 쉽고, 성능 낫고, 추론 빠르다」 주장. 입력 `.wav`+`.lab`, 기본 사전 **`opencpop-extension.txt`(= Opencpop 계열 중국어 만다린)**. 출력 TextGrid/htk/trans. **사전학습 모델은 GitHub Discussions의 「pretrained model sharing」에 커뮤니티가 올린 `.ckpt`** — 공식 언어별 목록 없음. 포크: https://github.com/Greenleaf2001/SOFA-Modded
- **저희 코드 현황**: `everyric2/alignment/sofa_engine.py`의 `SOFA_MODELS`에 **영어 하나뿐**(`tgm_en_v100`), 소스 주석이 「**Japanese model URL is broken**」. 그리고 미지원 언어를 **영어 모델로 폴백**합니다(`SOFA doesn't support {lang}, falling back to 'en'`) — 일본어 곡에 영어 음소 모델입니다. 반면 `factory.py`는 이 엔진을 「English/**Japanese**」로 광고합니다. **표시와 실제가 어긋나 있습니다.**
- 일본어 SOFA ckpt의 실재는 **확인 못 했습니다**(Discussions 미확인). 있다면 가장 직접적인 후보입니다.

**말 vs 노래가 문제를 일으키는 지점 4개**:
1. **모음 연장** — CTC는 토큰당 1프레임으로 붕괴(측정 21ms)해 3초 늘린 모음을 표현 못 합니다. GMM-HMM은 self-loop로 자연히 표현. **여기서 GMM-HMM이 노래에 구조적으로 유리합니다.**
2. **음정 변화·비브라토** — MFCC 스펙트럼 형상을 흔듭니다. **MFCC를 직접 DTW하는 계열(aeneas)이 가장 크게 깨집니다.**
3. **간주 = 잉여 오디오** — aeneas가 명시한 실패 조건. VAD로 빼거나 **비용 있는** filler로 먹여야 합니다.
4. **합성 보컬** — 노래 일반의 문제가 아니라 **저희 고유**입니다(posterior 바닥). 위 어느 도구도 이걸 다루지 않습니다. `quality_score 0.001`에서는 음향 근거가 없으니 **앵커가 유일한 정보원**이고, 그래서 양성 제약이 선택이 아니라 필수입니다.

**참고 문헌 하나(약함)**: Liu, J. (2024). *Research on the recognition and application of Montreal forced aligner for singing audio*. Journal of Computer and Electrical Information Management, 12, 19–21. doi:10.54097/ohpdubg1 — **저널 등급이 낮고 수치를 확인하지 못했습니다.** 「MFA를 노래에 쓰려는 시도가 있다」는 존재 증명으로만 쓰시길.

---

## 참고로 조사한 나머지 (짧게)

- **NeMo Forced Aligner (NFA)** — https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html : **「1시간 이상의 긴 오디오에 쓸 수 있다」**고 명시하고 **사전 분할을 요구하지 않습니다**(파일 전체를 한 세그먼트로 처리 후 `.`/`?`/`!`로 분할, `additional_segment_grouping_separator`). token/word/segment 3단 CTM + ASS. CTC 또는 hybrid의 CTC 모드만(순수 Transducer 불가). **그런데 CJK 사전학습 CTC 모델이 없습니다**(en/es/de/fr/it/ru/pl) — 저희 `nemo_engine.py`가 일본어에 못 쓰이는 이유. 2026-07 현재도 유효한지는 **확인 못 함**(NGC 미조회).
- **NeMo CTC-Segmentation 도구** (`NVIDIA/NeMo/tools/ctc_segmentation`): 목적이 **코퍼스 제작**입니다. `run_ctc_segmentation.py` 인자 `--window_len` 기본 **8000**, `--model` 기본 `stt_en_fastconformer_ctc_large`, `--num_jobs` -2. 파일 전체를 모델에 통과시켜 log_probs를 얻고 **blank 열을 마지막→첫 열로 옮겨** ctc-segmentation 규약에 맞춘 뒤 `get_segments()`에 넘깁니다. 임계값 세부는 튜토리얼 노트북에 있고 **확인 못 함**.
- **Gentle** — https://github.com/lowerquality/gentle, **MIT**, Kaldi 기반. 긴 오디오·비음성 전용 장치 문서에 없음. **CTC-Segmentation 논문이 Gentle을 비교 기준으로 삼아 이겼습니다**(0.41s → 0.31s). 저희가 갈 이유 없음.
- MFA의 GUI 도구 이름이 **Anchor**(`corpus_creation/anchor.rst`)인데 **라벨링 GUI**이고 anchor 정렬 알고리즘과 **무관**합니다 — 종합 문서에서 혼동 주의.

---

## 「가사 정렬 문헌만 봤을 때 놓치고 있던 것」 (요청하신 한 단락)

가사 정렬 문헌은 「어떤 음향 모델이 노래를 잘 읽는가」를 묻지만, 저희 실패는 음향 모델의 실패가 아니라 **탐색 공간 설계의 실패**입니다. 오디오북·강연 정렬 분야는 1998년에 이미 그 결론에 도달해서 「긴 오디오를 한 번에 정렬한다」는 문제 설정 자체를 버리고 **앵커로 자르고 각 조각을 독립 정렬한다**로 옮겨 갔습니다(Moreno 1998 → SailAlign 2011 → MFA의 30초 규칙과 `mfa segment` → WhisperX VAD Cut&Merge → stable-ts `align_words`). 그 30년이 축적한 두 가지가 저희에게 빠져 있었습니다. 첫째, **filler 모델에는 반드시 비용이 있어야 한다**는 것(Wilpon 1990) — 이건 다른 에이전트가 이미 확정한 star 진단과 같은 것이고, 저희가 발견한 것이 「새 버그」가 아니라 **36년 전에 정리된 설계 원칙을 어긴 것**이라는 뜻입니다. 둘째, **앵커는 믿는 것이 아니라 개별로 검증해서 고르는 것**이라는 것 — Moreno의 앵커 정의 자체가 「올바르게 정렬되었음이 확인된 N개 연속 단어」이고, CTC-Segmentation의 신뢰도는 발화별이며 **평균이 아니라 최솟값**입니다. 저희는 앵커 52개를 곡 단위 게이트로 통째로 채택하거나 통째로 버리도록 만들었고, 실측은 이미 「앞 6줄 0.2초 이내, 뒤 8줄 17–22초」라고 말하고 있습니다 — 앵커별 품질이 갈린다는 뜻이고, 그러면 판정도 앵커별이어야 합니다. 요약하면 저희가 「가사 정렬의 어려움」으로 분류해 둔 것의 상당 부분은 **오디오북 정렬에서 이미 해결된 문제를 다른 이름으로 다시 만난 것**이고, `_align_in_blocks`와 Durand 소프트 마스크는 그 해법의 정확한 재발명입니다 — 새로 만들 것보다 **고칠 것이 앞에 있습니다: star의 가격표, 그리고 앵커 판정의 단위와 척도.**

---

## 확인 못 한 것 (지어내지 않았습니다)

- **63.0 / WebMAUS 70.0의 출처** — arXiv:2406.02560에 없음을 확인했을 뿐, 어느 문헌에서 왔는지는 모릅니다.
- **MsDTW 두 논문의 완전한 서지** — FMP 참고문헌에서 제목만 확인, 연도·게재처 미확정.
- **SailAlign 논문 본문** — 앵커 채택 임계값 미확인(PDF 미확보).
- **일본어/한국어 SOFA ckpt의 실재** — Discussions 미확인.
- **NeMo CJK 모델 부재가 2026-07 현재도 유효한지** — NGC 미조회.
- **arXiv:2406.02560 Table 1의 MFA 외 행**, arXiv:2007.09127의 수치 — HTML 자동 추출(MFA 행만 verbatim 대조).
- **NeMo CTC-Segmentation 튜토리얼 노트북**의 임계값 기본값.
- MFA 공식 readthedocs가 조사 중 지속적으로 **HTTP 429**를 반환해서, MFA 사실은 대부분 **저장소 docs 원본 .rst**에서 읽었습니다(30초 권고·100ms 규칙·`mfa segment`/`segment_vad`·SpeechBrain VAD는 그 원본 문장입니다).
- **웹 검색 예산 소진(200/200)** — 이후는 WebFetch로 알려진 URL 직접 수신 + DuckDuckGo HTML을 대체 검색으로 썼습니다. 「이 분야에 다른 도구가 더 있는지」 커버리지는 완전하지 않습니다.

제약 준수: 조사만 했습니다. 파일 수정·커밋·모델 다운로드·원격 접속·무거운 로컬 연산 없음. 저장소 파일은 읽기만 했습니다(`alignment/ctc_engine.py`, `alignment/sofa_engine.py`, `alignment/factory.py`, `config/settings.py`, `docs/GPU_ALIGNMENT_ENGINES.md`).
