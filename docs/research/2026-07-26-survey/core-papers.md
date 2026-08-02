# core-papers — 핵심 6편 정독과 적용 방안

6편 전부 공개 PDF 원문(수식·표 포함)을 확보해 읽었습니다. 추출한 본문은 `C:\Users\user\AppData\Local\Temp\claude\C--devat-everyric2\ccf2cc19-4a2e-4cc6-9f6d-e92f242a5aa3\scratchpad\*.pdf.txt` 에 있습니다.

---

# 0. 먼저: ①②③은 서로 다른 버그가 아니라 하나의 뿌리입니다

torchaudio의 `<star>` 는 emission에 이렇게 붙습니다 (공식 튜토리얼 원문 코드):

```python
star_dim = torch.zeros((1, emission.size(1), 1), device=..., dtype=...)
emission = torch.cat((emission, star_dim), 2)
```

`zeros` = **log p = 0, 즉 p = 1**. 실제 토큰은 `log_softmax` 결과라 항상 ≤ 0입니다. 따라서 **star는 모든 프레임에서 예외 없이 최우선 토큰이며, 어디서 흡수할지에 대한 선호가 정확히 0**입니다. 저희 `C:\devat\everyric2\everyric2\config\settings.py:566` 이 이미 "star scores log(1.0)=0"으로 적어 둔 그 사실입니다.

여기서 세 증상이 전부 따라 나옵니다.

- **①** 가사 8줄을 8.7~24.9s에 몰고 25~58s를 star로 비우는 경로와, 그 반대 경로의 DP 비용이 **같습니다**. VAD 갭이 유일한 신호였던 게 문제가 아니라, VAD 갭이 없으면 DP에 선호가 **아예 없다**는 것이 문제입니다.
- **②** 위 때문에 동률·근동률 경로가 폭발합니다. quality_score가 낮을수록(=posterior가 평평할수록) 편차가 큰 것은 축퇴가 심할수록 부동소수점 차이가 argmax를 뒤집기 때문입니다. 정확히 관측하신 대로입니다.
- **③** 금지 마스크는 star의 선호를 만들어 주지 않습니다. 금지 구간을 피한 다음의 배치는 여전히 임의이고, 탐색공간만 좁혀 놓고 방향은 안 준 상태라 오차가 커질 수 있습니다. 11.22 → 22.58초는 이 구조에서 예상 가능한 결과입니다.

6편 중 이 문제에 **훈련 없이** 쓸 수 있는 해답을 가진 것은 **Durand 2023의 line-based decoding 하나**입니다. 아래에서 논문별로 정리합니다.

---

# 1. Stoller, Durand & Ewert (ICASSP 2019)

**서지** Daniel Stoller, Simon Durand, Sebastian Ewert, "End-to-end Lyrics Alignment for Polyphonic Music Using an Audio-to-Character Recognition Model", ICASSP 2019, pp. 181–185. arXiv:1902.06797.
**PDF** https://arxiv.org/pdf/1902.06797

**방법**
- 입력 raw waveform 22.05 kHz, **352,243 샘플(15.97s)**. Wave-U-Net 변형: **다운샘플링 블록 12개 → 업샘플링 블록 2개**. 출력은 입력 중앙 225,501 샘플(10.23s)만 커버하고, 곡 전체는 225,501 샘플 배수로 창을 밀어 붙여 하나의 확률행렬 P로 이어붙입니다.
- 출력 시간해상도 **초당 약 20개** 문자 확률분포(=50ms/프레임). 어휘는 a–z + apostrophe + space + blank ε = 29.
- 학습은 줄 단위 약한 라벨만 씀. 출력창에 **완전히 포함된** 가사 줄에만 CTC를 적용. 5.11s 이하 줄이 78.4%, 10.22s 이하가 97.2%. 출력창과 겹치는 줄이 없으면 **빈 라벨(empty label)을 만들어** 모델이 간주에서 ε을 내도록 학습시킵니다(원문 §4.1.1). 44,232곡(train 39,232 / val 5,000), Tesla P100 4장으로 25시간.
- 추론: `argmax_{ŷ: B(ŷ)=y} Π_t P[t,ŷ_t]`, O(TO) DP, log 도메인. **P의 모든 항에 U[1e-11, 1e-10] 균등노이즈를 더해** 어떤 심볼이 전 구간 0확률이 되어 모든 경로가 0이 되는 사태를 막습니다. 마지막에 **상수 180ms 지연**을 더합니다(소규모 validation으로 고른 값).

**간주 처리** blank ε 하나뿐입니다. "empty labels are generated to ensure the model predicts silence by outputting the blank symbol ε for instrumental sections."

**결과** Mauch AE 0.35s / 77.2%, Jamendo AE 0.82s / 70.4%. MIREX 2017 최고(AK3 9.03s) 대비 큰 개선.

**코드·모델** 논문이 공개하는 것은 **평가 데이터뿐**: https://github.com/f90/jamendolyrics (현재 DEPRECATED 표시). 모델 가중치 공개 없음. 제3자 재구현 https://github.com/jhuang448/E2E-LyricsAlignment-Implementation (동작·라이선스 **확인 못 함**).

**①②③ 적용**
- 직접 쓸 것 두 개가 있고 둘 다 추론 전용입니다. (a) **emission에 균등노이즈 더하기** — 저희도 `ctc_engine.py`에서 마스크가 실행 불가능해지는 사태를 겪었는데(주석 300행), 노이즈 플로어는 그 종류의 파국을 값싸게 막습니다. (b) **상수 지연 보정** — 저희 파이프라인에 캘리브레이션된 상수 오프셋이 없다면 곡 전체 편향을 공짜로 줄입니다.
- ①에는 도움이 안 됩니다. blank 하나로 간주를 흡수하는 구조는 저희 star와 같은 약점을 갖습니다.

**구현 비용** 추론만. GPU 1대로 충분. 학습 불필요.

---

# 2. Gupta, Yılmaz & Li (Interspeech 2019)

**서지** Chitralekha Gupta, Emre Yılmaz, Haizhou Li, "Acoustic Modeling for Automatic Lyrics-to-Audio Alignment", Interspeech 2019, pp. 2040–2044. arXiv:1906.10369.
**PDF** https://arxiv.org/pdf/1906.10369

**방법**
- Kaldi. GMM-HMM(40k Gaussian, MFCC 39차)으로 정렬을 얻어 NN 학습용 라벨을 만들고, **CNN-TDNN-F**(conv 2층 + TDNN 10층 + rank reduction)를 표준 Kaldi 레시피 5.4로 학습. 프레임 10ms/25ms, **frame subsampling 3 → 실효 30ms**.
- 특징: 기본 40차 MFCC + 100차 i-vector(=140차)에, OpenSMILE LLD 5군 **154차**를 덧붙임 — Auditory(RASTA 26+delta), Energy(4+delta), Chroma(12), Spectral(15+delta), Voicing(F0/jitter/shimmer/HNR 6+delta).
- 도메인 적응: DAMP 솔로 35,662줄로 학습한 모델의 은닉층을 초기값으로 두고, **DALI 폴리포닉 70곡**으로 추가 forward-backward를 소수 epoch. 최적은 **초기 LR 그대로 1 epoch**.
- 속도 교란 증강(×0.9, ×1.1). [10]의 **duration-based modified pronunciation lexicon** 사용(싱잉의 장모음 대응 — 상세는 [10]에 있고 본 논문에는 없음).

**긴 간주 대응 — 여기서 전제를 정정해야 합니다**
논문에는 **긴 간주를 위한 전용 기법이 없습니다.** VAD/노래구간 검출기도 없고, garbage 모델도 없고, 간주 전용 filler phone도 없고, 디코딩 그래프 개조 서술도 없습니다. 논문이 말하는 것은 관찰뿐입니다:

> "One main difference between the Hansen's and Mauch's datasets is that the songs in the Mauch's dataset are rich in long-duration musical interludes that have no singing vocals... We observe that the content-informed features and domain adaptation help to improve the boundaries next to these long interludes."

즉 **잡음강건 특징(RASTA 계열) + 폴리포닉 도메인 적응이 결과적으로 간주 옆 경계를 낫게 했다**는 것이고, 메커니즘은 「간주를 명시적으로 모델링」이 아니라 「간주 구간에서 음향모델이 헛디디지 않게 함」입니다. Kaldi 프레임워크가 제공하는 optional-silence 자기루프가 실질적으로 간주를 흡수하지만, 그건 프레임워크 기본값이고 **논문의 기여로 서술되지 않습니다** — 이건 제 추론이므로 그렇게 표시합니다.

**결과 (저희 문제와 같은 프로파일)**

| 구성 | Hansen-poly med/mean/%C | Mauch-poly med/mean/%C |
|---|---|---|
| C1 (솔로모델, 미적응) | 30.10 / 36.20 / 14.5 | 20.33 / 39.70 / 10.5 |
| C5 (poly 적응, MFCC+ivec) | 0.08 / 1.82 / 71.8 | 0.15 / 3.78 / 60.9 |
| C6 (poly 적응 + 5군 특징) | 0.11 / **2.37** / 64.7 | 0.18 / **1.93** / 57.5 |

중앙값 0.11~0.18초인데 평균이 1.9~2.4초입니다. 저자 본인의 결론:

> "there are hypothesized boundaries that are far away from the true boundaries, that needs to be investigated in future."

**코드·모델** 본 논문에는 URL 없음. 후속작의 도구 AutoLyrixAlign이 https://github.com/chitralekha18/AutoLyrixAlign 에 있습니다(Durand 2023이 실제로 이걸 받아 다국어 평가에 씀 — 즉 실행 가능한 공개물입니다). 영어 전용 발음사전·LM에 묶여 있습니다.

**①②③ 적용**
- **①에 직접 쓸 기법은 없습니다.** 「긴 간주 대응」을 기대하고 이 논문을 고르셨다면 얻을 것은 다른 것입니다.
- 대신 두 가지가 유효합니다. (a) **저희 실패 프로파일이 이 분야의 알려진 미해결 문제라는 확증** — median은 좋고 mean이 터지는 것이 정상 상태이며, 따라서 곡 단위 quality_score로 이상치 곡을 격리하는 저희 전략이 방향은 맞습니다. (b) **잡음강건 특징이 간주 구간 오발화를 줄인다**는 관찰 — 저희는 Demucs로 분리하니 이미 다른 경로를 타지만, 분리 잔향이 간주에 남아 star 대신 가사가 붙는 경우라면 원인 후보가 됩니다.
- Gupta 계열 전체가 **DNN-HMM + 학습 필요**라 저희 추론 전용 환경엔 이식 대상이 아닙니다.

**구현 비용** 도입하려면 Kaldi 스택 전체 + 학습. GPU 1대로 가능하나 **저희 「학습 안 함」 제약과 정면 충돌**. 권하지 않습니다.

---

# 3. Vaglio, Hennequin, Moussallam, Richard & d'Alché-Buc (ISMIR 2020)

**서지** Andrea Vaglio, Romain Hennequin, Manuel Moussallam, Gaël Richard, Florence d'Alché-Buc, "Multilingual Lyrics-to-Audio Alignment", ISMIR 2020, pp. 512–519.
**PDF** https://program.ismir2020.net/static/final_papers/101.pdf (HAL: https://hal.science/hal-02996940)

**방법**
- Spleeter로 보컬 분리 → **BLSTM 3층 + dense** 음향모델(입력: mel log filterbank + energy + delta + double-delta) → CTC → Viterbi forced alignment.
- 오디오를 5초 세그먼트(step 2.5s)로 잘라 posteriogram을 만들고, 각 조각을 **중앙 기준 절반으로 크롭해** 이어붙입니다.
- **문자 세트**: 라틴 알파벳 + apostrophe + **instrumental 토큰** + space 토큰 + blank ε = **30**.
  **음소 세트**: 보편 IPA 62개 + instrumental + space + blank = **65**.
- G2P는 Phonemizer(https://github.com/bootphon/phonemizer), IPA 중 모음·자음만 사용. 9개 언어 합쳐 보편 음소집합 62개, language sharing factor 5.35.
- forced alignment는 표준 CTC Viterbi(§3.3 식 1–6). Stoller와 같은 **균등노이즈 추가** 트릭을 씁니다(미학습 음소가 전 구간 0이 되는 것 방지).

**여기서 저희에게 중요한 설계 하나** — **blank와 별개인 명시적 instrumental 토큰**이 있고, 학습 세그먼트에 단어가 없으면 instrumental 라벨을 생성합니다. 이건 「간주는 blank가 알아서」가 아니라 **간주를 하나의 방출 토큰으로 지도학습**한 것입니다. 저희 star(log p=0, 균등)와 성질이 완전히 다릅니다: 학습된 instrumental 토큰은 프레임마다 **다른** 확률을 내므로 DP에 「여기가 간주다」라는 선호를 줍니다.

**문자 vs 음소 — 실제 결론**
- 음소가 문자보다 거의 항상 낫지만, **이득의 출처는 언어 간 전이**입니다. 원문: "The only models that are not improved are the ones trained and tested on the same languages."
- 다국어 학습 + 보편 음소집합이 zero-resource 언어까지 최고. 데이터 균형화(oversampling)는 저자원 언어를 개선하지 않고 영어만 크게 망쳤습니다.
- 영어 문자 모델 SOTA 대조: Hansen AAE 0.18s/PCO 95%, Mauch 0.22s/91%, Jamendo 0.37s/92%.

**비라틴 문자 처리** — **논문이 다루지 않습니다.** 실험 언어는 source(en, de, fr, es, it) + target(pt, pl, fi, nl)로 **전부 라틴 문자**입니다. 일본어·한국어·중국어·러시아어에 대한 실험이나 전사(romanization) 논의는 **없습니다**. 음소 시간을 다시 문자/단어로 되돌리는 매핑도 명시적 서술이 없고, 단어 시작시각만 평가합니다.

**코드·모델** 데이터 분할만 공개: https://github.com/deezer/MultilingualLyricsToAudioAlignment. 모델 미공개. 언어별 정확한 AAE/PCO 수치는 **supplementary materials에 있고 저는 확인 못 했습니다** (본문은 Figure 2 산점도만).

**①②③ 적용**
- **①에 직접**: 명시적 instrumental 토큰이 개념적 처방입니다. 단 저희는 학습을 안 하므로 **MMS 어댑터에 instrumental 토큰을 새로 넣을 수는 없습니다**. 이식 가능한 형태는 「star를 균등이 아니라 프레임별로 다른 점수를 갖게 만드는 것」이고, 학습 없이 그걸 할 재료는 저희가 이미 갖고 있습니다(VAD 에너지, Demucs 보컬 RMS, FCPE의 f0 유무·유성확률). **f0가 없는 프레임에서 star에 보너스, f0가 있는 프레임에서 star에 페널티** — FCPE는 이미 돌고 있으니 추가 연산이 거의 없고, VAD 하나에 매달렸던 ①의 단일점 의존도 깨집니다.
- **한국어 음소 전환에는 부정적 근거**입니다(§6에서 종합).

**구현 비용** 논문 방식 도입은 학습 필요. 위에 적은 「star 점수 성형」 파생 아이디어는 추론만, GPU 1대, 반나절~1일.

---

# 4. Teytaut, Bouvier & Roebel (Interspeech 2022) — 지목 질문 ①

**서지** Yann Teytaut, Baptiste Bouvier, Axel Roebel, "A study on constraining Connectionist Temporal Classification for temporal audio alignment", Interspeech 2022, pp. 5015–5019. DOI 10.21437/Interspeech.2022-10940.
**PDF** https://www.isca-archive.org/interspeech_2022/teytaut22_interspeech.pdf (HAL: https://hal.science/hal-03976279v1/document)

**문제의식 (저희와 동일)**

> "CTC measures by nature a transcription cost, therefore it can be minimized without guaranteeing alignment properties."

**구조** 16 kHz mono, Hamming 1024 / hop 256 → mel 128 bin → **F0-free MFCC 20계수**. 전부 conv 블록(BatchNorm + Conv1D 512필터 kernel 3 + dropout 0.2) + **self multi-head attention H=4**. 출력 posteriogram P ∈ [0,1]^{T×(L+1)} 및 MFCC 재구성 X̂. 4.5M 파라미터. **GTX 1080 Ti 1장**, 2.1시간 학습.

**손실 4개 (전부 학습시)**

1. `L_CTC = -log Σ_{ŷ∈B⁻¹(y)} Π_t P[t,ℓ]`
2. **엔벨로프 재구성** `L_REC = ||X - X̂||₁` — 스펙트로그램이 아니라 **MFCC(엔벨로프)**를 복원. 이유: |S| 복원은 F0를 네트워크로 통과시켜야 하고 F0는 정렬에 무관.
3. **시간구조 불변** `L_STR = ||S - Ŝ||₁`, S/Ŝ는 |S|와 최종 CTC dense층의 코사인 자기유사도행렬(4×4 average pool, stride 2×2 → T/2 × T/2).
4. **guided monotony** — 핵심입니다.
   `D[t,m] = exp( -( t/T - m/M )² / 2σ² )`, **σ = 0.1**
   `L_DIA = || D ⊙ softmax(P yᵀ) - D ||₁`

**손실 스케일링이 결정적입니다.** 각 손실을 최악값 분석으로 T에 선형이 되도록 정규화: `L_CTC ← L_CTC/log(L+1)`, `L_REC ← L_REC/F`, `L_STR ← (4/T)L_STR`, `L_DIA ← L_DIA/(2σM)`. 최종 `L = L^n_CTC + (1/3)Σ δ_i L^n_i`. 스케일링 없이 조합하면 MAE가 127~374ms로 폭발하고, 스케일링하면 22~40ms가 됩니다 — 원문 "changing the MAE's order of magnitude".

**결과** 최선(CTC + 스케일된 3제약): **TIMIT 음성 22.6ms, DIMITRIOS 노래 29.8ms**. 참조 [18](Teytaut&Roebel 2021 RNN, 48M 파라미터, 4.8h): 20.6ms / 35.8ms. 즉 노래에서는 이 논문이 더 좋습니다. 단독 SIM 제약은 무효("posteriograms can be trickily shaped to have correct structures yet without predicting the full duration of labels").

**데이터** TIMIT(깨끗한 솔로 영어 음성 5h), DIMITRIOS(솔로 그리스 비잔틴 창법 3h). **둘 다 무반주 솔로**. 저자 스스로 §4.4에서: "A major challenge is to apply our model on complete songs. Non-a cappella recordings are much harder to process."

**코드·모델** 논문에 URL 없음. TF 2.6 기반이며 CTCModel(hal-02420358)에서 착안했다는 서술만. 공개 체크포인트 **확인 못 함**. 저자 박사논문(https://theses.hal.science/tel-04229423v1/file/TEYTAUT_Yann_these_2023.pdf)에 더 자세한 내용이 있을 텐데 **읽지 않았습니다**.

**리드님의 ③ 실패와 어떻게 다른가**

| | 리드님 시도 | Teytaut L_DIA |
|---|---|---|
| 부호 | 부정(금지) | **양성** |
| 값 | 하드 -inf | 소프트 [0,1] 가우시안 |
| 대상 | emission 마스킹 | 정렬행렬 `softmax(PyT)`를 D에 맞추도록 |
| 적용 시점 | **추론** | **학습 전용** ("exclusively exploited during training, so that phoneme sequences are not inputs of the model at inference time") |
| 정보원 | 자막 앵커(외부) | **균등 대각선**(앵커 아님) |

**답: 양성 제약이 맞습니다. 그러나 (a) 학습 전용이고, (b) 앵커가 아니라 「모든 라벨이 등간격」이라는 균등 사전확률입니다.** 저자도 이 한계를 §4.4 첫 항목으로 인정합니다 — "pseudo-diagonal matrix D... carries a prior that all labels have similar duration, which is intrinsically not true. One could investigate a phoneme-informed or duration-focused approach."

**추론 전용 환경에서 쓸 수 있는가**: **논문 그대로는 못 씁니다.** 다만 **D의 수식 자체는 학습과 무관한 순수 함수**이고, 저희는 논문이 갖지 못한 것(자막 앵커 시각)을 갖고 있습니다. `t/T`와 `m/M`을 균등 비례가 아니라 **앵커 기반 단조 시간매핑**으로 바꾸면, D는 정확히 리드님이 필요하다고 지목한 「앵커 시각을 사전확률로 쓰는 양성 제약」이 됩니다. σ=0.1(정규화 시간 단위)이 출발 하이퍼파라미터입니다. **이 이식은 논문의 주장이 아니라 제 제안임을 분명히 해 둡니다.**

**구현 비용** 논문 재현 = 학습 필요(1080 Ti 2.1h 규모라 3090 1대로 충분하나, 저희는 학습을 안 함). D를 추론시 사전확률로 이식하는 것 = 추론만, 1~2일.

---

# 5. Durand, Stoller & Ewert (ICASSP 2023) — 저희에게 가장 중요한 논문

**서지** Simon Durand, Daniel Stoller, Sebastian Ewert, "Contrastive Learning-Based Audio to Lyrics Alignment for Multiple Languages", ICASSP 2023. arXiv:2306.07744.
**PDF** https://arxiv.org/pdf/2306.07744

**구조**
- **오디오 인코더**: 스펙트로그램 5초, `y → log(1+y)`, **11025 Hz, FFT 512, hop 256**. Residual conv block 10개(각각 GroupNorm + ReLU + Conv2D 3×3 64feat ×2, residual) → 1D conv로 주파수축 제거 → **A ∈ R^{T×64}**. **수용영역을 의도적으로 930ms로 작게** 유지.
- **텍스트 인코더**: 심볼 임베딩 → 각 심볼 s_n에 대해 `(s_{n-C},…,s_n,…,s_{n+C})` 부분열을 처리(FC 1층, 언어조건부면 3층) → **L ∈ R^{N×64}**. 양쪽 다 L2 정규화.
- 1.2M 파라미터, **총 4.8MB**.

**대조손실**
```
m(X,s) = max_t  f_ℓ(s) · f_a(X)_tᵀ
L = E_{(X,s⁺)~p_d} [ (m(X,s⁺) - 1)² + E_{s⁻~p_s} m(X,s⁻)² ]
```
양성 = 그 오디오 구간 가사에 등장하는 심볼, 음성 = 데이터셋 내 다른 가사에서 뽑은 미등장 심볼(예당 **1000개**). 순서 정보를 안 쓰는 **bag-of-symbols 약지도**입니다. 자기회귀 seq2seq를 일부러 피한 이유: 강한 언어모델이 음향모델을 억누르면 정렬에 해롭다는 초기 실험.

**디코딩 — 여기가 핵심입니다 (§2.4)**
1. `S = ½(A·Lᵀ + 1)`, `S ∈ [0,1]^{T×N}`.
2. 1차: S 위에서 **누적 유사도를 최대화하는 단조 경로**를 찾음.
3. 1차 결과로 각 줄의 구간을 추정: **t_c = 그 줄 중간 토큰의 추정 시작시각**(원문 "which we found to be robust to outliers"), **t_d = 줄의 토큰 수 × d**, d는 문자 **0.2s** / 음소 **0.4s**. 구간은 `t_s = t_c - (t_d - d)/2`, `t_e = t_c + (t_d + d)/2`.
4. **line-mask M ∈ [0,1]^{T×N}** 을 만들어 각 토큰을 그 줄의 추정 구간 근처에 몰아넣음. 줄 경계에는 **길이 2.5s의 선형 완충창**(파라미터에 robust하다고 명시).
5. 2차: **S ∘ M** (Hadamard product)에 **같은 디코딩을 다시** 실행.

> "Note that M does not require any additional training or external system, as opposed to using an external vocal activity or boundary detection module, and is really fast to obtain."

**저희 ③에 대한 직접 증거 — Table 1**

| 모델 | 손실 | 토큰 | C | line-mask | AAE | PCO |
|---|---|---|---|---|---|---|
| M1 | **CTC** | 음소 | 0 | 자기 마스크 | **0.90** | 86 |
| M2 | **CTC** | 음소 | 0 | **좋은 마스크로 교체** | **0.20** | 89 |
| M3 | Sim | 음소 | 0 | ✓ | 0.39 | 89 |
| M5 | Sim | 음소 | 1 | ✓ | 0.16 | 93 |
| M6 | Sim | 문자 | 1 | ✓ | **0.15** | 92 |
| M7 | Sim | 문자 | 1 | **✗** | 0.24 | 91 |

**같은 CTC 모델이 마스크 품질만으로 0.90 → 0.20 (4.5배)** 입니다. 저자 해석: "This suggests CTC approaches rely heavily on such additional constraints." 그리고 마스크를 빼면(M7) AAE가 0.15 → 0.24로 유의하게 나빠집니다 — "the mask M indeed helps removing outliers."

**다국어 (Table 3)** 영어만 학습: M5(음소) All AAE **1.11** vs M6(문자+문맥) All **0.35**. 전 언어 학습 + 언어조건부 M6: All **0.18**/PCO 94 (EN 0.21, ES 0.13, DE 0.16, FR 0.19). 원문: "This highlights the risk of error propagation of phoneme models if we try to extend the scope to additional languages, and that the performance is limited by the performance of the external phoneme representation."

**학습 필요 여부** 필요합니다. **87,785곡, epoch당 약 400시간 오디오, 최대 100 epoch × 20,000 iteration**, ADAM lr 0.001, 조기종료 patience 20. 감독 수준은 **줄 단위 시작·종료 시각**. GPU 사양·학습시간은 논문에 **없습니다(확인 못 함)**.

**공개 체크포인트** **없습니다.** 공개한 것은 JamendoLyrics Multi-Lang의 **수작업 단어 단위 정렬 주석**뿐(https://github.com/f90/jamendolyrics). 모델 코드·가중치 공개 서술 없음.

**①②③ 적용**
- **③의 정답입니다.** 리드님 실패와의 차이는 부호만이 아니라 **형태 3가지**입니다. (a) 하드 -inf가 아니라 **[0,1] 소프트 곱셈 가중**, (b) 「없다」가 아니라 **앵커 중심 선호 분포**, (c) **1패스 → 사전확률 → 2패스** 되먹임. 저희는 (c)의 인프라를 이미 갖고 있습니다 — `C:\devat\everyric2\everyric2\config\settings.py:344` 의 「줄의 프레임 창 위에서 `F.forced_align` DP를 다시 돌리고 **모델 forward는 없음**」 경로입니다.
- **①에 직접**: 자막 앵커가 있으면 1차 디코딩을 건너뛰고 t_c를 앵커에서 바로 얻을 수 있습니다. Durand보다 유리한 조건입니다. 간주 8.7~24.9s에 8줄이 몰리는 배치는 마스크 안에서 애초에 실행 불가능해집니다.
- **②에 간접**: 마스크가 탐색공간을 줄이면 근동률 경로 수가 줄어 부동소수점 뒤집힘이 줄어듭니다. 다만 이건 완화이고 근본 처방은 아닙니다(§6 참조).

**CTC로 옮길 때 논문이 답하지 않는 설계 하나** — Durand의 S에는 **blank도 star도 없습니다**(순수 유사도행렬). 저희 emission에는 blank와 star 열이 있고, M을 그 열에 어떻게 적용할지는 논문 밖입니다. 제 권고: **가사 토큰 열에는 M을 로그 도메인 가산 보너스로, star 열에는 M의 여집합을 보너스로** 주십시오. star가 log p = 0으로 고정된 상태에서 가사 토큰만 올려도 star가 여전히 모든 프레임을 이기므로, **star 점수도 함께 성형해야 합니다.** 이건 제 도출이며 논문의 주장이 아닙니다.

**구현 비용** 모델 재현 = 학습 필요, 저희 제약과 충돌. **line-mask 디코딩 이식 = 추론만, 학습 0, GPU 1대, 2~3일.** 6편 중 비용 대비 효과가 압도적으로 최고입니다.

---

# 6. Cheng, Nakano & Goto (DAFx 2025)

**서지** Tian Cheng, Tomoyasu Nakano, Masataka Goto, "Improving Lyrics-to-Audio Alignment Using Frame-wise Phoneme Labels with Masked Cross Entropy Loss", Proc. DAFx25, Ancona, Italy, 2025, pp. 342–349. (AIST)
**PDF** https://dafx.de/paper-archive/2025/DAFx25_paper_15.pdf (사본: https://staff.aist.go.jp/m.goto/PAPER/DAFX2025cheng.pdf)

**구조** **HT Demucs**로 보컬 분리(44.1kHz stereo → mono 16kHz), mel 128bin, window 1024 / **hop 256 (=62.5 fps)**, [0,1] 정규화. Teytaut&Roebel 2021 구조의 **단순화판 오토인코더**:
- spectral encoder = CNN block(Conv2D 3×3, 필터 16→32, BN + 절반 pooling + dropout 25%) + RNN block
- RNN block / CTC decoder / spectral decoder **전부 BLSTM 2층 × 512 units** 동일
- CTC decoder 출력 41차, spectral decoder 출력 128차(sigmoid)
- **attention을 일부러 제거**했습니다 — "we found that adding attention degraded alignment performance in our preliminary experiments". Teytaut와 반대 결과입니다.

**토큰** CMU 39음소 + **token 0 = blank ε("staying on the same phoneme")** + **token 40 = silence/space** = **41**. G2P는 https://github.com/Kyubyong/g2p.

**손실**
```
L = L_CTC + λ₁ L_REC + λ₂ L_maskedCE
L_REC = ||X̂ - X||₂   (실제로는 MSE)
L_maskedCE = -(1/N_mask) Σ_i Σ_t  B_{i,t} · L_{i,t} log D_{i,t}
```
**마스크 B 구성 (핵심 기여)** — 단어 단위 주석에서 프레임 라벨을 **부분적으로** 유도:
1. 음소 2개 이상 단어: **onset 프레임에 첫 음소, offset 프레임에 마지막 음소**
2. 음소 1개 단어: **그 단어 구간 전 프레임에 같은 음소**
3. 어떤 단어에도 속하지 않는 프레임: **silence(token 40)**

B는 `0_{41×T}`에서 시작해 마스크 프레임의 `i ∈ [1..40]`을 1로(blank 제외), **추가로 onset 프레임에서만 `B_{0,t}=1`** 로 둡니다. 이유가 정확히 저희 문제입니다 — onset에서는 라벨이 절대 blank(="같은 음소 유지")일 수 없으므로 **blank의 부재를 명시적으로 강제**합니다.

**결과 (Jamendo, 설정별 10 seed 평균)**

| λ₁ (REC) | λ₂ (maskedCE) | MAE | MedAE | PCO0.3 | PCO0.2 |
|---|---|---|---|---|---|
| 0 | 0 (베이스라인) | 0.247 | 0.049 | 93.8 | 90.9 |
| 1 | 0 | 0.248 | 0.047 | 94.6 | 91.8 |
| 0.1 | 0 | 0.283 | 0.056 | 91.7 | 87.8 |
| 0.01 | 0 | 0.269 | 0.055 | 92.2 | 88.1 |
| 0 | 1 | 0.220 | 0.044 | 95.0 | 94.1 |
| **1** | **1** | **0.216** | **0.041** | **95.2** | **94.3** |

**주목할 점: 재구성 손실 단독은 MAE를 개선하지 못했습니다**(0.248 / 0.269 / 0.283 vs 베이스라인 0.247). 효과는 masked CE에서 나옵니다. Teytaut의 L_REC를 저희가 이식할 가치가 있는지 판단할 때 이 결과가 중요합니다.

SOTA 대조: MedAE 0.041 / PCO0.3 95.2% / PCO0.2 94.3%로 **in-house 대규모 학습 모델까지 전부 능가**. MAE 0.216은 DSE(Durand, 88k곡) 0.15, HX-IH(Kang, 한국어+영어 67k곡) 0.16보다 나쁨.

**저희와 동일한 실패 프로파일과, 저자의 처방**

> "our alignment results... contained a large proportion of small absolute errors (as reflected by the good MedAE and PCO results), but also **several outliers with large absolute errors** that contributed to the relatively high MAE. **We expect that incorporating a line-level alignment stage could help reduce these outliers.**"

Durand와 독립적으로 같은 결론에 도달했습니다.

**추론 절차** **CTC-segmentation**(Kürzinger 2020), trellis `k_{t,m} = max(P_stay, P_transit)`, `P_stay = k_{t-1,m}·P(ε|t)`, `P_transit = k_{t-1,m-1}·P(y_m|t)`. 구현은 **torchaudio forced alignment 튜토리얼 그대로** — 저희와 같은 DP입니다.

**전처리 (추론 전용, 즉시 이식 가능)** 곡 앞뒤 무음 제거: mel을 주파수축으로 합해 프레임 에너지 벡터를 만들고 최대 1로 정규화, **임계값 0.05** 초과 프레임의 첫/마지막을 잡아 **양쪽으로 1초(63프레임) 확장**한 구간만 정렬에 사용. 그리고 곡 전체를 한 번에 입력합니다(줄 단위 분할 없이).

**코드·모델** 논문에 URL **없습니다**. 공개 여부 **확인 못 함**.

**①②③ 적용**
- **②(비결정성)에 대한 답: 직접적으로는 도움이 안 됩니다.** 이것은 **학습시 손실**이고, 그들의 10-seed 분산은 **학습 초기화/dropout 분산**이지 추론 비결정성이 아닙니다. 논문에 추론 재현성 논의는 없습니다. 다만 간접 경로는 실재합니다 — masked CE는 onset에서 blank 부재를 강제해 **posterior를 첨예화**하므로, 원리적으로는 축퇴(=②의 원인)를 줄입니다. 저희가 학습을 하지 않는 한 이 경로는 닫혀 있습니다.
- **①에 간접**: blank와 **분리된 silence 토큰을 지도학습**하고 비단어 프레임을 전부 silence로 라벨링하는 설계는 Vaglio의 instrumental 토큰과 같은 방향입니다. 학습 없이는 이식 불가.
- **즉시 이식 가능**: 앞뒤 무음 제거 전처리. T를 줄여 축퇴 경로 수를 줄이고, 곡 시작 전 star 흡수 구간을 없앱니다. 반나절.

**구현 비용** 논문 재현 = 학습 필요, 규모는 작습니다(DALI 영어 2,681곡, batch 32, 20 epoch, RMSprop lr 1e-4). **3090 1대로 현실적**입니다 — 6편 중 저희가 실제로 학습해 볼 수 있는 유일한 후보입니다. 단 「학습 안 함」 정책을 바꿔야 하고, MMS 다국어 어댑터를 버리고 CMU 영어 39음소 계열로 가는 것이므로 **일본어·한국어에는 그대로 못 씁니다**.

---

# 7. 지목 질문에 대한 정리된 답

**Q1. Teytaut 2022의 CTC 제약이 저의 ③ 실패와 어떻게 다른가? 양성 제약인가? 추론 전용에서 쓸 수 있나?**
양성 제약입니다(가우시안 대각 사전확률 D, σ=0.1). 리드님 것과의 차이는 **부호 + 경도(hard/soft) + 적용 시점** 세 가지입니다. 그러나 **학습 전용**이고("exclusively exploited during training"), 앵커가 아니라 **모든 라벨이 등간격이라는 균등 사전확률**입니다 — 저자도 이 한계를 향후과제 1번으로 적었습니다. **논문 그대로는 추론에 못 씁니다.** D의 수식만 떼어 `t/T ↔ m/M` 을 앵커 기반 단조 시간매핑으로 교체하면 추론시 양성 사전확률이 됩니다(제 제안). 다만 **③의 검증된 처방은 Teytaut이 아니라 Durand의 line-mask**입니다 — 후자는 학습이 전혀 필요 없고 CTC에서 0.90 → 0.20의 실측 증거가 있습니다.

**Q2. Gupta 2019의 긴 간주 대응을 ①에 직접 쓸 수 있나? 구체적 기법은?**
**기법이 없습니다.** 이 논문에는 간주 전용 장치(VAD 제약, garbage/filler 모델, 디코딩 그래프 개조)가 하나도 없습니다. 있는 것은 「RASTA 계열 잡음강건 특징 + 폴리포닉 도메인 적응이 결과적으로 긴 간주 옆 경계를 개선했다」는 관찰뿐이며, 저자는 큰 이상치를 미해결로 남깁니다. ①에 직접 이식할 것은 없습니다.

**Q3. Vaglio 2020의 음소 중간표현으로 바꾸면 한국어가 실제로 나아지는가? 받침 분해의 이득과 비용, 글자 단위 타임스탬프 매핑은?**
**권하지 않습니다.** 근거 3개:
- Vaglio 본인의 결과: 음소의 이득은 **언어 간 전이**에서 나오고, **같은 언어로 학습·평가하면 개선 없음**("The only models that are not improved are the ones trained and tested on the same languages"). 저희는 한국어 곡을 한국어로 정렬합니다 — 이득이 나오는 조건이 아닙니다.
- Durand의 더 강한 반증: 문맥창 문자(0.15s) ≈ 음소(0.16s), 그리고 **다국어로 확장하면 음소가 훨씬 나쁨**(1.11 vs 0.35). "risk of error propagation of phoneme models... performance is limited by the performance of the external phoneme representation." 한국어 G2P(연음·경음화·비음화)가 그 external representation이고, 저희는 이미 발음 파이프라인에서 조수사 음변화 같은 문제를 겪었습니다. 정렬 경로에 G2P 오류를 새로 들이는 것입니다.
- **두 논문 모두 일본어·한국어를 실험하지 않았습니다.** Vaglio의 9개 언어는 전부 라틴 문자입니다. 한국어에 대한 직접 근거는 **어느 논문에도 없습니다**.

비용 쪽도 명확합니다. 받침을 분해하면 「간」 → /k/ /a/ /n/ 로 토큰이 2~3배가 되고, 카라오케용 글자 타임스탬프는 **첫 음소의 시작 + 마지막 음소의 끝**으로 다시 합쳐야 합니다. 이 재합성 매핑은 **Vaglio도 Durand도 명시적으로 서술하지 않습니다**(둘 다 단어 시작시각만 평가). 즉 이득은 근거 없고, 토큰 길이 증가로 DP 축퇴(②)는 악화되고, 매핑 규칙은 저희가 새로 발명해야 합니다. **현행 음절 단위 유지가 맞습니다.**

**Q4. Durand 2023의 contrastive는 학습이 필요한가? 사전학습 체크포인트가 공개되어 있나?**
학습 필요합니다 — **87,785곡, epoch당 약 400시간, 최대 100 epoch × 20,000 iteration**, 감독은 줄 단위 시작·종료. **체크포인트·코드 공개 없습니다.** 공개된 것은 JamendoLyrics Multi-Lang의 단어 단위 평가 주석뿐. **그러나 저희가 필요한 부분(line-based decoding, §2.4)은 모델과 무관하고 학습이 전혀 필요 없습니다** — 원문이 "does not require any additional training or external system"이라고 명시합니다. 이게 이 과제의 핵심 소득입니다.

**Q5. Cheng 2025의 프레임 단위 완화가 ②(비결정성)에도 도움이 되나?**
**직접적으로는 아닙니다.** 학습시 손실이고, 논문의 10-seed 분산은 학습 분산이며, 추론 재현성 논의가 없습니다. 원리적으로는 onset에서 blank 부재를 강제해 posterior를 첨예화하므로 축퇴를 줄이지만, **저희가 학습을 하지 않는 한 이 경로는 쓸 수 없습니다.**
②의 실제 원인은 다른 곳입니다. `F.forced_align`의 DP는 **동일 emission에 대해 결정적**입니다(저희 `ctc_engine.py:261` 주석도 「모델 forward가 없다」고 적고 있습니다). 흔들리는 것은 **emission**이고, 원인은 GPU forward의 비결정성(TF32 허용, cuDNN 알고리즘 선택, 리덕션 순서)입니다. 그것이 축퇴된 DP에 증폭되어 21.74초로 나타납니다. 처방은 두 겹입니다 — (a) forward를 결정화(`torch.use_deterministic_algorithms(True)`, TF32 비활성, cuDNN benchmark 고정), (b) 양성 마스크로 축퇴 자체를 줄이기. **(a)는 논문에서 온 것이 아니라 엔지니어링 판단임을 밝힙니다.** 6편 중 추론 비결정성을 다룬 논문은 없습니다.

---

# 8. 도입 순서 추천

| 순위 | 무엇 | 출처 | 학습 | 대상 | 규모 |
|---|---|---|---|---|---|
| 1 | star 점수 성형(앵커·f0 기반) | Vaglio instrumental 토큰의 추론판 + 제 도출 | 불필요 | ① | 반나절 |
| 2 | 앞뒤 무음 제거 전처리 | Cheng §2.4.1 | 불필요 | ①② | 반나절 |
| 3 | emission forward 결정화 | 논문 아님 | 불필요 | ② | 1일 |
| 4 | Durand 2패스 soft line-mask | Durand §2.4 | 불필요 | ①③ | 2~3일 |
| 5 | emission 균등노이즈 플로어 + 상수 지연 캘리브레이션 | Stoller §3.3 | 불필요 | 안정성 | 1일 |
| 6 | Teytaut D를 앵커 기반으로 일반화해 추론 사전확률로 | Teytaut §3.4 + 제 이식 | 불필요 | ①③ | 1~2주 |
| 7 | Cheng masked CE로 자체 학습 | Cheng §2.3 | **필요** | ①② 근본 | 수 주, 정책 변경 |

7번은 「학습 안 함」 정책을 바꿔야 하고, MMS 다국어 어댑터를 포기하고 영어 CMU 음소로 가는 것이라 **일본어·한국어에 그대로 쓸 수 없습니다.** 6편 중 3090 1대로 현실적인 유일한 학습 후보이긴 하나, 지금 우선순위는 아닙니다.

---

# 9. 지금 당장 시도할 것 3개 (비용 순)

### 1) star 점수를 균등에서 앵커·f0 기반으로 성형 — 반나절, 추론만, 학습 0

①의 근본 원인은 star가 `log p = 0`으로 **어디서 흡수할지 선호가 0**이라는 것입니다. 금지(-inf)를 걷어내고, star 열에 **프레임별로 다른 로그 점수**를 주십시오.

- 앵커 줄 구간 **안**: star에 음의 값(예: −α) → 가사가 그 구간을 채우는 것이 유리해짐
- 앵커 갭(간주) 구간: star는 0 유지 → 흡수가 유리
- 앵커가 없는 곡: FCPE의 유성/f0 유무를 대리 신호로 사용(이미 돌고 있어 추가 연산 없음). VAD 단일 신호 의존이 여기서 깨집니다 — ①에서 방어장치 4개가 전부 한 신호에 매달렸던 문제의 직접 해소입니다.

**가사 토큰만 올리는 것으로는 부족합니다.** star가 p=1인 한 여전히 모든 프레임을 이기므로 **star 쪽을 반드시 눌러야** 합니다. 이게 리드님의 부정 제약이 실패한 기계적 이유이기도 합니다 — emission을 막아도 star는 건드리지 않았으니 DP는 star로 도피할 자유를 그대로 유지했습니다. α는 첫 실측으로 1.0~3.0(nat) 범위를 훑으십시오.

### 2) 곡 앞뒤 무음 제거 전처리 — 반나절, 추론만 (Cheng §2.4.1 그대로)

mel을 주파수축 합 → 최대 1 정규화 → **임계 0.05** 초과 프레임의 첫/마지막 → **양쪽 1초 확장** → 그 구간만 정렬. Cheng이 곡 단위 정렬의 「robustness」를 위해 명시적으로 넣은 단계입니다. T가 줄면 축퇴 경로 수가 줄어 ②가 완화되고, 곡 시작 전 구간에서 star가 엉뚱하게 흡수하는 경우가 사라집니다. 리스크가 거의 없고 되돌리기 쉽습니다.

### 3) Durand의 2패스 soft line-mask — 2~3일, 추론만, 학습 0 (효과 최대)

③의 검증된 정답입니다. **하드 금지를 소프트 선호로 바꾸는 것**이 전부입니다.

1. 1패스: 현행 그대로 정렬(또는 자막 앵커가 있으면 **1패스를 건너뛰고 앵커에서 t_c를 직접** — Durand보다 유리한 조건입니다)
2. 줄별 구간 추정: `t_c` = 그 줄 **중간** 토큰의 시작시각(원문이 이상치에 강하다고 명시 — 첫 토큰이 아니라 중간입니다), `t_d` = 토큰 수 × d. 저희는 한글 음절이므로 d의 출발값은 Durand의 문자 0.2s보다 크게 잡아야 합니다(음절 ≈ 영문 문자 2~3개분) — 실측으로 정하십시오.
3. **soft mask M ∈ [0,1]**: 줄 구간 안은 1, 경계에서 **2.5s 선형 완충**으로 0까지 내림. `-inf` 절대 금지 — 저희 `ctc_engine.py:300` 주석이 기록한 「전 구간 -inf면 실행 불가」와 리드님의 11.22 → 22.58초 악화가 같은 원인입니다.
4. 로그 도메인에서 emission에 `log M`을 **가산**하고, **star 열에는 M의 여집합을 가산**(설계 결정 — Durand의 S에는 blank/star 열이 없어 논문이 답하지 않는 부분입니다)
5. 2패스: 같은 `F.forced_align`을 다시 실행. `settings.py:344`의 「줄 프레임 창 위 DP 재실행, 모델 forward 없음」 경로가 이미 이 인프라이므로 새로 만들 것이 적습니다.

기대치의 근거: 같은 CTC 모델에서 마스크 품질만으로 **AAE 0.90 → 0.20**(M1 → M2), 마스크 제거시 **0.15 → 0.24**(M6 → M7). 다만 M2의 마스크는 더 좋은 모델에서 온 것이라 **마스크 품질이 결과를 지배합니다** — 저희에게는 자막 앵커 품질이 그 자리입니다. 자막이 부정확한 곡에서는 완충창을 넓히거나 마스크를 끄는 폴백이 필요합니다.

---

# 10. 확인 못 함 (추측으로 채우지 않은 항목)

- **Teytaut 2022 공개 코드·체크포인트**: 논문에 URL 없음. TF2.6 기반이며 CTCModel(hal-02420358) 착안이라는 서술만. 저자 박사논문 `tel-04229423` 은 **읽지 않았습니다** — guided monotony의 확장(informed D)이 거기 있을 가능성이 있어 필요하면 후속 조사 대상입니다.
- **Teytaut Figure 4의 설정별 정확한 수치 대응**: 추출 텍스트에서 8개 설정 × 4개 그룹의 값 배치가 모호합니다. 본문에 명시된 값(최선 22.6ms/29.8ms, 참조 20.6ms/35.8ms, 미스케일 시 127~374ms, 4.5M vs 48M 파라미터, 2.1h vs 4.8h)만 인용했습니다.
- **Vaglio 2020 언어별 정확한 AAE/PCO 수치**: supplementary materials에 있고 접근하지 않았습니다. 본문은 Figure 2 산점도뿐입니다. 인용한 표(Hansen 0.18/95, Mauch 0.22/91, Jamendo 0.37/92)는 Table 2의 SOTA 대조표입니다.
- **Cheng 2025 코드·모델 공개**: 논문에 URL 없음. 별도 공개 여부 확인하지 않았습니다.
- **Durand 2023 학습 GPU 사양·소요 시간**: 논문에 없습니다.
- **Gupta의 duration-based modified pronunciation lexicon 상세**: 참조 [10]에 있고 본 논문에는 없습니다. [10]은 읽지 않았습니다.
- **`jhuang448/LyricsAlignment-MTL`**: 검색 결과로 **BDR(boundary detection) 체크포인트를 포함한 공개 저장소**가 있습니다(Huang, Benetos & Ewert ICASSP 2022 = Durand 표의 HBE, AAE 0.23). 경계 모델 체크포인트가 실제로 공개되어 있다면 **자막 없는 곡의 앵커 공급원**이 될 수 있어 저희에게 직접 유용합니다. 다만 이는 지정 6편 밖이고, **저장소 내용·동작·라이선스를 확인하지 않았습니다.** 별도 조사를 권합니다.
- 저는 파일을 수정하거나 커밋하지 않았고, 원격 서버에 접속하지 않았으며, 무거운 로컬 연산을 돌리지 않았습니다(PDF 텍스트 추출만 PyMuPDF로 수행).
