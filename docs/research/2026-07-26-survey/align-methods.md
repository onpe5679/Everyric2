# align-methods — CTC 정렬 한계·대안 논문 조사 (SendMessage 전문 1건)

지정하신 3편 모두 원문(PDF) 확보·확인했습니다. 후속연구, 간주/무음 처리, 비결정성, 한국어/일본어 관련도 조사했습니다. 서지정보는 논문 원문에서 직접 확인한 것만 적었고, 확인 못 한 항목은 명시했습니다.

## 1) 지정하신 3편

### Durand & Stoller & Ewert, 2023, ICASSP — "Contrastive Learning-Based Audio to Lyrics Alignment for Multiple Languages"
arXiv: https://arxiv.org/abs/2306.07744 (ICASSP 2023, Rhodes Island, Greece, pp.1-5)
- 핵심 주장 그대로 확인됨: "CTC systems use a loss designed for transcription which can limit alignment accuracy." 대안은 오디오-텍스트 cross-modal contrastive embedding(대조학습)이며 CTC/toolkit 기반이 아님.
- 수치: JamendoLyrics(영어) 기준 average absolute error < 0.2초. 영어만 학습해도 다국어에 견고.
- 저희 문제 ③(CTC 손실의 구조적 한계)에 정확히 대응하는 근거 논문. 다만 구현 난이도는 높음 — 저희 파이프라인 전체를 CTC 기반에서 대조학습 기반 임베딩 정렬로 바꿔야 해서 사실상 재설계. 공개 코드 저장소는 확인 못 함(공식 repo 못 찾음).

### Cheng, Nakano, Goto (AIST), 2025, DAFx25 (Ancona, Italy, Sep 2-5) — "Improving Lyrics-to-Audio Alignment Using Frame-wise Phoneme Labels with Masked Cross Entropy Loss"
PDF: https://dafx.de/paper-archive/2025/DAFx25_paper_15.pdf (pp.342-349)
- 방법: baseline은 CTC loss + reconstruction loss(오토인코더, Teytaut 2021 구조 기반 CRNN). 여기에 단어 시작/끝 프레임 및 무음 프레임에 대해 "masked frame-wise CE loss"를 추가 — word-level 주석에서 유도 가능한 부분적 프레임 라벨(단일 음소 단어는 전체 프레임, 다음소 단어는 첫/끝 프레임만, 무음은 전 구간)만 마스킹해서 CE loss로 보강.
- 수치: baseline MAE 0.247→0.216초(masked CE+reconstruction 조합), MedAE 0.041초(SOTA), PCO0.3 95.2%, PCO0.2 94.3%. DALI로 학습한 모델 중 최고 성능(대형 in-house 데이터 학습 모델 제외).
- 저희 문제 ①(간주 구간 오배치)에 가장 바로 적용 가능. 무음/간주 프레임을 명시적으로 "silence label"로 마스킹해 학습시키는 방식이라 저희의 `<star>` 토큰 흡수 방식과 결합 가능성 있음. **후처리(4.2.1절)로 "노래 시작/끝의 무음 구간을 에너지 임계값(0.05)으로 잘라내고 전후 1초 패딩" 하는 간단한 트리밍도 함께 제안** — 이건 구현 난이도가 매우 낮고 즉시 적용 가능. 구현 난이도: masked CE 학습법 자체는 저희가 학습 파이프라인이 없다면(사전학습 MMS 어댑터만 쓰는 추론 전용 구조라면) 적용 불가 — 학습 파이프라인 보유 여부에 따라 난이도가 "낮음(트리밍)" ↔ "높음(재학습 필요)"으로 갈림. 코드 공개 여부 확인 못 함(논문 내 repo 링크 없음, AIST 개인 페이지에서도 못 찾음).

### Vaglio, Hennequin, Moussallam, Richard, d'Alché-Buc (Deezer/Télécom Paris), 2020, ISMIR — "Multilingual Lyrics-to-Audio Alignment"
PDF: https://program.ismir2020.net/static/final_papers/101.pdf (pp.512-519)
- 방법: BiLSTM(3층) + CTC, 출력을 문자 대신 IPA 기반 "universal phoneme set"(62개 음소, 9개 언어 통합)으로 — 언어 독립적 중간표현. 다국어 혼합 학습셋(5lang: en/de/fr/es/it)으로 학습하면 저자원·zero-resource 언어(pt/pl/fi/nl)에도 일반화.
- 수치: Jamendo AAE 0.37초(문자 기반, 영어만 학습 시 SDE2와 비슷한 수준). 다국어+음소 조합이 모든 케이스에서 최고 성능. 언어공유계수(language sharing factor) 5.35 — 평균적으로 음소 하나가 5~6개 언어에 공유됨.
- 저희 문제 ③ 관련성은 간접적(정렬 손실 자체보다 언어 간 일반화가 초점)이나, 한국어·일본어처럼 학습 데이터가 상대적으로 적은 언어에 대한 전이학습 근거로 유용. MMS는 이미 다국어 어댑터라 이 논문의 통찰(범용 음소 중간표현)과 사상이 겹침 — MMS 자체가 이미 이 방향으로 진화한 것으로 볼 수 있음. 구현 난이도 낮음(개념적으로는 이미 MMS 어댑터가 흡수한 접근). 코드: https://github.com/deezer/MultilingualLyricsToAudioAlignment (데이터 분할 정보 공개, 학습 코드는 일부만).

## 2) 후속/인용 연구 (2023~2026)

### Kick, Grötschla, Lanzendörfer, Wattenhofer, 2025, ICASSP (Hyderabad, India, April 2025) — "Contrastive Lyrics Alignment with a Timestamp-Informed Loss"
IEEE Xplore: https://ieeexplore.ieee.org/document/10888807/ · OpenReview(워크숍 버전 추정): https://openreview.net/pdf?id=peM113bm8Z (본문 인증장벽으로 직접 못 읽음 — 아래는 검색 스니펫 기반, 원문 대조 못 함)
- Durand 2023의 직접 후속. "box loss"라는 타임스탬프 정보를 손실함수에 직접 반영하는 방법 제안, DALI 데이터셋 노이즈를 정제, JamendoLyrics++ (장르 다양성 확장 평가셋) 공개.
- 문제 ③에 대한 최신 대안. 구현 난이도: 논문 원문을 못 읽어 정확한 판단 어려움 — **확인 못 함** 표시. 코드/데이터셋 공개 여부도 확인 못 함.

### Wang, Olvera, Richard (Télécom Paris), 2025 — "Melody-Lyrics Matching with Contrastive Alignment Loss"
arXiv: https://arxiv.org/abs/2508.00123
- 이건 저희 문제와 직접 관련 없음(멜로디만으로 가사 후보를 검색하는 역매칭 문제, forced alignment 아님) — 참고용으로만.

### Kang, Park, Choi, 2023, arXiv (미게재 확인, 학회 불명) — "HCLAS-X: Hierarchical and Cascaded Lyrics Alignment System Using Multimodal Cross-Correlation"
arXiv: https://arxiv.org/abs/2307.04377
- 문장 단위 정렬 후 단어 단위로 세분화하는 계층적 접근(cross-correlation 기반, CE loss로 line/word onset 예측). Cheng 2025 논문의 비교표에 등장(HX-D: DALI 학습 MAE 0.42, HX-IH: In-house 67k 한국어+영어 학습 MAE 0.16). **저자 중 일부가 한국어 데이터(6.7만곡, 한국어+영어) in-house 학습 결과를 보고 — 한국어 가창 정렬에 대한 몇 안 되는 실측 사례**지만 in-house 데이터라 재현 불가.
- 문제 ①과 관련: 계층적(줄→단어) 정렬이 간주 구간 오배치를 줄이는 전략 중 하나로 Cheng 2025 논문의 관련연구에서도 언급됨(Demirel 2021과 함께 "hierarchical alignment strategy"로 분류).

### Less peaky and more accurate CTC forced alignment by label priors, arXiv:2406.02560 (Interspeech 2024로 추정, 저자 확인 못 함 — PDF 메타데이터 파싱 실패로 저자명 못 얻음, 재확인 필요)
arXiv: https://arxiv.org/html/2406.02560v3
- **문제 ③에 가장 실전적으로 적용 가능한 논문.** CTC의 "peaky" 현상(blank 토큰 과다 예측)이 정렬 부정확의 핵심 원인이라 보고, 디코딩 시 `P(π|X) = ∏ y_π^t / P(π_t)^α` 형태로 토큰별 사전확률(label prior)로 나눠 페널티를 주는 보정을 제안. α=0.3일 때 최적.
- 수치: Buckeye PBE 44ms→38ms, WBE 58ms→43ms (12~40% 개선). MFA(GMM-HMM)에는 못 미치지만 CTC 계열 중 최선.
- **저희 시스템과 직결**: "학습 레시피와 사전학습 모델이 TorchAudio를 통해 공개됨"이라고 자체 명시 — 저희가 이미 쓰는 `torchaudio.functional.forced_align` 생태계와 같은 라인. 디코딩 시점에 label prior로 나누는 후처리 보정이라 **재학습 없이 추론 코드만 수정하면 적용 가능할 가능성 있음(난이도 낮음~중간)** — 다만 label prior를 우리 데이터(일본어/한국어 가창)에서 어떻게 추정할지는 별도 검증 필요. 비결정성과의 직접적 언급은 논문에 없음(제가 확인 못 함으로 명시).

## 3) 비결정성(problem ②) — 결론: 전용 논문을 찾지 못했습니다

"forced alignment의 비결정성/재현성"을 직접 다룬 논문은 검색으로 찾지 못했습니다(**확인 못 함** — 존재하지 않는다는 뜻은 아니고, 제가 못 찾았다는 뜻입니다). 대신 인접 근거:

- **일반 ML 재현성 문헌**: arXiv:2408.05148 "Impacts of floating-point non-associativity on reproducibility for HPC and deep learning applications" (2024) — 부동소수점 비결합성이 GPU 병렬 리덕션 순서에 따라 결과를 바꾸는 메커니즘의 일반론. arXiv:2001.11396 "Non-Determinism in TensorFlow ResNets" — GPU 비결정성이 정확도 표준편차의 74%, 손실 표준편차의 87% 이상을 차지한다는 실측. 두 논문 모두 forced alignment 특정 사례는 없음.
- **Thinking Machines Lab 블로그(비논문, 2025)**: "Defeating Nondeterminism in LLM Inference" — 핵심은 "배치 크기가 요청마다 달라져서 리덕션 순서가 바뀌고, 이게 낮은 확신도 상황에서 출력을 뒤집는다"는 배치 불변성(batch invariance) 결핍 문제. **다만 이 해법(batch-invariant kernel)은 순차적 동적계획법(Viterbi)에는 구조적으로 잘 안 맞습니다** — LLM 추론처럼 배치 차원에서 최적화할 여지가 Viterbi 순전파(forward pass)에는 없어서, 직접 이식은 어렵다고 판단됩니다.
- **CTC peakiness 문헌**(위 2406.02560 포함)이 간접적으로 관련: posterior가 flat(=quality_score 낮음)할 때 Viterbi 경로가 근소한 확률 차이로 갈리는 것은, peaky하지 않은 CTC posterior의 본질적 특성으로 볼 수 있습니다. 즉 "비결정성이 심한 것 = peakiness가 낮은 것"이라는 저희 관찰(quality_score 낮을수록 편차 큼)은 peakiness 관련 문헌과 정합적이지만, 이를 직접 실험으로 검증한 논문은 못 찾았습니다.
- **실무적 대응**: `torch.use_deterministic_algorithms(True)`로 완전 결정성 보장은 안 됨(`torch.cumsum` 등 일부 연산은 deterministic 모드에서도 비결정적이라는 PyTorch 공식 문서 확인). CUDA 버전/하드웨어 간 재현성도 PyTorch가 보장 안 함.

**결론**: 이 문제는 저희가 직접 실증 발견한 현상에 가깝고, 기존 문헌은 "왜 이런 일이 생기는가"에 대한 인접 이론(부동소수점 비결합성, CTC peakiness)만 제공합니다. 자체 실험 노트(21.74초 편차, quality_score 상관관계)가 오히려 선행 사례가 드문 관찰일 수 있습니다 — 논문화 가치가 있을 수도 있다는 뜻입니다.

## 4) 간주/무음 구간 처리(problem ①)

- **Cheng 2025**: 위 참조. 무음 프레임 명시적 라벨링 + 곡 시작/끝 무음 트리밍(에너지 임계값 0.05, 전후 1초 패딩).
- **Demirel, Ahlbäck, Dixon, 2021, arXiv:2108.02625 (originally MTDNN 계열, ALT 논문) — "MSTRE-Net: Multistreaming Acoustic Modeling for Automatic Lyrics Transcription"**: 원문 확인 완료. 모노포닉(DAMP)/폴리포닉(DALI) 각각에 별도 `<silence>`/`<music>` 토큰을 명시적으로 부여하고 가사 앞뒤에 태깅해서 학습 — 저희의 단일 `<star>` 토큰과 달리 **무음과 반주(음악)를 구분하는 토큰을 별도로 둔 것**이 핵심 차이. 수치: DALItest에서 WER 53.86%→47.00%(음악/무음 태깅 추가만으로 6.86%p 개선, Table 3). 폴리포닉에서만 개선, 모노포닉에서는 무효과 — 즉 **간주 처리는 폴리포닉(반주 있는) 상황에서만 유효하다는 것을 실측으로 보여줌**. 저희 상황과 정확히 일치(보컬 분리 후에도 잔향/누출이 있는 폴리포닉 오디오). 코드: https://github.com/emirdemirel/ALTA (GMM-HMM 파이프라인), 데이터: https://github.com/emirdemirel/DALI-TestSet4ALT. 구현 난이도: 중간 — `<star>` 토큰을 `<silence>`와 `<music>` 두 개로 분리하려면 MMS 어댑터의 출력 vocab을 확장하고 파인튜닝이 필요해 보임(순수 추론 단계 수정으로는 불가).
- 문제 ①의 "간주 구간에 가사가 몰려 들어가고 그 뒤가 비는" 현상 자체를 정면으로 다룬 논문(계층적 사전확률, VAD 이외의 명시적 구간 처리)은 위 두 편(Cheng 2025의 마스킹, Demirel 2021의 이중 토큰) 정도였고, 그 외에는 대개 "Demucs/Spleeter로 보컬 분리 → VAD로 트리밍"이라는 전처리 수준에 머물러 있었습니다(Vaglio 2020, MSTRE-Net 관련연구 절 다수 인용).

## 5) 일본어·한국어 가창 정렬

- **직접적인 일본어/한국어 전용 논문은 확인 못 함.** 가장 근접한 것은 Kang et al. HCLAS-X(위 3항, 한국어 6.7만곡 in-house 학습 실측, 코드/데이터 비공개)과, Cheng 2025 저자진(AIST, 일본 국립연구소)이 향후 일본어 확장을 시사할 법하나 논문 자체는 영어(DALI) 전용입니다.
- **Nightingale**(카라오케 오픈소스 프로젝트, 논문 아님): https://github.com/rzru/nightingale — WhisperX 기반으로 일본어/중국어/광둥어/한국어에 대해 문자 단위 강제정렬 + 로마자 표기(헵번식/병음/월병/한국어 로마자)를 얹는 실무 도구. 학술 논문은 아니지만 프로덕션에서 다국어(J/K 포함) 강제정렬을 다루는 드문 공개 사례라 참고할 가치 있음.
- Vaglio 2020의 "universal phoneme set" 접근이 이론적으로는 일본어/한국어에도 적용 가능하나 원 논문의 9개 언어 실험에는 포함 안 됨(로마자 계열 언어 위주). MMS가 이미 1000+ 언어를 다루므로 이 갭은 MMS 채택으로 실질적으로 메워진 상태로 판단됩니다.

## 요약 — 문제별 가장 실전적인 다음 스텝
- **①(간주 오배치)**: Cheng 2025의 시작/끝 무음 트리밍(구현 난이도 최저, 즉시 적용 가능) + Demirel 2021의 이중 토큰(무음 vs 반주) 아이디어는 재학습 필요.
- **②(비결정성)**: 기존 문헌에 직접 해법 없음. peakiness 이론(2406.02560)과 결합해 "posterior entropy가 임계값 이상이면 재정렬 또는 앙상블" 같은 자체 휴리스틱을 고려할 근거는 있으나, 이건 저희가 직접 설계해야 하는 영역.
- **③(CTC 구조적 한계)**: 단기적으로는 label-priors 보정(2406.02560, 재학습 불필요 가능성)이 가장 낮은 비용. 장기적으로는 Durand 2023 / Kick 2025의 contrastive/box-loss 방식이 근본적이지만 재설계 수준의 작업.

원문을 직접 못 읽어 "확인 못 함"으로 남긴 항목: Kick 2025 논문 본문(OpenReview 인증장벽), 2406.02560의 저자명, Durand 2023/Cheng 2025의 공개 코드 저장소 유무.
