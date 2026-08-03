# separation — 보컬 분리·전처리 영향 조사 (SendMessage 전문 1건)

조사 결과입니다. WebSearch 세션 예산이 조사 도중 소진되어 후반부는 WebFetch(원문 직접 열람)로만 보강했습니다. 수치는 가능한 한 원문에서 뽑았고, 2차 소스나 확인 못 한 것은 명시했습니다.

## 1. 분리가 정렬을 돕는가 — 실측 (핵심)

**a) Gao/Sharma/Gupta/Li, "Automatic Lyrics Alignment and Transcription in Polyphonic Music: Does Background Music Help?", arXiv:1909.10200 (ICASSP 2020)**
분리(Wave-U-Net M4) 후 정렬 vs 원곡(폴리포닉) 직접 정렬을 정면 비교:
- 정렬오차(초): Mauch 3.62(분리) → 0.21(폴리포닉+장르정보) / Hansen 0.67→0.18 / Jamendo 0.39→0.22
- WER(%): Mauch 76.31→54.08 / Hansen 78.85→60.77 / Jamendo 71.83→66.58
- 결론: **원곡 직접 학습이 전 지표에서 압도적으로 우수**. 저자들은 "분리 아티팩트, 특히 간주·장르전환 구간에서 실패"를 원인으로 지목.
- **중요 caveat**: "폴리포닉" 쪽은 범용 CTC가 아니라 폴리포닉 전용으로 특별 학습된 음향모델이고, 분리 쪽은 2018년식 Wave-U-Net(현대 Demucs보다 SDR 훨씬 낮음). 이 결과를 "최신 고품질 분리기+귀하의 MMS 어댑터" 조합에 그대로 외삽하기는 위험합니다.

**b) Stoller/Durand/Ewert, "End-to-end Lyrics Alignment...", arXiv:1902.06797 (ICASSP 2019)**
Mauch에서 0.35s 평균 정렬오차(기존 대비 "an order of magnitude" 개선). 별도 분리 서브모듈 없이 폴리포닉 오디오에서 직접 문자 확률 예측. 분리 vs 비분리 ablation 수치는 본문 텍스트 추출 실패로 확인 못함 — 다만 방법론 자체가 "분리 생략"을 전제로 설계됨.

**c) "Exploiting Music Source Separation for ALT with Whisper", arXiv:2506.15514 (2025)** — 과제는 정렬이 아니라 전사(transcription)지만 인접:
- 단문형 WER(%): Jam-ALT 믹스 20.99 / mdx 21.17 / mdx_extra 21.08(**개선 미미, 역전도 있음**) — MUSDB-ALT 믹스 23.59 / mdx 23.98(**악화**) / mdx_extra 20.00(개선) / 정답 vocal stem 14.19
- 장문형: Jam-ALT 믹스 20.35 / 분리 20.72(**악화**) — MUSDB-ALT 믹스 22.72 / 분리 20.07 / vocal stem 14.98
- 데이터셋마다 부호가 뒤집힘. 정답 stem과의 격차(14~15% vs 20~24%)는 분리 품질을 더 올릴 여지가 있다는 신호지만, 상용 분리기끼리(mdx vs mdx_extra) 차이만으로는 효과가 작고 불안정.

**d) "Enhancing Lyrics Transcription on Music Mixtures with Consistency Loss", arXiv:2506.02339 (2025)**
Whisper large-v2 WER(%): 믹스 36.80 / **HTDemucs 분리 37.50(무분리보다 악화)** / Music.AI 분리 32.36(뚜렷한 개선). 저자 스스로 "이전 연구와 달리 우리 예비실험은 분리 후 개선 가능함을 보였다"고 명시 — 즉 **분리 효과의 방향성 자체가 논문마다 갈린다는 걸 저자들도 인지**. 성능저하 원인 분석: "인코더 표현이 배경음악 정보까지 함께 인코딩해 ALT와 무관한 정보가 섞인다" — 팀이 보는 CTC posterior 붕괴와 기전적으로 유사(도메인 밖 신호가 표현을 오염).

**종합**: "분리를 더 좋은 모델로 바꾸면 정렬이 실제로 나아지는가"에 직접 답하는 강제정렬(forced alignment) 논문은 못 찾았습니다. 인접한 전사(transcription) 문헌에서는 (i) 분리 효과 방향이 데이터셋·분리기 조합마다 뒤집히고, (ii) 품질이 낮은 분리기(HTDemucs vanilla 일부 세팅)는 무분리보다 못한 사례가 실측됐고, (iii) 정답 vocal stem까지 품질을 올리면 확실한 상한 개선은 있습니다. **"htdemucs→BS-RoFormer로 바꾸면 정렬이 좋아진다"를 직접 뒷받침하는 수치는 확인 못함.**

## 2. Demucs 이후 분리 모델

| 모델 | SDR (MUSDB18HQ) | 속도 | 코드/가중치 | 라이선스 | RTX 3090 단일 추론 |
|---|---|---|---|---|---|
| htdemucs (v4, base) | 공식 README에 개별 수치 없음(확인 못함) | 기준 | github.com/facebookresearch/demucs (2025-01 archived) | 코드 MIT. **가중치 라이선스는 불명확**(이슈#327 미해결, 상업 재배포 시 주의 — 로컬 추론만 하는 팀 상황에서는 실무 리스크 낮음) | 이미 사용 중 |
| **htdemucs_ft** | 종합 9.0dB(공식 README) / vocals median 9.19dB(2차 소스 aistemsplitter.org·dev.to, 1차 대조 못함) | **htdemucs 대비 4배 느림(공식 README 명시)** | 동일 저장소 | 동일 | 품질↑ 속도↓ 트레이드오프, 42초 병목과 상충 |
| mdx_extra | 구체 수치 확인 못함(Whisper 논문에서 mdx보다 낫다는 것만 간접 확인) | 확인 못함 | github.com/kuielab | - | - |
| BSRNN (Luo&Yu, Tencent, arXiv:2209.15174, TASLP'23) | 평균 SDR 7.94dB (TFC-TDF-UNet v3 논문의 비교표 인용) | **0.7배 (실시간보다 느림!)** | 비공식: github.com/amanteur/BandSplitRNN-Pytorch, 공식 가중치 없음 | - | **속도상 불리, 팀 목적에 부적합** |
| **BS-RoFormer** (Lu et al., ByteDance, arXiv:2309.02612, ICASSP'24) | 9.80dB(추가데이터 없음) / **11.99dB(추가 500곡 학습, SDX23 MSS 1위)** | 어텐션 기반, 긴 오디오에서 무거움(283k 토큰@44.1kHz 언급, 1차 검증 못함). Demucs 대비 절대 속도 비교 불확실 | github.com/lucidrains/BS-RoFormer(비공식, MIT) + viperx 재현 가중치(MVSep/HuggingFace) | **공식 가중치 비공개, 커뮤니티 가중치 라이선스 확인 못함** | VRAM/속도 실측 필요, 확정 못함 |
| **Mel-RoFormer for Vocal Separation** (Wang et al., ByteDance, arXiv:2409.04702, ISMIR'24) | MUSDB18만 12.08dB / 전체(내부)데이터 13.29dB (44.1kHz, 105M params) — 같은 조건 BS-RoFormer 11.49~12.82dB, HDemucs 8.04dB. **24kHz mono 경량판**: 9.1M params 11.01dB / 50.7M params 12.69dB | 확인 못함 | "오픈소스 구현 참고"라고만 언급, 정확한 저장소 URL·공식 가중치 공개 여부 확인 못함 | 확인 못함 | 24kHz 경량판이 속도 후보로 흥미롭지만 **가중치 확보 가능성 자체 불확실** |
| **TFC-TDF-UNet v3** (MDX23 우승, Kim/Lee/Jung, arXiv:2306.09382) | 평균 SDR 7.90dB, vocals 9.22~9.38dB | **15배 실시간(overlap-add 미적용) / 3.9배(적용)** — 조사한 모델 중 가장 빠름 | github.com/kuielab/sdx23 | 확인 못함 | **속도 최우선이면 가장 유력한 후보**, 단 SDR은 htdemucs_ft(9.0)와 비슷한 수준 |

## 3. 분리 품질 vs 정렬 정확도 상관관계
1번 항목에 통합 서술. 요약: 방향성(더 좋은 분리=더 좋은 결과)은 대체로 존재하지만 **비선형적이고 임계치 의존적**입니다 — "적당히 좋은" 분리기(HTDemucs vanilla)는 무분리보다 못할 수 있고, 정답 수준 분리와의 격차는 여전히 큽니다. 강제정렬 자체에서 SDR-정렬정확도 상관을 직접 측정한 논문은 확인 못함.

## 4. 합성 보컬(보카로 등)의 특수성
**직접 이 현상(CTC posterior가 바닥으로 평평해지는 것)을 보고한 논문은 확인 못했습니다 — 중요한 문헌 공백입니다.** 인접 근거만 있습니다:
- SVS(노래합성) 명료도 연구(검색상 arXiv:2511.13910 계열 문헌 추정, 정확한 서지 100% 확정은 못함): "합성 음색이 악기음(안정된 배음구조·정적 공명)에 가까워질수록 자음의 과도적 조음 단서가 약해져 명료도가 떨어진다"고 보고. 음성 인식 음향모델이 의존하는 바로 그 단서(파열음/마찰음의 순간적 스펙트럼 변화)가 구조적으로 약하다는 뜻이라, CTC posterior 붕괴와 **기전적으로는 들어맞지만 직접 검증은 아닙니다.**
- MMS/wav2vec2는 인간 발화(주로 낭독체) 1100+ 언어로 학습되었고 노래·합성음성은 학습 분포에 없음 — "노래"와 "합성음색"이라는 **이중의 도메인 밖 이동**이라는 것은 구조적으로 타당하나, 이를 정량화한 논문은 확인 못함.
- **대응 방법**: 문헌에서 어댑터 미세조정 없이 시도할 만한 것으로 제시된 것은 없었습니다. "확인 못함"이 정직한 답입니다.

## 5. 전처리 요소
- **샘플레이트/리샘플링**: 정량 연구 확인 못함. MMS_FA(torchaudio)는 특정 샘플레이트(통상 16kHz)로 학습된 모델이므로 리샘플러 구현·목표 샘플레이트가 사전학습 조건과 정확히 일치하는지는 **공짜로 점검 가능한 항목**이지만 수치 근거는 확보 못함.
- **라우드니스 정규화**: RNNoise 등 공격적 디노이징이 WER를 오히려 1.27%p 악화시킨 사례 확인(특정 방언 ASR 논문 문맥). 라우드니스 정규화 자체의 A/B 정량 효과는 확인 못함. wav2vec2 계열은 [-1,1] 진폭 정규화가 일반적 관행이라는 것만 확인.
- **대역 제한**: 관련 문헌 확인 못함.
- **잔향 제거(de-reverb)**: 원거리마이크 ASR(REVERB Challenge)에서는 WPE로 8채널 기준 실조건 WER 20% 상대감소, 다른 세팅 41.5%/13.8% 상대감소 확인 — 단 이건 "방 잔향" 상황이지 음악 믹스 안의 **프로덕션 리버브 이펙트**가 아닙니다. 보컬 잔향제거용 RIR 데이터셋(ReverbFX, arXiv:2505.20533)은 존재하나, 이를 ALT/정렬 downstream에서 검증한 수치는 확인 못함.

## 특히 답해주신 질문에 대해

**분리를 더 좋은 모델로 바꾸면 정렬이 실제로 나아집니까?** — **근거 약함/불명확**. 방향성은 문헌마다 갈리고(위 c, d 참조), 강제정렬 자체에서 이를 측정한 연구는 없습니다.

**posterior 평평해짐을 보고한 자료?** — **확인 못함**. 인접 근거(자음 조음 단서 약화)만 있고 직접 검증 문헌은 공백입니다.

**분리를 건너뛰는 게 나은 경우?** — 있습니다(1909.10200). 단, 그 결과는 "폴리포닉 전용 음향모델"과 짝지어진 결과라 귀하의 오프더셀프 MMS 어댑터에 그대로 적용될지는 불확실합니다.

**42초 줄일 여지?** — 조사한 것 중 TFC-TDF-UNet v3가 가장 빠름(15배 실시간, vocals SDR 9.2~9.4dB, htdemucs_ft와 비슷한 품질). Mel-RoFormer 24kHz mono 경량판(9.1M params)도 후보지만 가중치 확보 가능성이 불확실.

## 바꿔 볼 가치가 있는 것 하나

**분리 모델 교체 자체는 바꿀 근거가 약합니다.** 문헌들이 서로 부호가 다르고, 팀이 실측한 실패 양상(보카로 곡 quality_score≈0.001)을 분리 품질 향상이 고친다는 직접 증거가 없습니다. 오히려 이 붕괴는 "노래+합성음색"이라는 이중의 도메인 불일치에서 올 가능성이 있고, 그렇다면 분리기를 아무리 좋게 바꿔도 근본 원인(자음 단서 자체가 원신호에 약함)은 안 건드립니다.

대신 제안: **원인부터 가르는 값싼 A/B 진단** — 지금 무너지는 보카로 곡 소수(quality_score≈0.001인 곡들)만 골라 "분리 있음 vs 분리 없음(원곡 그대로 CTC 통과)"을 재실행해 비교하세요. GPU 재학습·새 모델 다운로드 불필요, 기존 파이프라인에서 분리 단계만 스킵한 소수 곡 재실행이면 됩니다.
- 분리 생략 쪽이 낫다면 → 1909.10200이 시사하듯 분리 아티팩트가 범인 → 분리기 품질보다 "그 곡들만 분리 스킵" 또는 약한 wet/dry 블렌딩을 검토
- 분리 생략도 여전히 붕괴한다면 → 문제는 분리가 아니라 MMS의 합성보컬 도메인 불일치 → 분리기 교체에 시간/GPU 쓰는 건 헛수고이고, 저신뢰 구간 폴백 처리나 소량 어댑터 미세조정 쪽이 근본적

Sources: [1909.10200](https://arxiv.org/pdf/1909.10200) · [1902.06797](https://arxiv.org/abs/1902.06797) · [2506.15514](https://arxiv.org/html/2506.15514v1) · [2506.02339](https://arxiv.org/html/2506.02339) · [Demucs README](https://github.com/facebookresearch/demucs) · [Demucs #327 라이선스 이슈](https://github.com/facebookresearch/demucs/issues/327) · [BSRNN 2209.15174](https://arxiv.org/pdf/2209.15174) · [BS-RoFormer 2309.02612](https://arxiv.org/pdf/2309.02612) · [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) · [Mel-RoFormer 2409.04702](https://arxiv.org/pdf/2409.04702) · [TFC-TDF-UNet v3 2306.09382](https://arxiv.org/pdf/2306.09382) · [github.com/kuielab/sdx23](https://github.com/kuielab/sdx23) · [ReverbFX 2505.20533](https://arxiv.org/html/2505.20533)
