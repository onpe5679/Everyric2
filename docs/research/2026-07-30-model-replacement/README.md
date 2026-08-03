# 라이선스 프리 모델 교체 + 최적화 조사 종합 (2026-07-30)

> 이용자 증가·상업화 대비: 라이선스 제약 모델 제거와 파이프라인 최적화를 겸한 3갈래 교체 조사.
> 조사 주체: codex 서브에이전트 3기(gpt-5.6-luna) + [2026-07-26 논문·특허 통합 조사](../2026-07-26-alignment-papers-patents-survey.md) 교차 검증.
> 트랙 원문: [A 보컬 분리](track-A-separation.md) · [B 정렬 음향모델](track-B-ctc-model.md) · [C MFA GPU](track-C-mfa-gpu.md)

## 현행 스택의 라이선스 실태

| 구성요소 | 현행 | 라이선스 | 상업 판정 |
|---|---|---|---|
| 보컬 분리 | htdemucs (demucs CLI) | 코드 MIT / **가중치 불명확**(이슈#327 무응답, 저장소 2025-01 아카이브) | ⚠️ 리스크 — 승인받지 않은 상태로 봐야 |
| 정렬 음향모델 | facebook/mms-1b-all | **CC-BY-NC-4.0** (모델카드 재확인) | ❌ 상업 부적합 — 교체 필요의 진원 |
| 일본어 정렬 | jonatasgrosman xlsr-53-japanese | Apache-2.0 | ✅ |
| f0 | RMVPE(기본)/FCPE(폴백) | 추론 코드 MIT / RMVPE 가중치 출처 확인 필요 | △ 가중치 provenance 점검 |
| MFA (비사용) | 한국어·일본어 사전학습 모델 | **CC BY 4.0** | ✅ (참고: 오프라인 심판 용도로 가치) |

## 트랙별 결론

### A. 보컬 분리 — 교체 가치 있음 (라이선스 동기)
- **1순위: Kim FT MelBand RoFormer** — HuggingFace 체크포인트가 2026-04-22부로 **MIT 전환**. lucidrains(MIT)+ZFTurbo(MIT) 코드 조합으로 "코드+가중치 모두 상업 가능"이 확인되는 사실상 유일 조합. ZFTurbo 목록 vocals SDR 10.98(viperx보다 높음).
- viperx BS/Mel 계열·MVSep·SCNet·TFC-TDF·BSRNN: 코드는 명확하나 **가중치 라이선스 미명시 → 전부 보류**.
- **Mel-RoFormer의 "분리+멜로디 전사 동시"는 우리 f0 경로를 대체 못 함**: 전사 헤드는 노트 이벤트 출력(연속 f0 아님)이고 공개 체크포인트도 없음. RMVPE/FCPE 경로 유지, 분리 front-end만 교체.
- 통합 형태: demucs 서브프로세스는 폴백으로 유지 + RoFormer는 **장기 실행 워커 프로세스**(요청마다 재로드 금지, 동시성 1). VRAM 추정 5~9GB(미검증) — 9GB 예산에서 BS보다 MelBand부터 실측.
- 주의(7/26 조사): 분리 품질↑ = 정렬 정확도↑ 근거는 문헌상 없음 — **정렬 회귀는 A/B로만 판정**.

### B. 정렬 음향모델 — MMS 대체 후보 확정, 즉시 교체는 금지
- 한국어 1순위: **Kkonjeong/wav2vec2-base-korean** (Apache-2.0, 94M, **자모 CTC** 명시) — VRAM ~1/10.
  - 유의: 7/26 조사의 "자모 분해 반대"(ASR WER 실측)와 긴장 관계. 단 그 근거는 free decoding이고, 강제정렬은 자모 정렬 후 음절 스팬 재조립이 기계적이라 별개 — **실측으로 판정**.
  - 비교군: kresnik xlsr-korean (Apache-2.0, 300M, 음절 vocab ~1,203 — 현행 음절 경로 drop-in에 가장 가까움).
- 일본어: **reazon-research/japanese-hubert-base-k2-rs35kh** (Apache-2.0, 98M, 히라가나 정규화 경로) / 직접 한자·가나 정렬 유지 시 기존 jonatas vs **reazon wav2vec2-large-rs35kh** A/B.
- 다국어 실험군: **facebook/omniASR-CTC-300M** (Apache-2.0, 공식 추론 VRAM ~2GiB) — 단 SentencePiece vocab이라 문자 정렬 부적합 가능성, fairseq2 경로 필요.
- 제외 확정: MahmoudAshraf/mms-300m-forced-aligner(CC-BY-NC 직접 표기), OWSM-CTC(CC-BY-4.0이나 1B·80ms 프레임·ESPnet 경로), VoxPopuli(NC).
- 1B→95M 축소 시 타이밍 회귀 여부의 직접 문헌 없음 — **보카로 붕괴 곡 포함 A/B가 유일한 판정 수단**.

### C. MFA GPU 개조 — 비추천 (가설 기각)
- Kaldi GMM-HMM/FST Viterbi에 공식 CUDA 경로 없음 → 개조는 디코더 재구현 수준. 멀티코어 병렬화도 코퍼스 단위지 단일 파일 지연 개선 아님.
- 한국어 **가창**에서 MFA > CTC 직접 근거 없음(speech 벤치마크 우위만 존재). 가창 정렬 품질의 지배 요인은 GPU가 아니라 singing 음향모델·분리 품질·duration/melody prior.
- 추천 노선: **현 CTC 유지 + singing-aware 디코딩 보정**(7/26 조사의 star 성형·label prior·무음 트리밍과 합류). MFA는 CC BY 4.0이므로 **오프라인 심판(teacher)/불일치 탐지** 용도로 활용.
- 신규 후보: **Qwen3-ForcedAligner**(2026, 한국어 포함 11개 언어, ≤5분) — 라이선스·가창 성능 미확인, 벤치마크 후보.
- 부수 발견: `factory.py`가 SOFA를 "영어/일본어"로 광고하나 실제 `SUPPORTED_LANGUAGES=["en"]`(불일치) / `torchaudio<2.9` 고정 + `forced_align` deprecated 예정 → 장기 k2 backend 검토.

## 후보 선정 정책 (2026-07-30 사용자 확정)

- **명백한 비상업(NC) 라이선스만 후보에서 제외**한다 (MMS 계열, MahmoudAshraf aligner, VoxPopuli 등).
- **라이선스 "미확인"은 실측 대상에 포함**한다 — viperx BS/Mel, SCNet, TFC-TDF 등은 벤치마크에 넣고, 상업 채택 시점에 라이선스를 재확인·교섭한다.
- 실측 장비: 로컬 RTX 5090(개발·벤치 용도 사용 승인) + 서버 3090. 전 후보 다운로드 → 실측 → 회귀 테스트가 채택의 유일한 게이트.
- 반주(instrumental) 품질 트랙 별도 조사 진행(과제 D — 링크 반주 상관·MR 용도).
- **OWSM-CTC v4 1B(CC-BY-4.0)·Meta omniASR-CTC(Apache-2.0)도 실측 비교 대상에 포함**(사용자 지정 2026-07-30). 통합 마찰(ESPnet 추론 경로, OWSM 80ms 프레임, omniASR SentencePiece vocab·fairseq2)은 실측 하네스에서 어댑터로 흡수하고, 정렬 해상도·문자 타이밍 재구성 가능 여부를 실측으로 판정한다.
- **광역 스윕 원칙**(사용자 확정): 분리든 CTC든 다른 아키텍처든, 그럴듯한 후보는 **전부 다운로드해 같은 평가 세트로 비교**한다. 후보 목록은 닫지 않고 각 트랙 조사에서 나온 전 후보(분리: htdemucs/ft·Kim FT·viperx BS/Mel·SCNet·TFC-TDF·mdx 계열, 정렬 ko: Kkonjeong·kresnik·w11wo·thisisHJLee·Taeham·42MARU·SungBeom, ja: jonatas·ttop324·Reazon 3종·sakasegawa, 다국어·타 아키텍처: omniASR·OWSM·Qwen3-FA·NeMo NFA, 참조: MFA ko/ja CPU teacher)를 기본 포함한다. 명백한 NC만 제외.

## 권장 실행 순서 (전 트랙 공통: 회귀 없는 검증이 게이트)

1. **평가 기반 구축**: mir_eval + 실곡 층화 세트(일반 보컬 / 보카로·합성 / 붕괴 곡 / 랩·장음 / ko·ja·발음 변환 외국어) + 현행 스택 기준선 측정(MAE·P95·붕괴율·VRAM 피크). 곡 단위 paired 비교.
2. **B 트랙 A/B**: Kkonjeong(ko)·Reazon HuBERT(ja)를 shadow로 — 라이선스 제거 효과가 가장 크고(NC 소거) VRAM 절감도 최대(4GB→0.4GB급).
3. **A 트랙 A/B**: Kim FT MelBand 워커 도입, htdemucs와 shadow 비교(분리 SDR이 아니라 **하류 정렬·f0 지표**로 판정).
4. **C 후속**: Qwen3-ForcedAligner·NeMo NFA ko 벤치마크(저우선), MFA는 평가 세트 주석 보조(teacher)로.
5. 승격 조건(제안): MAE·P95 비악화, 합성보컬 붕괴율 비악화, 300ms 초과 라인 비율 비악화, VRAM 피크 9GB 이내.

## 미확정·후속 확인 필요

- Kim FT MelBand의 학습 데이터 provenance (체크포인트 MIT 표기 ≠ 데이터 권리 완결) — 배포 전 법무 검토 + hash 고정.
- Kkonjeong·Reazon의 노래·합성보컬 성능 (문헌 부재 — 실측만이 답).
- omniASR-CTC의 ko/ja 문자 단위 정렬 가능 여부.
- Qwen3-ForcedAligner 라이선스·가창 성능.
- RMVPE 가중치(rmvpe.pt)의 정확한 라이선스 계보.
- NeMo NFA 한국어 모델 존재 주장(C 보고)과 7/26 조사("CJK 없음")의 상충 — 재확인 필요.
