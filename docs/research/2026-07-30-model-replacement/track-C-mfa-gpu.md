# 과제 C — MFA 한국어 정렬의 GPU 가속·개조 현실성 (codex gpt-5.6-luna, 2026-07-30)

> 3갈래 모델 교체 조사(A 분리 / B 정렬 음향모델 / C MFA GPU) 중 C 트랙 보고 원문.
> 선행 자료: [2026-07-26 논문·특허 통합 조사](../2026-07-26-alignment-papers-patents-survey.md)

## 1. 결론 요약

| 노선 | 판정 | 핵심 이유 |
|---|---|---|
| MFA GMM-HMM GPU 개조 | 비추천 | MFA의 핵심 정렬 경로는 Kaldi GMM-HMM/FST/Viterbi이며 공식 CUDA 경로가 없습니다. GPU로 옮겨도 정확도 향상은 보장되지 않습니다. |
| 가창 특화 정렬기 도입 | 조건부 추천 | 가창 음향모델은 장모음, 비브라토, 피치 변화에 맞춰야 합니다. 다만 한국어 공개 모델과 직접 비교 근거가 부족합니다. |
| 현 CTC 유지 + 디코딩 보정 | 1순위 추천 | 이미 155초 오디오를 약 5초에 처리하고 있으며, 성능 회귀 위험이 가장 낮습니다. CTC 정렬 전용 개선 연구도 존재합니다. |

> "MFA를 GPU로 개조하면 한국어 가창 정렬 SOTA가 된다"는 가설은 현재 근거가 부족합니다. MFA의 음성 정렬 정확도는 매우 강하지만, 한국어 노래에서 wav2vec2 CTC보다 우수하다는 직접적인 비교 결과는 확인하지 못했습니다.

## 2. MFA의 GPU 가속 가능 지점

MFA 2.x/3.x는 Kaldi GMM-HMM 기반이며 (MFA 3.x는 Kalpy Python 바인딩으로 중간 산출물을 줄였을 뿐 핵심 모델은 동일), MFCC/i-vector 계산은 GPU 가능하지만 HMM/FST Viterbi 디코딩은 시간축 의존성 때문에 GPU화 난도가 높습니다. MFA 공식 설치 문서는 Kaldi를 `kaldi=*=cpu*`로 안내합니다. 단순 개조가 아니라 사실상 Kaldi 정렬기 일부를 새로 구현하는 수준의 작업입니다. (확신도: 높음)

## 3. 멀티코어 병렬화

MFA 병렬화는 주로 다수 파일/화자 단위이며, 단일 155초 파일 하나에는 선형 가속이 적용되지 않습니다. 원 논문의 12코어로 1,000시간을 80시간에 처리한 수치는 corpus throughput이지 단일 파일 지연시간이 아닙니다. 실측이 필요합니다. (확신도: 중간)

## 4. 정확도 전제 검증

- 음성 벤치마크(TIMIT, Buckeye)에서는 MFA가 MMS/wav2vec2, WhisperX보다 우세 (Interspeech 2024 실측치 제시).
- 단, label-prior로 보정된 CTC 정렬 연구(2024)는 Buckeye에서 MFA와 비등, TIMIT에서는 MFA 열세.
- **한국어 가창 도메인에서 MFA가 wav2vec2 CTC보다 우수하다는 직접 근거는 확인하지 못함.** 가창 정렬은 보컬 분리 품질과 singing-adapted 음향모델이 정렬기 자체보다 더 크게 작용한다는 근거(ICASSP 2019: U-Net 분리 후 오차 33.81초→1.39초/6.34초)만 존재.

## 5. GPU 네이티브 대안

- **NeMo Forced Aligner**: CUDA 기반, 한국어 모델 선택 가능하나 speech 중심, 가창 실측 자료 없음.
- **k2/icefall**: CTC backend 교체 후보, 한국어 가창 음향모델은 별도 필요.
- **SOFA**(가창 특화): 공개 pretrained 모델은 중국어/광동어/일본어/영어/프랑스어뿐이며 **한국어 모델은 없음**. 리포지토리의 실제 SOFA 구현(`everyric2/alignment/sofa_engine.py`)도 모델 맵이 영어만 포함하고 `SUPPORTED_LANGUAGES = ["en"]`으로, factory.py의 "영어/일본어 지원" 표기와 불일치 — critical issue로 지적됨.
- **Qwen3-ForcedAligner**(2026 공개): 11개 언어 중 한국어 포함, 최대 5분 정렬. 다만 벤치마크는 MFA pseudo-label 기반 speech 평가이며 가창 직접 결과 아님. 가장 먼저 시험할 최신 후보로 제시.

## 6. 한국어 G2P 사전 한계 (가창 도메인)

MFA 한국어 사전(~54,074단어)은 speech alignment용. 가창에서는 종성 약화/탈락, 연음, 장모음(melisma), 비브라토로 인한 F0 변동이 formant 경계 관찰을 방해하며, 사전 오류가 정렬 오류로 확대(OOV/오탈자 → line drift). 무제한 variant 추가는 오히려 잘못된 발음 경로 선택 위험을 높임 — duration-aware decoder와 singing acoustic model이 필요하다는 결론.

## 7. 추천 실행 순서 (핵심)

1. 현 CTC를 기준선으로 고정하고 정량 지표(MAE, p90 drift, failure rate) 측정.
2. Qwen3-ForcedAligner 한국어, NeMo NFA 한국어를 후보로 비교 실험 (SOFA는 한국어 모델 없어 즉시 후보 제외, 별도 연구 트랙).
3. CTC blank/label prior, vocal onset anchor, 장모음 duration prior, chroma/melody feature 등 singing-aware 보정을 CTC에 추가하는 것이 MFA GPU 개조보다 투자 대비 효과가 높음.
4. MFA는 production 엔진이 아니라 offline teacher/disagreement 탐지용으로만 활용.

부수적으로 발견된 이슈: `torchaudio.functional.forced_align`이 deprecated 예정이며 리포지토리는 `torchaudio<2.9`로 고정 중(`pyproject.toml:36`) — 장기적으로 k2 기반 backend 등으로 교체 검토 필요.

## 8. 최종 판정

MFA GMM-HMM의 GPU 개조는 기술적으로 불가능하지 않으나 공식 경로가 없고 decoder까지 재구현해야 하므로 현실성이 낮음(비추천). 음성 도메인에서 MFA 우세 근거는 있으나, 한국어 가창에서 MFA가 현 CTC보다 우수하다는 직접 근거는 확인하지 못함. 가창 정렬 품질의 핵심은 GPU 여부가 아니라 singing acoustic model·source separation·duration/melody prior. 추천 노선: "현 CTC 유지 + k2 등 유지 가능한 backend + singing-aware 디코딩 보정", 신규 후보로 Qwen3/NeMo NFA를 벤치마크. SOFA는 한국어 모델 확보 시에만 별도 개발 검토.
