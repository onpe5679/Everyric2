# 최종 승격 후보 — 가중치 라이선스 확정 조사 (2026-07-31)

> 목적: 코드 라이선스가 아니라 **가중치(체크포인트) 자체의 라이선스와 출처 체인**을 확정한다.
> 판정은 두 시나리오로 분리한다 — **(서버) 상업 서비스**: 가중치를 서버에서 추론에만 쓰고 배포하지 않음.
> **(클라) 클라이언트 배포**: 가중치 자체를 브라우저/앱 패키지에 포함해 배포함.
> 방법: 리포 내 기존 기록 확인 → HF 모델 카드/GitHub README·LICENSE·이슈를 웹에서 원문 재확인. 추정 배제 — 명시된 문구만 인용하고, 없으면 "미명시"로 남긴다.
> 조사 주체: codex 서브에이전트 3기(병렬) + 핵심 주장 2건(NVIDIA Riva 라이선스 원문, demucs #327 실제 댓글)은 1차 출처로 직접 재검증.

관련 문서: [2026-07-31 아카펠라 품질 후보 조사](../2026-07-31-acapella-quality-candidates.md) · [2026-07-30 모델 교체 종합](README.md) · [Track A 분리](track-A-separation.md)

---

## 1. bs-polarformer (최우선 — 승격 유력 분리기)

### 출처 체인
- 아키텍처: BS-RoFormer + PoPE(극좌표 위치 임베딩). PoPE 원저자는 Anand Gopalakrishnan, Robert Csordás, Jürgen Schmidhuber, Michael C. Mozer(arXiv 2509.10534). 저자들의 공식 코드 저장소(`agopal42/pope`, MIT)는 언어/게놈/음악 시퀀스 모델링 실험용이며 **음원분리 가중치를 배포한 적이 없다** — 즉 논문 저자가 이 체크포인트를 공개한 것이 아니다.
- 배포 경로: ZFTurbo Music-Source-Separation-Training(MSST) v1.0.20 릴리스 자산(`model_bs_polarformer_float16.ckpt`). 릴리스 노트 원문(전문): *"Version of BS Roformer based on PoPE (Polar Coordinate Positional Embeddings). Arxiv paper: https://arxiv.org/pdf/2509.10534"* — 제3자 귀속 표기("...by 아무개") 없이 게시되어, **ZFTurbo 본인 또는 그의 팀이 PoPE 기법을 BS-RoFormer에 이식해 자체 학습한 것**으로 읽힌다(pretrained_models.md에서 viperx/aufr33처럼 저자를 명기하는 다른 항목들과 대비됨).
- MVSep 서비스 전용 "BS PolarFormer 124 bands"(vocals SDR 12.02)와는 **별개 모델**이며 조달 불가. 다운로드 가능한 이쪽은 vocals SDR 11.00(저장소 자체 보고).

### 명시된 라이선스
**가중치: 미명시.** 확인한 위치 전부:
- `docs/pretrained_models.md`(PolarFormer 행에 SDR/Config/Weights 링크만 있고 라이선스 컬럼 없음)
- 저장소 README(polarformer/PoPE/license 언급 전무)
- v1.0.20 릴리스 노트 본문(위 인용문이 전부)
- 저장소 `LICENSE`(MIT, 원문 발췌: *"...this software and associated documentation files (the 'Software')..."* — "Software"만 명시, checkpoint/weight/data 언급 전혀 없음)

**결정적 증거 — Issue #90** ("Redistribution of pre-trained models", ZFTurbo/Music-Source-Separation-Training). 관리자 ZFTurbo 본인 답변 원문:
> "not all models were posted by me. While I can add some open license on repo I think I can't really decide on each model."

즉 **저장소 관리자 본인이 "가중치별 라이선스는 내가 일괄 결정할 권한이 없다"고 명시적으로 선을 그었다.** 코드 라이선스(MIT)가 가중치까지 커버한다고 주장할 근거가 없다.

참고로 제3자 HF 미러(`bgkb/bs_polarformer`)가 자체적으로 `license: mit` 태그를 달아둔 사례가 있으나, 이는 재업로더의 임의 표기일 뿐 ZFTurbo나 PoPE 저자의 공식 확인이 아니므로 근거로 채택하지 않는다.

### 서버 상업 사용 판정
**불확실 (리스크 낮은 회색지대).** LICENSE는 코드에만 적용되고 관리자 본인이 가중치 결정 권한을 부인했다. 다만 강한 이의제기·DMCA 사례는 확인되지 않았다. 실측 벤치에는 포함하되(리포 정책상 "명백한 NC만 제외"), **채택(상업 배포) 시점에는 저자 확인이 선행 조건**이다.

### 클라이언트 배포 판정
**불가 권장.** 명시적 허가 없음 + 관리자의 "일괄 결정 불가" 선언 + PoPE 저자와의 관계(파생 여부)도 불명확. 가중치를 앱/브라우저에 패키징하는 것은 지금 근거로는 정당화되지 않는다.

### 판정 개정 (2026-07-31 재검증) — **MIT 확정으로 상향**
사용자 측 추가 리서치와 교차 검증으로 위 "미확인" 판정을 개정한다. 근거 체인:
1. **자가 학습·자가 배포 자산** — 본 문서 출처 체인에서 이미 확인했듯 릴리스 노트에 제3자
   귀속이 없고, `pretrained_models.md`는 서드파티 가중치에 일관되게 "(by 아무개 edition)"을
   달아 **외부 리포**로 링크하는 반면(viperx/KimberleyJensen/aufr33/JusperLee 등 실측 확인),
   PolarFormer는 무귀속 + ZFTurbo **본인 릴리스 자산**(v1.0.20)이다 — MDX23C·HTDemucs4 등
   자가 학습 계열과 같은 패턴.
2. **Issue #90 재해석** — "not all models were posted by me"는 서드파티 게시물에 대한
   유보이고, "While I can add some open license on repo"는 자가 게시 자산에 라이선스를 부여할
   의사·권한을 전제한다. 즉 이 발언은 자가 배포 자산의 저장소 라이선스 상속을 부정하지 않는다.
3. **시점** — 저장소 LICENSE(MIT)는 2024-11-04 커밋으로 추가됐고 v1.0.20은 그 이후 릴리스라
   "라이선스 존재 상태에서 배포된 자산"이다(소급 해석 불요).
4. **선례** — mlx-community가 같은 저장소의 자가 릴리스 자산을 "MIT 저장소의 릴리스 자산"
   해석으로 MIT 재배포 중.

**개정 판정: 서버 상업 사용 가능, 클라이언트 배포 가능 (MIT).** 준수 사항: 배포물에
"Copyright (c) Roman Solovyev (ZFTurbo)" 고지 + MIT 사본 포함, **릴리스 v1.0.20 자산 pin**
(HF 미러 `bgkb/*` 대신 원본 ckpt 직접 사용 — 체인 청결). 잔존 유의: 훈련 데이터 층위의
저작권은 별개(코퍼스 미명시 — 모든 분리 후보 공통의 통상 회색지대).

### 리스크 / 확인 채널
- 기존 이슈: `https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/90` (가중치 라이선스 결정 권한 부인), `.../issues/31` (코드 파생 여부 문의, 별개 사안)
- 신규 문의: `https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/new` (미개설 — 채택 전 열어야 함)

---

## 2. kimft-melband (현행 채택)

### 출처 체인
개인 제작자 Kimberley Jensen(MVSep/UVR 커뮤니티) → 2024-08-06 HF `KimberleyJSN/melbandroformer`에 `MelBandRoformer.ckpt` 최초 업로드(당시 라이선스 태그 없음). ZFTurbo MSST 학습 코드 사용, 데이터 기여자로 aufr33/Anjok/bascurtiz(모두 MVSep/UVR 커뮤니티 인물) 표기. MVSep 리더보드에 이 체크포인트가 직접 참조됨(vocals SDR ~10.98~12.85, 조사 시점별 차이).

### 명시된 라이선스
HF 모델 카드 YAML frontmatter 현재 `license: mit`. **이력이 있다** (커밋 로그로 확인):
1. 최초 업로드: 라이선스 태그 없음
2. 2025-06-17, 커밋 `f45f9e3`: `license: gpl-3.0` 추가 — HF discussions #2에서 Audacity 개발자 RyanMetcalfeInt8의 요청에 대응한 조치
3. 2026-04-19, xocialize가 discussions에서 "GPL 3 can be a little restrictive"(MLX 이식 목적)라며 재검토 요청
4. **2026-04-22, 커밋 `ac9b061`: `-license: gpl-3.0` / `+license: mit`로 변경**

리포 내 로컬 기록("2026-04-22부로 MIT 전환")은 정확하다. 다만 **근거는 HF 태그(YAML)와 커밋 diff뿐 — 별도의 서면 라이선스 성명이나 LICENSE 파일은 존재하지 않는다.** 출처: `https://huggingface.co/KimberleyJSN/melbandroformer/discussions/2`, 커밋 `ac9b061`.

### 서버 상업 사용 판정
**가능.** 현재 태그 MIT, 상업 제한 문구 없음.

### 클라이언트 배포 판정
**가능.** 단, 제작자가 "태그 없음 → GPL-3.0 → MIT" 순으로 라이선스를 두 번 바꾼 이력이 있어 **재변경 리스크가 있다.** 채택 시점의 커밋 해시를 기록해 pin하는 것을 권장(현재 벤치 어댑터는 `resolve/main/`으로 최신을 그때그때 받는 구조라 재변경에 노출됨).

### 리스크 / 확인 채널
- 학습 데이터셋(aufr33/Anjok/bascurtiz 제공) 자체의 권리 관계는 미확인 — "체크포인트 MIT 표기 ≠ 데이터 권리 완결"(기존 리포 기록과 일치).
- 2025-06-17~2026-04-22 사이 배포된 사본(캐시·미러)은 GPL-3.0 조건이 적용될 수 있음.
- 확인 채널: `https://huggingface.co/KimberleyJSN/melbandroformer/discussions`, `/commits/main`

---

## 3. demucs-onnx-fp16

### 출처 체인
Meta(`facebookresearch/demucs`) `htdemucs_ft` PyTorch 4-bag 앙상블 체크포인트 → StemSplit이 ONNX로 변환(fp16 변형 포함) → HF `StemSplitio/htdemucs-ft-vocals-onnx` 배포. 관련 코드: `StemSplit/demucs-onnx`(PyPI 동일).

### 명시된 라이선스
StemSplit 측: HF/GitHub/PyPI 모두 `license: mit`, 기술 노트 원문: *"This repo is MIT-licensed, matching the original HT-Demucs."* / *"All artefacts in this release are MIT-licensed."* — **전부 StemSplit 자체 선언**이며 Meta에 문의하거나 인용한 근거는 확인되지 않는다.

**★결정적 반증 (직접 GitHub API로 재검증 완료).** Meta 원저자이자 저장소 관리자 Alexandre Défossez(`adefossez`, author_association: CONTRIBUTOR)가 이슈 #327("License of pre-trained models")에 2022-05-23 남긴 댓글 원문 그대로:
> "The model weights are not covered by the MIT license, and are provided only for scientific purposes."

이후 2024-06 사용자가 "Intel의 OpenVINO 변환판(`Intel/demucs-openvino`)은 MIT로 태깅했는데 어느 라이선스가 '과학적 목적'에 맞는 정답이냐"고 재질문했으나 **관리자의 추가 답변은 없다** (2026-07-31 현재까지 open). 저장소는 2025-01-01 archived(read-only) 처리됨.

기존 리포 기록(`docs/research/2026-07-30-model-replacement/track-A-separation.md`)은 이 이슈를 "답변 없이 남아 있다"고 적었는데, **이는 부정확하다 — adefossez의 답변은 존재하며, 내용은 "MIT 아님, 과학적 목적 한정"이라는 부정적 답변이다.** 이번 조사로 정정한다.

코드 자체의 LICENSE(MIT, "Demucs is released under the MIT license")는 확인되나, 이는 **소프트웨어(코드)에 대한 것이고 가중치는 관리자 본인이 명시적으로 그 범위 밖이라고 못 박았다.** StemSplit의 ONNX 변환은 이 가중치를 형태만 바꾼 파생물이므로, 원 저작물의 라이선스 제한이 변환 과정에서 자동으로 해제된다고 볼 근거가 없다.

### 서버 상업 사용 판정
**불가에 가까움.** StemSplit의 MIT 재선언이 원저작자의 명시적 "과학적 목적 한정" 발언을 법적으로 무효화할 권한이 있는지 불명확하며, 오히려 정반대 방향(더 제한적)의 1차 증거가 존재한다.

### 클라이언트 배포 판정
**불가.** 가중치 자체를 포함해 배포하는 행위는 "scientific purposes only" 제한과 정면으로 충돌한다. 경량축(브라우저 배포) 후보로서의 전제 자체가 흔들린다 — `docs/research/2026-07-31-acapella-quality-candidates.md` Part 2의 R1 "1순위" 판정은 **라이선스 재검토가 필요**하다(SDR·RTF 수치는 유효하나 배포 가능성 전제가 무너짐).

### 리스크 / 확인 채널
- MUSDB18(원 학습 데이터) 자체도 일부 트랙 CC BY-NC-SA — 하류 제약 가능성 이중.
- 저장소가 archived되어 Meta 측 추가 해명을 받기 어려움. 문의 채널: `https://github.com/StemSplit/demucs-onnx/issues`(StemSplit에 직접 질의), 또는 archived된 `facebookresearch/demucs/issues/327`에 신규 댓글(코멘트는 가능할 수 있음, 재확인 필요).
- 참고: Intel의 `Intel/demucs-openvino`도 동일한 미해결 리스크를 안고 있음 — 업계 관행이 "MIT 재태깅"으로 흘렀으나 원저작자 발언과 배치됨.

---

## 4. nemo-nfa (한국어 음향모델)

### 출처 체인
개인/커뮤니티 개발자 SungBeom이 **NVIDIA RIVA Conformer ASR Korean**(NGC 카탈로그 사전학습 체크포인트, `speechtotext_ko_kr_conformer`)을 한국어 AI Hub 데이터셋(약 1,390만 샘플, ~14,000시간)으로 파인튜닝해 HF `SungBeom/stt_kr_conformer_ctc_medium`에 배포.

### 명시된 라이선스
HF 모델 카드 YAML frontmatter: `license: apache-2.0`. 본문 원문: *"해당 모델은 [RIVA Conformer ASR Korean]을 AI hub dataset에 대해 파인튜닝을 진행했습니다."*

**★결정적 충돌 (NVIDIA Riva License Agreement 원문을 PDF로 직접 확보해 재검증 완료, v. January 27, 2023, `developer.download.nvidia.com/assets/riva/NVIDIA-Riva-License-Agreement(27Jan2023).pdf`).** 기반 체크포인트가 속한 NGC 페이지는 *"By downloading and using the models and resources packaged with Riva Conversational AI, you would be accepting the terms of the Riva license"*라고 명시하며, 해당 라이선스 원문에 다음 조항들이 있다:

- **§1.5**: "'Riva Products' means NVIDIA Riva software and materials, which may include software, **models**..." — 모델 체크포인트가 Riva Products 정의에 명시적으로 포함됨.
- **§5.2**: "Any **model checkpoints** available under this license are licensed **only for deployment and use with the Riva Product**."
- **§5.6**: "...you may not copy, sell, rent, sublicense, transfer, distribute, modify or create derivative works of any portion of Riva Products, including (without limitation) in **any publicly accessible software repositories**."
- **§5.10**: "You may not use Riva Products in any manner that would cause them to become subject to an open source software or shareware license... (iii) redistributable at no charge." — **Apache-2.0으로 재라이선싱하는 행위 자체가 이 조항이 금지하는 상황과 정확히 일치한다.**
- **§9**: "NVIDIA reserves all rights, title and interest in and to Riva Products not expressly granted to you under this license."

즉 SungBeom이 HF에 붙인 "apache-2.0" 태그는 **NVIDIA가 소유권을 유보한 가중치에 대한 무단 재라이선싱으로 보이며, HF 공개 업로드 자체가 §5.6·§5.10을 위반하는 정황이 강하다.** 기존 리포 코드 주석(`scripts/bench_adapters/nemo_nfa.py:12` 등)의 "Apache-2.0" 확인은 **HF 표면 태그를 그대로 신뢰한 것으로, 상류 체크포인트 계보를 추적하면 근거가 무너진다.**

### 서버 상업 사용 판정
**불확실 → 실질적으로 불가에 가까움.** HF 태그와 무관하게 실제 지배 라이선스는 NVIDIA Riva License(§5.2: Riva Product와 함께 배포·사용하는 용도로만 한정)일 가능성이 높다. 이 프로젝트가 NVIDIA Riva 제품군 내에서 이 체크포인트를 쓰는 것이 아니라 독립 추론 서버에 붙이는 구조이므로, §5.2의 "only for deployment and use with the Riva Product" 요건과 충돌한다.

### 클라이언트 배포 판정
**불가.** §5.6이 공개 저장소 배포·서브라이선스·파생물 생성을 명시적으로 금지한다.

### 리스크 / 확인 채널
- AI Hub 원천 데이터 자체의 라이선스는 이번 조사에서 별도 확인하지 못함(추가 확인 필요).
- 이 후보는 **로컬 조사 문서의 라이선스 판정이 명백히 틀린 사례**이므로, 벤치마크 성능과 무관하게 승격 후보에서 제외하거나 NVIDIA에 직접 문의해 재확인해야 한다.
- 확인 채널: HF discussions(`https://huggingface.co/SungBeom/stt_kr_conformer_ctc_medium/discussions`), NVIDIA 라이선스 문의 창구(`Riva-license-questions@nvidia.com`, 라이선스 §15.9에 명시).

---

## 5. omniasr-ctc (다국어 음향모델)

### 출처 체인
Meta/FAIR, Omnilingual ASR 프로젝트(`facebook/omniASR-CTC-300M`).

### 명시된 라이선스
HF 모델 카드: `license: apache-2.0` (자매 저장소 `facebook/omniASR-LLM-300M`도 동일). GitHub LICENSE 원문(`facebookresearch/omnilingual-asr/blob/main/LICENSE`):
> "Copyright 2025 (c) Meta Platforms, Inc. and affiliates. Licensed under the Apache License, Version 2.0..."

순정 Apache-2.0이며, Llama류처럼 별도 Acceptable Use Policy 파일은 저장소에서 발견되지 않았다(LICENSE·CODE_OF_CONDUCT.md·CONTRIBUTING.md만 존재). 훈련 코퍼스 `facebook/omnilingual-asr-corpus`는 `cc-by-4.0` 태그.

### 서버 상업 사용 판정
**가능.** 모델·코드 모두 Apache-2.0이 1차 출처(HF 태그 + GitHub LICENSE 파일)로 명시되며, 다른 4개 후보와 달리 상류 소유권 충돌 정황이 발견되지 않았다.

### 클라이언트 배포 판정
**가능.** Apache-2.0은 재배포를 허용한다(저작권 고지·라이선스 사본 유지 조건).

### 리스크 / 확인 채널
훈련에 Common Voice·FLEURS·VoxPopuli·MLS·MMS·Babel 등 다수 공개 코퍼스가 섞였다는 것이 일반적으로 알려져 있으나, HF/GitHub 어디에도 OWSM처럼 데이터셋별 라이선스 전체 목록이 공개돼 있지 않다. 논문(arXiv:2511.09690) 본문은 이번 조사에서 직접 열람하지 못했다 — **"미확인"으로 남긴다.** NC 라이선스 데이터가 섞였다는 증거는 발견되지 않았으나, 완전히 배제할 근거도 없다. 6개 후보 중 **가장 깨끗한 판정**이지만 데이터 provenance 재확인이 남는다.

---

## 6. owsm-ctc-v4-1b (다국어 음향모델)

### 출처 체인
CMU WAVLab + Honda Research Institute Japan(ESPnet/OWSM 프로젝트). `espnet/owsm_ctc_v4_1B`는 기존 OWSM 데이터 믹스처(~180k시간) + 신규 정제 YODAS(166k시간, 총 ~320k시간)로 학습.

### 명시된 라이선스
HF 모델 카드: `license: cc-by-4.0`. 신규 추가분인 YODAS 데이터셋(`espnet/yodas_owsmv4`)은 `cc-by-3.0` 태그이며, 원 YODAS2 데이터셋 카드는 *"We made sure that our dataset only consisted of videos with CC licenses during our downloading"*라고 명시한다 — YouTube 크롤링이지만 CC 라이선스 영상만 선별했다는 주장.

**★핵심 긴장관계.** v4-1B가 그대로 물려받는 **"기존 OWSM 데이터"(~180k시간)** 는 OWSM-CTC 논문(arXiv:2402.12654) 부록 A.1에 코퍼스별 라이선스가 원문으로 명시되어 있다:
> "AIDATATANG (CC BY-NC-ND 4.0), AISHELL-1 (Apache 2.0)... CoVoST2 (CC BY-NC 4.0), Fisher Switchboard (LDC), Fisher Callhome Spanish (LDC)... GigaST (CC BY-NC 4.0)... MagicData (CC BY-NC-ND 4.0), MuST-C (CC BY NC ND 4.0)... TEDLIUM3 (CC BY-NC-ND 3.0)... Russian OpenSTT (CC-BY-NC)... VoxPopuli (Attribution-NonCommercial 4.0 International)..."

즉 **NC(비상업)·ND(변경금지)·LDC(유료회원제, 재배포 제한)** 코퍼스가 훈련 데이터에 다수 포함되어 있다. 전작 OWSM v3.1이 이를 의식해 "낮은 라이선스 제약" 데이터만으로 학습한 별도의 소형 LR(low-restriction) 모델을 따로 공개한 전례(arXiv:2401.16658 §2.2)가 있는데, **v4-1B는 그 LR 계열이 아니라 전체(제약 포함) 데이터 계열이다.**

ESPnet 측이 최종 가중치를 CC-BY-4.0으로 재배포하는 것은 "학습된 가중치는 훈련 데이터의 저작권 파생물이 아니다"라는 (모델 훈련 커뮤니티에서 흔하지만 법적으로 확정 판례가 없는) 입장 위에 서 있다.

### 서버 상업 사용 판정
**불확실.** HF 태그(CC-BY-4.0)만 보면 가능해 보이지만, 훈련 데이터 다수가 NC/ND/LDC 조건이라는 사실이 명시적으로 공개돼 있어 완전히 깨끗하다고 보기 어렵다. 6개 후보 중 nemo-nfa 다음으로 리스크가 크다.

### 클라이언트 배포 판정
**불확실 → 서버보다 더 위험.** "배포(distribution)"는 저작권법상 더 직접적인 행위이므로, 위 긴장관계가 더 첨예하게 적용된다.

### 리스크 / 확인 채널
- 확인 위치: HF 모델카드, `https://huggingface.co/datasets/espnet/yodas_owsmv4`, arXiv:2402.12654 부록 A.1, arXiv:2401.16658 §2.2.
- ESPnet/OWSM 팀에 "v4-1B가 NC/ND/LDC 데이터를 포함한 상태에서 CC-BY-4.0으로 가중치를 재배포하는 법적 근거"를 문의할 채널: `https://github.com/espnet/espnet/issues`, 또는 HF 리포 discussions.
- 기존 리포 코드 주석(`scripts/bench_adapters/owsm_ctc.py:32-33`)의 "Commercial use is permitted with attribution"은 **HF 표면 태그만 반영한 것으로, 훈련 데이터 제약까지 고려하면 과장된 판정**이다.

### 재검증 (2026-08-01, 외부 리서치 반입분 대조)

사용자가 "v4는 YODAS(CC-BY-3.0) 기반이라 NC 우려가 해소됐다"는 외부 조사를 반입해 1차 자료로 재확인했다.
**결론: 상단 판정 유지.** 반입 조사의 결정적 전제가 사실과 다르다.

| 반입 주장 | 1차 자료 확인 결과 |
|---|---|
| v4는 v3.1 데이터를 대체한 깨끗한 버전 | **오류 — 대체가 아니라 추가.** OWSM v4 논문(arXiv:2506.00338): *"trained on the cleaned YODAS dataset **in conjunction with previous OWSM data** (320k hours in total)"*. 데이터셋 카드도 *"combined with existing OWSM data"*. 구성은 YODAS 166k + 기존 OWSM 154k → NC/ND/LDC 코퍼스가 그대로 잔류 |
| 가중치 라이선스 태그 미기재 | **오류 — 기재돼 있다.** 모델 카드 `license: cc-by-4.0`(원래 감사 내용과 일치) |
| YODAS 자체가 CC-BY-3.0 | **사실.** `espnet/yodas_owsmv4` = cc-by-3.0 |
| v4가 v3.1보다 데이터 라이선스상 깨끗 | **부분 사실.** 절대량은 개선(166k 클린 추가)이나 제약 코퍼스가 제거된 것은 아님 |

추가 확인: v3.1에 있던 LR(low-restriction, 70k시간 — AMI·CommonVoice·FLEURS·KsponSpeech·LibriSpeech·MLS·VCTK만)
계열이 **v4에는 없다**(논문·HF 컬렉션에 해당 변형 부재). 즉 "제약 없는 OWSM v4"라는 선택지는 존재하지 않는다.

**리스크 등급 정정(완화 방향)**: 다만 nemo-ko와 같은 등급으로 묶는 것은 과했다. 둘의 리스크는 **종류가 다르다** —
- nemo ko ckpt: Riva EULA가 **가중치 자체의 파생·재배포를 계약으로 직접 금지**(§5.2/5.6/5.10). 해석 여지가 좁다.
- owsm v4: ESPnet이 가중치에 **CC-BY-4.0을 명시적으로 허여**했고, 쟁점은 "훈련 데이터 제약이 가중치까지
  미치는가"라는 **법적 미확정 영역**이다. 업계 관행은 미치지 않는다는 쪽이며, 실제로 다수 상용 제품이 같은
  전제로 출시된다.

따라서 owsm-ctc-v4는 **"클린 아님 / 그러나 nemo-ko보다 한 등급 낮은 리스크 — 경영 판단 영역"**으로 재분류한다.
채택 시 준수사항: CC-BY-4.0 저작자 표시(ESPnet/OWSM + YODAS 출처), 원 체크포인트 pin.

---

## 종합 요약표

| 모델 | 가중치 출처 | HF/저장소 표기 라이선스 | 서버 상업 사용 | 클라 배포 | 핵심 리스크 |
|---|---|---|---|---|---|
| **bs-polarformer** | ZFTurbo MSST v1.0.20 릴리스 자산(자가 학습 패턴 확인) | **MIT (저장소 상속 — 2026-07-31 개정, §1 개정 절 참조)** | **가능** (고지+사본 포함) | **가능** (v1.0.20 pin 권장) | 훈련 데이터 층위는 통상 회색 |
| **kimft-melband** | Kimberley Jensen(MVSep 커뮤니티), 2024-08 최초 배포 | **MIT** (2026-04-22 GPL-3.0→MIT 전환 커밋 확인) | 가능 | 가능(커밋 pin 권장) | 유일 근거가 HF 태그, 재변경 이력 있음(2회 변경) |
| **demucs-onnx-fp16** | Meta `htdemucs_ft` → StemSplit ONNX 변환 | StemSplit 주장 MIT, **원저작자는 명시적으로 부인**("not covered by MIT... scientific purposes only", 이슈 #327) | **불가에 가까움** | **불가** | 1차 증거가 StemSplit 주장과 정반대 |
| **nemo-nfa** | NVIDIA Riva Conformer ASR Korean → SungBeom 파인튜닝 | HF 태그 Apache-2.0, **기반 체크포인트는 NVIDIA Riva License(§5.2/5.6/5.10) 적용 대상** | **불가에 가까움** | **불가** | HF 태그가 상류 라이선스와 정면 충돌(무단 재라이선싱 정황) |
| **omniasr-ctc** | Meta/FAIR Omnilingual ASR | **Apache-2.0** (HF 태그 + GitHub LICENSE 파일 일치) | **가능** | **가능** | 훈련 코퍼스 전체 목록 미공개(데이터 provenance만 미확인) |
| **owsm-ctc-v4-1b** | CMU WAVLab/Honda RI Japan, YODAS+기존 OWSM 믹스처 | HF 태그 CC-BY-4.0, 단 훈련 데이터에 **NC/ND/LDC 코퍼스 다수 포함**(논문 부록 명시) | 불확실 | 불확실(서버보다 위험) | 가중치-훈련데이터 파생 관계의 법적 미확정 |

### 판정 요지
- **깨끗한 후보는 omniasr-ctc 하나뿐이다.** 서버·클라 모두 문제없이 진행 가능.
- **kimft-melband(현행)는 실무적으로 유지 가능**하나 라이선스 재변경 이력이 있어 커밋을 pin해야 한다.
- **bs-polarformer(최우선 승격 후보)는 MIT로 개정 확정**(2026-07-31 재검증, §1 개정 절) — 자가 학습·자가 배포 자산의 저장소 라이선스 상속. 라이선스 게이트 해소, 남은 승격 관문은 청취뿐.
- **demucs-onnx-fp16과 nemo-nfa 둘 다 기존 리포 기록의 판정("MIT", "Apache-2.0")이 상류 계보 추적 결과 뒤집혔다.** 특히 nemo-nfa는 NVIDIA Riva 라이선스 조항(모델 체크포인트를 Riva 제품과만 결합해야 하고, 공개 저장소 배포·재라이선싱을 명시적으로 금지)과 정면 충돌하는 정황이 강해 **승격 후보에서 제외하거나 즉시 NVIDIA에 문의**해야 한다.
- **owsm-ctc-v4-1b는 표기 라이선스(CC-BY-4.0)와 훈련 데이터 구성(NC/ND/LDC 다수 포함) 사이에 논문 저자들도 인지하고 있던 긴장관계**가 있다(전작이 별도 LR 모델을 낸 이유). 완전한 청신호로 볼 수 없다.
