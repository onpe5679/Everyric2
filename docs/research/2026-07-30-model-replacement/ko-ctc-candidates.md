# 클린 라이선스 한국어 CTC 강제정렬 모델 후보 조사 (2026-08-01)

> 목적: NVIDIA Riva EULA에 막힌 `SungBeom/stt_kr_conformer_ctc_medium`(난곡 정밀도 82.4%)을
> **상업 사용이 확실히 가능한 한국어 CTC 음향모델**로 대체한다.
> 방법: 웹 조사만(다운로드·실행·GPU 사용 없음). HF 모델 카드 본문·`vocab.json` 원문·학습 데이터
> 출처·AI Hub 이용정책 원문까지 직접 추적. 추정과 확인을 문장 단위로 구분했다.
>
> 관련 문서: [Track B — CTC 음향모델 교체](track-B-ctc-model.md) · [최종 가중치 라이선스](final-weights-licenses.md) · [UST 정밀도 비교](ust-precision-comparison.md)

---

## 0. 요약

**상위 3 추천**

| 순위 | 모델 | 왜 nemo를 대체할 수 있는가 | 라이선스 판정 |
|---|---|---|---|
| 1 | `espnet/owsm_ctc_v4_1B` | **한국어 학습량이 10.9k시간**으로, nemo(SungBeom, 13,946시간)와 유일하게 같은 자릿수. 지금까지 후보가 nemo에 못 미친 근본 원인이 학습량(Zeroth 51.6시간)이었다면 이것만이 그 축을 메운다 | CC-BY-4.0 — **깨끗함(확정)** |
| 2 | `anantoj/wav2vec2-xls-r-1b-korean` | 현행 kkonjeong(94.4M)과 **데이터가 같고 용량만 10배**. 73.2% → 상승분이 순수 용량 효과인지 분리 측정 가능한 유일한 후보. Zeroth WER 4.49%로 kresnik(4.74%)보다 우수 | Apache-2.0 + CC-BY-4.0 데이터 — **깨끗함(확정)** |
| 3 | `slplab/wav2vec2-xls-r-300m_phone-mfa_korean` | **vocab이 43개 음소뿐**. CTC 클래스 수가 적을수록 blank 경쟁이 줄어 posterior가 집중된다 — 합성보컬 posterior 바닥 문제에 구조적으로 가장 유리. PER 3.88% | Apache-2.0, 서울대 자체 코퍼스 — **깨끗함(확정, 단 코퍼스 자체 라이선스 문서는 미확인)** |

**이번 조사가 뒤집은 기존 판정 2건** (§1에 근거)

1. **AI Hub / KsponSpeech는 "비상업 전용"이 아니다.** `한국어 음성`(KsponSpeech, dataSetSn=123)은 경진대회 데이터가 아니라 ETRI가 2018년 구축한 **일반 개방데이터**이고, AI Hub 이용정책 원문은 개방데이터에 대해 "영리적・비영리적 연구・개발 목적으로 활용할 수 있습니다"라고 명시한다. 비상업 제한은 경진대회·KETI·오디오북 등 **별도 카테고리**에만 걸린다.
2. **`kresnik`/`anantoj` 계열의 vocab은 음절 단위이며, 일본어 음차에 필요한 음절을 실제로 담고 있다.** `츠·쿄·큐·튜·퓨`가 모두 존재함을 `vocab.json` 원문에서 확인했다. 다만 `캬` 등 일부 요음은 없어 **완전 커버는 아니다**(§5).

**가장 중요한 미확인 사항**: OWSM-CTC v4의 토크나이저 단위. 50k vocab이라 BPE일 가능성이 높고, BPE라면 한 토큰이 여러 음절을 묶어 **음절 단위 카라오케 타이밍을 직접 얻을 수 없다**. 1순위 추천의 전제가 여기 걸려 있으므로 실측 전에 반드시 확인해야 한다(확인 방법 §7).

---

## 1. 라이선스 체인 판정의 기준선

후보별 판정에 반복해서 쓰이는 상류 라이선스를 먼저 확정한다.

### 1.1 AI Hub — 카테고리별로 갈린다 (기존 판정 정정)

AI Hub 공식 이용정책 원문([aihub.or.kr/intrcn/guid/usagepolicy.do](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do)) 확인 결과:

- **AI 허브 개방데이터**: *"지능형 제품・서비스, 챗봇 등 다양한 분야에서 영리적・비영리적 연구・개발 목적으로 활용할 수 있습니다."* → **영리 목적 활용 허용**
- 단서: *"AI 허브에서 제공하는 AI 데이터셋의 판매 등 상업적 이용을 희망하는 경우 수행기관과 별도 협의가 필요합니다."* → 제한 대상은 **데이터셋 자체의 판매**
- **비상업 전용 카테고리(별도)**: 2020·2021 인공지능 온라인 경진대회 데이터, AI Starthon X 네이버 데이터셋, KETI·공항CCTV·오디오북 등 → *"비상업적인 목적의 연구나 개발에 활용될 수 있습니다."*
- **재배포 금지**: *"제공 받은 AI데이터 등을 수행기관 등과 한국지능정보사회진흥원의 승인을 받지 않은 다른 법인, 단체 또는 개인에게 열람하게 하거나 제공, 양도, 대여, 판매하여서는 안됩니다."* → 대상은 **데이터**이며 학습된 가중치를 명시적으로 포함하지는 않는다
- **2차적 저작물 표기 의무**: *"본 AI데이터 등을 이용한 2차적 저작물에도 동일하게 [한국지능정보사회진흥원의 사업결과임을] 밝혀야 합니다."*

**KsponSpeech의 소속 카테고리**: 데이터셋 페이지([dataSetSn=123](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123)) 확인 결과 정식명 `한국어 음성`(KsponSpeech), 구축년도 2018, 수행기관 ETRI, **일반 개방데이터**이며 경진대회 관련 표기 없음. 활용 예시에 "금융 및 보험 등 서비스 자동화"가 포함된다.

**판정**: AI Hub 개방데이터로 학습한 모델을 상업 서비스에서 **추론에 쓰는 것**은 이용정책 문언상 금지되지 않는다. 단, (a) NIA 사업결과 표기 의무가 2차적 저작물에 따라붙고 (b) 데이터 자체는 재배포 불가이며 (c) "내국인만 데이터 신청 가능" 제약이 있다. — **확신도: 중상.** 문언 해석이며, "연구·개발 목적 활용"과 "상용 서비스 배포"의 경계에 대한 NIA 공식 유권해석은 확인하지 못했다.

> 2020 인공지능 온라인 경진대회가 KsponSpeech 파생 데이터를 사용한 것은 사실이나, 그 **경진대회 배포본**이 비상업인 것이고 AI Hub 개방데이터 원본이 비상업이 되는 것은 아니다. 두 배포 경로를 혼동하면 안 된다.

### 1.2 Zeroth-Korean — CC-BY-4.0, 가장 안전

Atlas Guide Inc.(한국)와 Gridspace Inc. 협업 구축, OpenSLR SLR40, 51.6시간 + 테스트 1.2시간. **CC BY 4.0** — 출처표시만으로 상업 이용 명시 허용.
[openslr.org/40](https://openslr.org/40/) · [HF 데이터셋 카드](https://huggingface.co/datasets/kresnik/zeroth_korean) — **확신도: 상**

### 1.3 상류 사전학습 모델

| 베이스 | 라이선스 | 사전학습 데이터 | 비고 |
|---|---|---|---|
| `facebook/wav2vec2-base` | Apache-2.0 | LibriSpeech 960h | 깨끗 |
| `facebook/wav2vec2-xls-r-300m` / `-1b` | Apache-2.0 | *"436k hours of unlabeled speech, including VoxPopuli, MLS, CommonVoice, BABEL, and VoxLingua107"* ([모델 카드](https://huggingface.co/facebook/wav2vec2-xls-r-300m)) | **주의**: VoxPopuli·BABEL은 원 배포처 조건이 별개다. Meta가 전체를 Apache-2.0으로 재배포한 것이 발행자 선언 |
| `facebook/w2v-bert-2.0` | MIT | SeamlessAlign 등, 143개 언어(목록 비공개) | 인코더만 MIT. SeamlessM4T **완성 모델**의 CC-BY-NC와는 별개 |
| `facebook/wav2vec2-large-xlsr-53` | Apache-2.0 | CommonVoice + BABEL + MLS | XLS-R과 동일 유의점 |

XLS-R 계열의 상류 데이터(BABEL은 LDC 라이선스, VoxPopuli는 별도 조건)는 [Track A](track-A-separation.md)·[final-weights-licenses.md](final-weights-licenses.md)에서 확인한 다른 후보들과 **동일한 통상 회색지대**다. 발행자 Meta가 Apache-2.0으로 선언했고 철회·이의제기 사례는 확인되지 않았다. — **확신도: 중.**

### 1.4 확정 배제군

- `facebook/mms-*` 전 계열, `MahmoudAshraf/mms-300m-1130-forced-aligner`, torchaudio `MMS_FA` — **CC-BY-NC-4.0**, 배제 확정
- `espnet/xeus` — **CC-BY-NC-SA-4.0**, 배제 확정
- `SungBeom/stt_kr_conformer_ctc_medium` — HF 태그는 apache-2.0이나 모델 카드 본문이 *"Base Model: RIVA Conformer ASR Korean from NVIDIA"*라고 명시([모델 카드](https://huggingface.co/SungBeom/stt_kr_conformer_ctc_medium)). 원본은 [NGC RIVA Conformer ASR Korean](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/models/speechtotext_ko_kr_conformer)(120M, Conformer-CTC, ASRSet 3500h+)이며 Riva 계열 EULA는 *"You may not sell, rent, sublicense, transfer, distribute, modify, or create derivative works of any portion of the SOFTWARE."*([Riva 2.4 EULA 아카이브](https://docs.nvidia.com/deeplearning/riva/archives/2-4-0/eula/index.html))라고 규정한다. **업로더의 Apache-2.0 재선언은 상류 권한 없이 이뤄진 것**으로 보아야 한다 — 기존 판정 유지, 배제 확정

---

## 2. 후보 표

CTC 헤드 유무·vocab 단위는 가능한 한 `vocab.json` 원문 또는 모델 클래스로 확정했다. `(추정)` 표기는 확정하지 못한 것이다.

### 2.1 한국어 전용 CTC

| 모델 ID | 아키텍처 | CTC | 파라미터 | vocab 단위 (실측) | 라이선스 태그 | **실제 라이선스 체인 판정** | 공개 지표 |
|---|---|---|---:|---|---|---|---|
| [`espnet/owsm_ctc_v4_1B`](https://huggingface.co/espnet/owsm_ctc_v4_1B) | E-Branchformer + self-conditioned CTC (encoder-only) | ✅ 확정 | 1.01B | **미확인** (50k vocab, BPE 추정) | cc-by-4.0 | **클린(확정)** — YODAS(CC 라이선스 웹크롤) 재정제, ESPnet이 CC-BY-4.0으로 재배포 | ko CER **6.74** (FLEURS), ko 학습 10.9k시간 |
| [`anantoj/wav2vec2-xls-r-1b-korean`](https://huggingface.co/anantoj/wav2vec2-xls-r-1b-korean) | Wav2Vec2ForCTC | ✅ 확정 | 1B | **한글 음절 1,205** (확정) | apache-2.0 | **클린(확정)** — XLS-R-1b(Apache-2.0) + Zeroth clean(CC-BY-4.0) | Zeroth **WER 4.49** |
| [`slplab/wav2vec2-xls-r-300m_phone-mfa_korean`](https://huggingface.co/slplab/wav2vec2-xls-r-300m_phone-mfa_korean) | Wav2Vec2ForCTC | ✅ 확정 | 300M | **음소 43개** (확정, 로마자 음소셋) | apache-2.0 | **클린(확정, 1점 유보)** — XLS-R-300m + 서울대 자체 코퍼스 108h. 코퍼스는 비공개이며 별도 라이선스 문서 미확인 | **PER 3.88** |
| [`kresnik/wav2vec2-large-xlsr-korean`](https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean) | Wav2Vec2ForCTC | ✅ 확정 | 300M | **한글 음절 1,205** (확정) | apache-2.0 | **클린(확정)** — XLSR-53 + Zeroth(CC-BY-4.0) | Zeroth WER 4.74 / CER 1.78 |
| [`Kkonjeong/wav2vec2-base-korean`](https://huggingface.co/Kkonjeong/wav2vec2-base-korean) **(현행)** | Wav2Vec2ForCTC | ✅ 확정 | 94.4M | **호환자모 54개** (확정, 초·중·종성 미구분) | (태그 없음, 카드 본문 Apache-2.0) | **클린(확정)** — wav2vec2-base(Apache-2.0) + Zeroth(CC-BY-4.0) | Zeroth CER 7.3 · **난곡 정밀도 73.2% (자체 실측)** |
| [`slplab/wav2vec2-XLSR-300m_KoreanPhonene_spoken_by_foreigners`](https://huggingface.co/slplab/wav2vec2-XLSR-300m_KoreanPhonene_spoken_by_foreigners) | Wav2Vec2ForCTC | ✅ 확정 | 300M | **조합형 자모 82개** (확정, U+11xx 초/중/종성 **구분**) | **태그 없음** | **조건부** — 베이스 XLS-R 깨끗, 데이터는 AI Hub(§1.1 적용). HF 라이선스 태그 자체가 없어 배포자 의사 미표명 | PER 3.15 / CER 2.30 |
| [`w11wo/wav2vec2-xls-r-300m-korean`](https://huggingface.co/w11wo/wav2vec2-xls-r-300m-korean) | Wav2Vec2ForCTC | ✅ 확정 | 300M | 미확인 | apache-2.0 | **클린(확정)** — XLS-R-300m + Zeroth | Zeroth WER 29.54 / CER 9.53 — **성능 열위** |
| [`Miniijune/...-Korean-children-pronunciation-jamo-based`](https://huggingface.co/Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation-jamo-based) | Wav2Vec2ForCTC | ✅ 확정 | 300M | 자모 (추정, 모델명 근거) | apache-2.0 | **조건부** — 학습 데이터 *"More information needed"*로 미기재 | CER 10.84 |
| [`slplab/wav2vec2_xlsr50k_korean_phoneme_aihub-40m`](https://huggingface.co/slplab/wav2vec2_xlsr50k_korean_phoneme_aihub-40m) | Wav2Vec2ForCTC | ✅ 확정 | 300M | 음소 46 (MFA-v1 phoneset) | apache-2.0 | **조건부** — AI Hub 자유대화(§1.1). **학습량 41분**으로 실용성 낮음 | 미공개 (*"will be updated soon"*) |
| [`HERIUN/w2v-bert-2.0-korean-colab-CV16.0`](https://huggingface.co/HERIUN/w2v-bert-2.0-korean-colab-CV16.0) | Wav2Vec2-BERT, AutoModelForCTC | ✅ 확정 | 0.6B | 미확인 | mit | **클린(확정)** — w2v-BERT 2.0(MIT) + CommonVoice 16(CC0) | **미공개** (*"More information needed"*) |
| [`team-lucid/hubert-{base,large,xlarge}-korean`](https://huggingface.co/team-lucid/hubert-base-korean) | HuBERT **SSL 인코더** | ❌ **없음** | 95M~ | — | apache-2.0 | 조건부 — AI Hub 3종 4,000시간 | 파인튜닝 전제. 즉시 사용 불가 |

### 2.2 다국어 CTC (한국어 포함)

| 모델 ID | CTC | 파라미터 | 한국어 근거 | 라이선스 | 판정 |
|---|---|---:|---|---|---|
| [`facebook/omniASR-CTC-300M`](https://huggingface.co/facebook/omniASR-CTC-300M) / [`-1B`](https://huggingface.co/facebook/omniASR-CTC-1B) | ✅ | 325M / 975M | `lang_ids.py`에 **`kor_Hang` 존재 확정** ([원문](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)) | Apache-2.0 (모델·코드), 데이터 CC-BY-4.0 | **클린(확정)**. 단 **HF transformers 미지원** — `pip install omnilingual-asr` + fairseq2 필요 |
| [NGC `parakeet-ctc-riva-1-1b-unified-ml-cs-universal`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/models/parakeet-ctc-riva-1-1b-unified-ml-cs-universal) | ✅ | 1.1B | 25개 언어에 **ko-KR 명시** | NVIDIA AI Foundation Models Community License. 카드에 *"This model is ready for commercial use."* | **회색지대** — 카드 문구와 Community License의 "Production Use는 NIM/AI Enterprise 필요" 조항이 충돌. 배포 포맷이 `.nemo`인지 `.riva`인지 미확인(로그인 게이트) |

### 2.3 부적합 판정 (CTC 없음 / 한국어 없음 / 라이선스)

| 모델 | 사유 |
|---|---|
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | 한국어 ko-KR **지원하고** OpenMDW-1.1에 "ready for commercial use"지만 **FastConformer-CacheAware-RNNT, CTC 헤드 없음**. 게다가 학습셋에 NVIDIA Riva 독점 데이터 포함 |
| `nvidia/parakeet-tdt-0.6b-v3`, `nvidia/canary-1b-v2` | **한국어 미지원**(유럽 25개어 전용) |
| `nvidia/stt_XX_conformer_ctc_large` 시리즈 (~66종, CC-BY-4.0, Riva 무관) | **`stt_ko_` 버전이 존재하지 않음**. HF API `search=stt_ko` 결과에 nvidia 네임스페이스 한국어 모델 없음 확인 |
| `CohereLabs/cohere-transcribe-03-2026` (2B, Apache-2.0, 한국어 지원) | **순수 encoder-decoder attention**. 공식 블로그·기술 문서 어디에도 CTC 언급 없음 |
| `Qwen/Qwen3-ASR-0.6B`/`1.7B` (Apache-2.0, 52개어) | AED(attention encoder-decoder). CTC는 데이터 준비 보조용일 뿐 최종 헤드 아님 |
| `Qwen3-ForcedAligner-0.6B` | 논문이 **명시적으로 "CTC의 대안"**으로 포지셔닝. blank·frame posterior 없는 타임스탬프 bin 분류기 |
| `ibm-granite/granite-speech-4.1-2b` | dual-head CTC 있으나 **한국어 미지원**(en/fr/de/es/pt/ja) |
| `espnet/xeus` | CC-BY-NC-SA-4.0 |
| `microsoft/Phi-4-multimodal-instruct` | 5.6B로 파라미터 상한 초과 |
| `eesungkim/stt_kr_conformer_transducer_large` | CC-BY-4.0·Riva 무관·KsponSpeech 965h로 체인은 깨끗하나 **Transducer(RNN-T), CTC 아님** |
| `ddwkim/asr-conformer-transformerlm-ksponspeech` | hybrid CTC/attention이라 구조는 가능하나 **SpeechBrain 프레임워크**. NFA 직결 불가 |
| ETRI, Naver Clova(NEST), 리턴제로, 셀바스AI | **API 전용, 가중치 미공개** |
| `skt/A.X-K2-ALM` | char-CTC 헤드 존재하나 **가중치 미공개("coming soon") + h-research 연구전용 라이선스** |
| `seny1004`/`weekcircle`의 `wav2vec2-large-mms-1b-korean-*` | MMS 파생 → **cc-by-nc-4.0** |

---

## 3. 3분류

### ① 즉시 실측 가치 있음

1. **`espnet/owsm_ctc_v4_1B`** — CC-BY-4.0 확정, 한국어 10.9k시간, `ctc-segmentation` 기반 forced alignment 예제가 모델 카드에 이미 있음
2. **`anantoj/wav2vec2-xls-r-1b-korean`** — Apache-2.0 확정, 현행 kkonjeong과 데이터 동일·용량 10배, 음절 vocab
3. **`slplab/wav2vec2-xls-r-300m_phone-mfa_korean`** — Apache-2.0 확정, 43음소 vocab, PER 3.88%
4. **`kresnik/wav2vec2-large-xlsr-korean`** — Apache-2.0 확정, anantoj의 300M 대조군(같은 vocab·같은 데이터라 용량 스케일링 곡선을 얻을 수 있음)
5. **`facebook/omniASR-CTC-300M/1B`** — Apache-2.0 확정, `kor_Hang` 확정. fairseq2 어댑터 비용은 있으나 [기존 어댑터 작업](track-B-ctc-model.md)과 겹침

### ② 조건부 (라이선스 확인 필요 또는 성능 미지수)

- **`slplab/wav2vec2-XLSR-300m_KoreanPhonene_spoken_by_foreigners`** — 조합형 자모 82개 vocab은 **일본어 음차에 OOV가 원천적으로 없는** 구조라 매력적이고 PER 3.15%로 우수하지만, **HF 라이선스 태그가 아예 없다**. 배포자(SNU SLP Lab) 의사 표명이 없어 채택 전 문의 필요. 또한 학습 데이터가 **외국인 화자의 한국어**라 도메인이 크게 다르다
- **`HERIUN/w2v-bert-2.0-korean-colab-CV16.0`** — MIT + CommonVoice로 체인은 가장 깨끗한 축이나 이름에 `colab`이 붙은 개인 실험물이고 **성능 지표가 하나도 공개되지 않았다**
- **NGC `parakeet-ctc-riva-1-1b-unified-ml-cs-universal`** — Riva EULA는 아니지만 Community License의 Production 조항과 "ready for commercial use" 문구가 충돌. 배포 포맷도 미확인. **NGC 로그인 후 파일 목록 확인 + 라이선스 PDF 정독이 선행 조건**
- **`Miniijune/...jamo-based`**, **`thisisHJLee/*`**, **`Hyuk/wav2vec2-korean-v*`** — 학습 데이터 미기재. 체인 추적 불가

### ③ 부적합

§2.3 표 전체. 요약하면 — **CTC 헤드 없음**(nemotron-3.5, Cohere transcribe, Qwen3-ASR, Qwen3-ForcedAligner, canary, eesungkim, team-lucid HuBERT), **한국어 미지원**(parakeet-tdt-v3, canary-1b-v2, granite-speech, nvidia stt_XX 시리즈 전체), **라이선스 배제**(MMS 전 계열, xeus, SungBeom/Riva), **가중치 미공개**(ETRI, Clova, SKT A.X-K2-ALM, 리턴제로, 셀바스).

---

## 4. 상위 3 추천 — "왜 이게 nemo를 대체할 수 있는가"

지금까지의 실측(nemo 82.4% > kkonjeong+한글음차 73.2% > omniasr 네이티브 64.4%)에서 nemo의 우위를 만든 요인은 셋 중 하나다. 후보를 이 셋에 각각 대응시켜 골랐다.

**가설 A — 학습량**: nemo는 한국어 13,946시간, kkonjeong은 Zeroth 51.6시간. 278배 차이다.
→ **`espnet/owsm_ctc_v4_1B` (한국어 10.9k시간)**. 클린 라이선스 후보 중 이 자릿수에 도달한 **유일한** 모델이다. 다른 어떤 후보도 100시간을 넘지 않는다. 가설 A가 맞다면 이것 말고는 답이 없고, 틀리다면 다른 두 후보가 더 싸게 같은 성능을 낸다. **가장 먼저 돌려서 가설을 가르는 실험이어야 한다.**
CC-BY-4.0이고 모델 카드에 `ctc-segmentation` forced alignment 예제가 이미 있어 착수 비용도 낮다.

**가설 B — 모델 용량**: nemo는 Conformer medium(약 120M 상속) 위에 대규모 파인튜닝, kkonjeong은 wav2vec2-**base** 94.4M.
→ **`anantoj/wav2vec2-xls-r-1b-korean`**. **데이터를 Zeroth로 고정한 채 용량만 94.4M → 1B로 올린 통제 실험**이 된다. `kresnik`(300M, 동일 데이터·동일 1,205 음절 vocab)을 중간점으로 넣으면 94.4M → 300M → 1B 스케일링 곡선을 한 번에 얻는다. [Track B가 "확인하지 못했다"고 남긴 항목](track-B-ctc-model.md)이 바로 이 곡선이다. Zeroth WER 4.49%로 kresnik(4.74%)보다 낫고, fp16이면 약 1.9GB로 VRAM 4GB 예산에 들어간다.

**가설 C — vocab 설계와 posterior 집중도**: 기존 붕괴 분석대로 합성보컬 파국의 조건은 posterior 바닥 + blank 우세이고, star 성형이 무효였던 이유도 blank가 무료 필러로 동작하기 때문이었다. CTC에서 클래스 수가 많을수록 확률질량이 분산되고 blank가 이긴다.
→ **`slplab/wav2vec2-xls-r-300m_phone-mfa_korean`, vocab 43개**. kkonjeong 54개보다도 적고, 음절 모델 1,205개의 **28분의 1**이다. 클래스가 적을수록 target token에 질량이 몰려 blank 경쟁에서 살아남는다 — 합성보컬 posterior 바닥에 구조적으로 가장 강할 후보다. PER 3.88%(ICPhS 2023, [arXiv 2306.10821](https://arxiv.org/abs/2306.10821))로 절대 성능도 확인됐다.
**통합 비용**: 한글 → 이 43음소 셋으로 가는 G2P가 필요하다. vocab 실측값은 `A B BB CHh D DD E EO EU G GG H I J JJ Kh L M N NG O Ph R S SS Th U euI iA iE iEO iO iU k oA oE p t uEO uI |` 로, 종성이 `k/p/t/L/M/N/NG` 7종으로 중화된 표준 한국어 음운 체계다. 자모 → 음소 결정론적 매핑으로 충분하며 **연음·경음화 같은 형태음운 규칙은 적용하면 안 된다**(일본어 음차 한글에는 해당 규칙이 성립하지 않음).

---

## 5. vocab 단위 실측 — 정렬 설계에 직결되는 발견

이전 조사에서 "미확인"으로 남았던 항목을 `vocab.json` 원문으로 확정했다. 강제정렬 품질을 좌우하므로 별도로 정리한다.

| 모델 | 토큰 수 | 체계 | 일본어 음차 OOV 위험 |
|---|---:|---|---|
| `slplab/...phone-mfa_korean` | **43** | 로마자 음소 (종성 7종 중화) | **없음** (G2P가 전사) |
| `Kkonjeong/wav2vec2-base-korean` | **54** | **호환자모** U+31xx. 초성 ㄱ과 종성 ㄱ을 **구분하지 않음**, 복합종성 ㄳㄵㄶㄺ… 별도 토큰 | **없음** (자모 분해로 전 음절 표현 가능) |
| `slplab/...spoken_by_foreigners` | **82** | **조합형 자모** U+11xx. 초성 `ᄀ`·중성 `ᅡ`·종성 `ᆨ`을 **구분** | **없음** |
| `kresnik`, `anantoj` | **1,205** | **한글 음절 완성형** | **있음 — 부분 커버** |

**음절 vocab의 OOV 실측**: `anantoj`의 정렬된 vocab을 직접 조회한 결과 `츠·쿄·큐·튜·퓨·쿠·코·타·토`는 모두 존재한다. 그러나 `캬`는 ㅋ 초성 48개 토큰(`카 칸 칼 캉 캐 캔 캘 캠 커 … 크 큰 클 큼 키 킥 킨 킬 킷 킹`) 안에 없다. 일본어 요음(きゃ·ひゃ·びゃ·みゃ·にゃ·りゃ 계열)과 촉음 받침 조합에서 **부분적 OOV가 발생한다**. OOV는 `[UNK]`로 떨어져 해당 음절의 정렬이 소실되므로 실측 전에 반드시 대조해야 한다.

**이것이 순위에 미치는 영향**: 자모/음소 vocab 후보(slplab, Kkonjeong)는 OOV가 원천적으로 없다. 음절 vocab 후보(anantoj, kresnik)는 카라오케 음절 타이밍을 **어셈블리 없이 바로** 얻는다는 큰 장점이 있는 대신 이 리스크를 안는다. 둘 다 실측 대상에 넣되, anantoj/kresnik은 §7의 커버리지 체크를 통과한 뒤에 평가해야 결과가 해석 가능하다.

---

## 6. VRAM 추정

파라미터 수 × 정밀도로 계산한 **가중치 크기**이며 런타임 peak가 아니다(emission 버퍼·CTC DP·긴 곡 chunk가 추가된다).

| 후보 | 파라미터 | FP32 | FP16 | 4GB 예산 |
|---|---:|---:|---:|---|
| `Kkonjeong` (현행) | 94.4M | 0.38GB | 0.19GB | 여유 |
| `slplab phone-mfa` / `kresnik` | 300M | 1.2GB | 0.6GB | 여유 |
| `anantoj` | 1B | 3.7GB | 1.9GB | **FP16 권장** |
| `owsm_ctc_v4_1B` | 1.01B | 3.8GB | 1.9GB | **FP16 권장** |
| `omniASR-CTC-300M` | 325M | 1.3GB | — | 공식 추정 추론 VRAM 약 2GiB (A100·BF16·30초·batch1) |

---

## 7. 미확인 사항과 확인 방법

실측 착수 전에 처리해야 할 순서대로.

1. **OWSM-CTC v4의 토크나이저 단위** — 1순위 추천의 전제. 논문(Table 1·3)에 vocab 종류 기재 없음.
   → 확인: `https://huggingface.co/espnet/owsm_ctc_v4_1B/tree/main`에서 `bpe.model`/`tokens.txt`/`config.yaml`의 `token_type` 필드 조회. BPE로 확인되면 음절 타이밍은 CTC 경로 내부에서 재추정해야 하므로 2·3순위를 먼저 돌리는 편이 낫다.
2. **음절 vocab의 일본어 음차 커버리지** — GPU 불필요, 수초.
   → 확인: `anantoj`/`kresnik`의 `vocab.json`을 받아, DB에 이미 있는 ja 채택 곡의 한글 음차 전체를 문자 집합으로 만들어 차집합을 구한다. OOV 음절이 하나라도 나오면 그 곡은 평가에서 제외하거나 자모 후보와만 비교해야 한다.
3. **`slplab phone-mfa`의 정확한 G2P 규칙** — vocab 43개는 확정했으나 어떤 한글→음소 변환기로 학습 전사를 만들었는지가 카드에 없다. 모델명의 `mfa`는 MFA 한국어 phone set을 시사하나 실제 토큰은 IPA가 아닌 로마자 표기라 MFA 표준과 다르다.
   → 확인: ICPhS 2023 논문([arXiv 2306.10821](https://arxiv.org/abs/2306.10821)) 본문의 phone set 정의(자음 19·모음 17·변이음 4 = 40) 정독. 필요시 [MFA `korean_jamo_mfa` G2P](https://montreal-forced-aligner.readthedocs.io/) 대조.
4. **`slplab spoken_by_foreigners`의 라이선스** — HF 태그 없음.
   → 확인: SNU SLP Lab에 직접 문의, 또는 HF discussions 개설.
5. **NGC parakeet-ctc 다국어의 배포 포맷과 Production 조항** — 로그인 게이트로 미확인.
   → 확인: NGC 계정 로그인 후 File Browser로 `.nemo`/`.riva` 확인 + `NVIDIA-Models-Community-License` PDF 원문에서 "Production Use" 정의 정독. `.riva`뿐이면 NFA 직결이 불가하므로 후보에서 빠진다.
6. **AI Hub 개방데이터의 "상용 서비스 배포" 유권해석** — 문언상 허용으로 읽히나 공식 해석 미확인.
   → 확인: NIA/AI Hub(aihub@aihub.kr)에 서면 질의. `slplab spoken_by_foreigners`·`team-lucid` 계열을 채택할 경우에만 필요하다. **Zeroth 기반 후보(anantoj·kresnik·Kkonjeong)와 OWSM은 이 질의와 무관하다** — 이것이 상위 3 추천을 Zeroth/YODAS 계열로 고른 이유이기도 하다.
7. **XLS-R 상류 데이터(BABEL·VoxPopuli)** — Meta의 Apache-2.0 선언 외 상류 조건은 미확인. [final-weights-licenses.md](final-weights-licenses.md)에서 정리한 분리기 후보들과 동일한 성격의 회색지대이며, 리포 정책상 "명백한 NC만 제외" 기준에서는 실측 포함 대상이다.

---

## 8. 권장 실측 순서

1. §7-2 vocab 커버리지 체크(로컬, GPU 불필요) → 음절 후보 평가 가능 여부 확정
2. §7-1 OWSM 토크나이저 확인 → 1순위 유효성 확정
3. `anantoj`(1B) + `kresnik`(300M) + 현행 `Kkonjeong`(94.4M) 3점 스케일링 — **데이터·vocab이 동일해 용량 효과만 분리된다.** 가장 해석이 쉬운 실험이므로 먼저 돌린다
4. `espnet/owsm_ctc_v4_1B` — 학습량 가설 검증
5. `slplab phone-mfa` — G2P 어댑터 작성 후. 합성보컬 붕괴곡 서브셋에서 posterior 집중도를 별도 지표로 측정
6. 기존 A/B 프로토콜([Track B §6.3](track-B-ctc-model.md))의 붕괴 지표(forced path 실패율·평균 target log prob·blank 비율)를 hard gate로 유지

**어떤 후보도 "품질 비회귀"가 조사만으로 보증되지 않는다.** nemo 82.4%를 넘거나 최소한 근접하는지는 동일 곡·동일 전처리 실측으로만 판정 가능하며, 특히 합성보컬 붕괴율은 공개 CER/PER과 상관이 없다.
