# 과제 B — CTC 강제정렬 음향모델 교체 (codex gpt-5.6-luna, 2026-07-30)

> 3갈래 모델 교체 조사 중 B 트랙 보고 원문. 선행: [2026-07-26 통합 조사](../2026-07-26-alignment-papers-patents-survey.md) §8

# Everyric2 CTC 강제정렬 음향모델 교체 조사 보고서

조사 기준일: 2026-07-30  
판정 기준: 상업 사용 가능성, 문자/자모 단위 정렬 가능성, VRAM, 노래·합성보컬 강건성, 기존 품질 회귀 위험

## 요약 결론

- `facebook/mms-1b-all`은 Hugging Face 모델 카드 기준으로 현재도 `CC-BY-NC-4.0`입니다. 상업 서비스의 기본 모델로는 부적합합니다. [MMS 모델 카드](https://huggingface.co/facebook/mms-1b-all) — 확신도: 상
- 한국어 1순위 후보는 `Kkonjeong/wav2vec2-base-korean`입니다. Apache-2.0, 94.4M 파라미터, 명시적인 한국어 자모 CTC 출력이라 현재의 “한국어 발음 한글 변환 → 강제정렬” 경로와 가장 잘 맞습니다. [Kkonjeong 모델 카드](https://huggingface.co/Kkonjeong/wav2vec2-base-korean) — 확신도: 상
- 일본어 소형 후보는 `reazon-research/japanese-hubert-base-k2-rs35kh`입니다. Apache-2.0, 98.4M, CTC 모델이며 공개 음성 CER도 우수합니다. 다만 정확한 가사 토큰화 단위는 반드시 직접 확인해야 합니다. [Reazon HuBERT 모델 카드](https://huggingface.co/reazon-research/japanese-hubert-base-k2-rs35kh) — 확신도: 중
- 다국어 후보로는 `facebook/omniASR-CTC-300M`이 가장 유망하지만, 현재 everyric2의 torchaudio Wav2Vec2 CTC 경로에 바로 꽂을 수 있는지, 한국어·일본어의 정확한 출력 vocab이 문자 단위인지 확인되지 않았습니다. [Omnilingual ASR 공식 저장소](https://github.com/facebookresearch/omnilingual-asr) — 확신도: 중
- 모델 크기 감소가 강제정렬 타이밍 품질을 보존한다는 직접적인 1B→300M→95M 비교 문헌은 확인하지 못했습니다. 특히 보카로·합성보컬에 대한 후보별 비교 실측도 확인하지 못했습니다.
- 따라서 즉시 교체는 권장하지 않습니다. 후보 모델을 shadow A/B로 운영하고, 합성보컬 붕괴율을 포함한 비회귀 게이트를 통과한 모델만 단계적으로 승격하는 것이 안전합니다.

---

## 1. `facebook/mms-1b-all` 최신 라이선스 확인

### 확인 결과

Hugging Face 원문 모델 카드에는 다음과 같이 명시되어 있습니다.

- 모델 라이선스: `CC-BY-NC-4.0`
- 파라미터: 약 1B
- 가중치 타입: F32
- 1,162개 언어 어댑터
- Wav2Vec2 기반 CTC ASR 모델

[facebook/mms-1b-all Hugging Face 모델 카드](https://huggingface.co/facebook/mms-1b-all) — 확신도: 상

MMS의 기반 모델인 `facebook/mms-1b`도 동일하게 `CC-BY-NC-4.0`으로 표시됩니다. [facebook/mms-1b 모델 카드](https://huggingface.co/facebook/mms-1b/tree/main) — 확신도: 상

### 상업 사용 판단

`CC-BY-NC-4.0`의 NC는 비상업적 사용만 허용한다는 의미이므로, everyric2를 상업 서비스로 운영할 경우 모델 가중치의 상업 사용은 허용되지 않는 것으로 판단해야 합니다. 이는 법률 자문이 아니라 모델 카드와 라이선스 표기에 근거한 기술·법무 리스크 판단입니다.

별도의 Hugging Face `LICENSE` 파일 원문은 이번 접근에서 정상 확인하지 못했습니다. 다만 모델 카드의 라이선스 표기는 명확합니다.

**결론:** 사용자의 “MMS 계열 가중치가 CC-BY-NC 4.0일 수 있다”는 인식은 현재 Hugging Face 모델 카드 기준으로 정확합니다.

---

## 2. 한국어 단일언어 CTC 후보

### 2.1 강제정렬에서 vocab 단위가 중요한 이유

torchaudio의 CTC forced alignment는 음향모델이 출력하는 frame별 emission과 입력 transcript를 동일한 token ID 시퀀스로 변환한 뒤 정렬합니다. 따라서 transcript의 각 기호가 모델 vocab에 존재해야 합니다.

[torchaudio CTC forced alignment 문서](https://docs.pytorch.org/audio/2.1/tutorials/ctc_forced_alignment_api_tutorial.html) — 확신도: 상

정확히 말하면 반드시 “문자 단위”여야 하는 것은 아닙니다. 다음 모두 가능합니다.

- 한글 음절
- 한글 자모
- 분해 자모 Unicode
- 일본어 가나·한자
- 음소·IPA
- BPE·SentencePiece subword

다만 가사 라인·단어·문자 타이밍을 안정적으로 얻으려면 자모·음절·음소처럼 경계가 분명한 token이 유리합니다. SentencePiece처럼 한 token이 여러 문자를 포함하면 내부 문자별 시간을 다시 추정해야 합니다. 이 부분은 torchaudio 정렬 알고리즘의 token 단위 동작에서 직접 도출되는 설계 판단입니다. [torchaudio CTC 문서](https://docs.pytorch.org/audio/2.4.0/tutorials/ctc_forced_alignment_api_tutorial.html) — 확신도: 상

### 2.2 후보 비교

| 모델 | CTC / 크기 | 라이선스 | vocab 단위 | 평가·주의 |
|---|---:|---|---|---|
| `Kkonjeong/wav2vec2-base-korean` | Wav2Vec2 CTC, 94.4M | Apache-2.0 | 한국어 자모. 카드에 text를 jamo로 변환한다고 명시 | Zeroth CER 7.3%, 단 음성 데이터 평가 |
| `kresnik/wav2vec2-large-xlsr-korean` | Wav2Vec2 CTC, 0.3B | Apache-2.0 | 카드에는 미명시. 공개 `vocab.json`과 보조 분석상 한글 음절 단위로 보는 것이 타당 | Zeroth WER 4.74%, CER 1.78% |
| `w11wo/wav2vec2-xls-r-300m-korean` | Wav2Vec2 CTC, 300M | Apache-2.0 | 원문 카드에서 자모·음절 여부 확인 못 함 | Zeroth WER 29.54%, CER 9.53% |
| `thisisHJLee/wav2vec2-large-xls-r-300m-korean-g` | Wav2Vec2 CTC, 300M | Apache-2.0 | 원문 카드에서 명확히 확인 못 함 | 공개 CER 0.1674, 문서화 부족 |
| `Taeham/wav2vec2-ksponspeech` | Wav2Vec2 CTC, 약 300M급 | Apache-2.0 | 확인 못 함 | KsponSpeech 기반, 오래된 카드 |
| `42MARU/ko-42maru-wav2vec2-conformer-del-1s` | Wav2Vec2 Conformer CTC | Apache-2.0 | 분해 자모 Unicode 출력. NFC 정규화 필요 | KsponSpeech, WER 21.52 |
| `SungBeom/stt_kr_conformer_ctc_medium` | NeMo Conformer CTC | Apache-2.0 | BPE 계열로 보이나 정확한 vocab 검증 필요 | 491MB 모델 파일, 2023 공개 |
| `Hyuk/wav2vec2-korean-v2/v3` | Wav2Vec2 CTC | Apache-2.0 | 확인 못 함 | 데이터·학습 설명이 부족하여 우선순위 낮음 |

주요 원문: [Kkonjeong](https://huggingface.co/Kkonjeong/wav2vec2-base-korean), [kresnik](https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean), [w11wo](https://huggingface.co/w11wo/wav2vec2-xls-r-300m-korean), [thisisHJLee](https://huggingface.co/thisisHJLee/wav2vec2-large-xls-r-300m-korean-g), [Taeham](https://huggingface.co/Taeham/wav2vec2-ksponspeech), [42MARU](https://huggingface.co/42MARU/ko-42maru-wav2vec2-conformer-del-1s), [SungBeom](https://huggingface.co/SungBeom/stt_kr_conformer_ctc_medium) — 확신도: 상~중

### 2.3 `Kkonjeong/wav2vec2-base-korean`

이 모델은 다음 조건을 명시적으로 만족합니다.

- `Wav2Vec2ForCTC`
- Apache-2.0
- 94.4M 파라미터
- Zeroth-Korean fine-tuning
- 특수문자 제거 후 텍스트를 자모로 변환
- 모델 출력도 한국어 자모 기반

[Kkonjeong 모델 카드](https://huggingface.co/Kkonjeong/wav2vec2-base-korean) — 확신도: 상

따라서 현재 everyric2의 한국어 발음 한글 경로에서 다음과 같이 사용할 수 있습니다.

```text
결정론적 발음 변환
→ Kkonjeong이 학습한 동일한 자모 정규화
→ CTC emission
→ forced alignment
```

단, 모델 카드의 CER은 Zeroth 음성에 대한 결과일 뿐이며 노래나 합성보컬 성능을 의미하지 않습니다. 모델 카드도 대표 도메인 샘플을 별도로 평가하라고 권고합니다. [Kkonjeong 모델 카드](https://huggingface.co/Kkonjeong/wav2vec2-base-korean) — 확신도: 상

### 2.4 `kresnik/wav2vec2-large-xlsr-korean`

이 모델은 Apache-2.0, 0.3B 파라미터, Wav2Vec2 CTC이며 Zeroth 평가 수치는 공개 후보 중 매우 좋습니다. [kresnik 모델 카드](https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean) — 확신도: 상

다만 모델 카드에는 자모인지 음절인지가 명시되어 있지 않습니다. 저장소에는 `vocab.json`이 제공되며, 공개된 vocab 분석 자료에서는 한글 음절 블록 중심의 약 1,203개 vocab으로 설명됩니다. [kresnik 저장소 파일 목록](https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean/tree/main), [vocab 분석 자료](https://coolseaweed.com/4) — 확신도: 중

**판정:** 음절 단위 모델로 보는 것이 합리적이지만, 원문 모델 카드만으로는 확정할 수 없습니다. everyric2의 발음 변환기가 자모를 출력한다면 추가 음절 조립 단계가 필요합니다.

### 2.5 Zeroth-Korean·KsponSpeech 라이선스 주의

Kkonjeong 모델 자체는 Apache-2.0이지만, 학습 데이터인 `kresnik/zeroth_korean`은 Hugging Face 데이터셋 카드에서 CC-BY-4.0으로 표시됩니다. [Zeroth-Korean 데이터셋 카드](https://huggingface.co/datasets/kresnik/zeroth_korean) — 확신도: 상

CC-BY-4.0은 적절한 attribution 조건하에 상업적 사용을 허용합니다. [Creative Commons CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) — 확신도: 상

반면 KsponSpeech 원본 AIHub 데이터의 상업적 재사용 조건은 이번 조사에서 모델 카드만으로 확정하지 못했습니다. 따라서 “모델 카드가 Apache-2.0이므로 모든 학습 데이터 권리까지 해결되었다”고 보면 안 됩니다.

### 2.6 한국어 추천

**1순위: `Kkonjeong/wav2vec2-base-korean`**

이유:

- 95M급으로 VRAM 절감 효과가 큼
- 자모 vocab이 명시되어 있음
- 현재 한국어 발음 변환 경로와 token 설계가 잘 맞음
- Apache-2.0
- 모델 구조가 Wav2Vec2 CTC라 기존 구현과의 이식 난도가 낮음

**실험 후보: `42MARU/ko-42maru-wav2vec2-conformer-del-1s`**

분해 자모를 출력한다는 점은 매력적이지만, 42MARU 자체 표기 규칙과 everyric2 발음 표기 규칙이 다를 수 있고 평가 품질도 Kkonjeong보다 불리합니다. [42MARU 모델 카드](https://huggingface.co/42MARU/ko-42maru-wav2vec2-conformer-del-1s) — 확신도: 중

**대체 후보: `kresnik/wav2vec2-large-xlsr-korean`**

음성 품질이 중요하고 음절 vocab 변환을 수용할 수 있다면 비교 대상에 넣을 가치가 있습니다. 다만 300M급이며 노래·보카로 검증이 없습니다.

---

## 3. 일본어 경로 후보

### 3.1 기존 후보: `jonatasgrosman/wav2vec2-large-xlsr-53-japanese`

확인된 사항:

- Wav2Vec2ForCTC
- Apache-2.0
- Common Voice 6.1, CSS10, JSUT fine-tuning
- 16kHz 입력
- Hugging Face 카드의 self-reported Common Voice WER 81.8, CER 20.16

[Jonatas 일본어 모델 카드](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-japanese) — 확신도: 상

WER 수치는 오래된 카드의 self-reported 값이며, 강제정렬 타이밍 품질을 직접 의미하지 않습니다. 또한 모델 카드에는 일본어 vocab이 한자·히라가나·가타카나 중 정확히 어떤 단위로 구성되었는지 명시되어 있지 않습니다.

**판정:** 라이선스 측면에서는 사용하기 좋지만, 직접 한자 정렬 경로에 투입하기 전 `vocab.json`과 모든 가사 문자 coverage를 검사해야 합니다.

### 3.2 `ttop324/wav2vec2-live-japanese`

이 모델은 일본어 텍스트를 히라가나로 변환하여 학습한 CTC 모델입니다. 모델 카드 검색 결과에는 100개 vocab과 히라가나 학습이 명시되어 있습니다. [ttop324 모델 카드](https://huggingface.co/ttop324/wav2vec2-live-japanese) — 확신도: 중

- Apache-2.0
- CTC Wav2Vec2
- 한자·가타카나를 히라가나로 변환
- Common Voice, JSUT, CSS10, TEDxJP-10K, JVS, JSSS 활용
- self-reported WER 21.48, CER 9.82

**적합한 경우:** 일본어 가사를 직접 한자로 정렬하지 않고, 기존 일본어 발음 변환기를 통해 히라가나로 정규화할 수 있는 경우입니다.

### 3.3 ReazonSpeech 계열

#### `reazon-research/japanese-hubert-base-k2-rs35kh`

- HuBERTForCTC
- Apache-2.0
- 98.4M 파라미터
- ReazonSpeech v2.0 기반
- 공개 CER 평균 11.23
- JSUT 9.94, Common Voice 11.59, TEDxJP 12.18

[Reazon HuBERT Base 모델 카드](https://huggingface.co/reazon-research/japanese-hubert-base-k2-rs35kh) — 확신도: 상

소형 모델 중 공개 음성 평가가 가장 강한 후보로 보입니다. 다만 tokenizer 설정은 CTC tokenizer임을 보여주지만, everyric2에서 필요한 “문자별 정렬 가능성”은 별도 확인이 필요합니다. [Reazon tokenizer 설정](https://huggingface.co/reazon-research/japanese-hubert-base-k2-rs35kh/blob/main/tokenizer_config.json) — 확신도: 상

#### `reazon-research/japanese-wav2vec2-base-rs35kh`

- Wav2Vec2ForCTC
- Apache-2.0
- 96.7M 파라미터
- 공개 CER 평균 20.40

[Reazon Wav2Vec2 Base 모델 카드](https://huggingface.co/reazon-research/japanese-wav2vec2-base-rs35kh) — 확신도: 상

#### `reazon-research/japanese-wav2vec2-large-rs35kh`

- Wav2Vec2ForCTC
- Apache-2.0
- 319M 파라미터
- 공개 CER 평균 16.25
- Base보다 긴 발화 평가에서 유리

[Reazon Wav2Vec2 Large 모델 카드](https://huggingface.co/reazon-research/japanese-wav2vec2-large-rs35kh) — 확신도: 상

Reazon 공식 벤치마크는 Base 모델이 평균 CER에서 좋은 결과를 냈다고 보고합니다. 단, 테스트셋 일부는 다른 모델의 학습 데이터와 겹칠 가능성이 있다고 명시합니다. [Reazon 공식 벤치마크](https://research.reazon.jp/blog/2024-10-21-Wav2Vec2-base-release.html) — 확신도: 상

### 3.4 Reazon 데이터 라이선스 주의

Reazon 모델 가중치는 Apache-2.0으로 표시되지만, ReazonSpeech 데이터셋은 CDLA-Sharing-1.0과 일본 저작권법 제30조의4 범위에 관한 별도 조건을 갖습니다. [ReazonSpeech 공식 프로젝트](https://research.reazon.jp/projects/ReazonSpeech/index.html), [ReazonSpeech 데이터셋 카드](https://huggingface.co/datasets/reazon-research/reazonspeech) — 확신도: 상

**결론:** 모델 라이선스만으로 상업적 법적 검토가 끝났다고 판단하면 안 됩니다.

### 3.5 일본어 추천

#### 직접 한자·가나 정렬을 유지하는 경우

1. 기존 `jonatasgrosman/wav2vec2-large-xlsr-53-japanese`
2. `reazon-research/japanese-wav2vec2-large-rs35kh`

두 모델을 먼저 A/B 비교하는 것이 현실적입니다. Reazon Large가 공개 음성 CER에서는 유리하지만, 기존 everyric2의 일본어 token normalization과 정확히 맞는지는 확인이 필요합니다.

#### 히라가나 정규화를 허용하는 경우

1. `reazon-research/japanese-hubert-base-k2-rs35kh`
2. `ttop324/wav2vec2-live-japanese`

VRAM을 우선하면 Reazon HuBERT Base가 가장 유망합니다.

#### 확인하지 못한 후보

`sakasegawa/japanese-wav2vec2-large-hiragana-ctc`는 Apache-2.0 일본어 CTC 후보로 검색되었으나, 이번 조사에서는 Hugging Face 원문 페이지가 정상적으로 열리지 않아 모델 크기·vocab·평가 결과를 확인하지 못했습니다. [sakasegawa 모델 페이지](https://huggingface.co/sakasegawa/japanese-wav2vec2-large-hiragana-ctc) — 확신도: 하

---

## 4. 다국어 대안

### 4.1 `MahmoudAshraf/mms-300m-1130-forced-aligner`

이 모델은 MMS-300M을 forced alignment 용도로 변환한 모델이며 다음을 명시합니다.

- 1,130개 언어
- AutoModelForCTC
- torchaudio MMS checkpoint 변환
- `CC-BY-NC-4.0`

[MahmoudAshraf MMS forced aligner 모델 카드](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner) — 확신도: 상

따라서 상업 서비스 후보에서 제외해야 합니다. “MMS 파생이므로 NC일 것”이라는 추론이 필요하지 않고, 해당 모델 카드가 직접 NC를 표시합니다.

### 4.2 OWSM-CTC

`espnet/owsm_ctc_v4_1B`는 다음을 제공합니다.

- 약 1.01B 파라미터
- multilingual ASR
- CTC 기반
- `ctc-segmentation`을 이용한 forced alignment 예제
- CC-BY-4.0

[OWSM-CTC v4 모델 카드](https://huggingface.co/espnet/owsm_ctc_v4_1B) — 확신도: 상

CC-BY-4.0은 attribution 조건을 지키면 상업적 사용을 허용합니다. [Creative Commons CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) — 확신도: 상

다만 OWSM-CTC는 everyric2의 MMS/Wav2Vec2 모델과 동일한 drop-in 모델로 보기 어렵습니다.

- 언어·task token을 별도로 사용
- ESPnet 추론 경로 필요
- 논문에서 CTC 시간 다운샘플링이 약 80ms 단위로 설명됨
- 1B급이라 VRAM 절감 목적과 맞지 않음

[OWSM 논문](https://aclanthology.org/2024.acl-long.549.pdf) — 확신도: 상

**판정:** 연구용 비교 대상이지 9GB 피크 예산을 가진 everyric2의 1차 교체 후보는 아닙니다.

### 4.3 Meta Omnilingual ASR CTC

`facebook/omniASR-CTC-300M`은 현재 가장 주목할 만한 상업 친화적 다국어 후보입니다.

공식 저장소 기준:

| 모델 | 파라미터 | FP32 다운로드 | 공식 추정 inference VRAM |
|---|---:|---:|---:|
| CTC 300M | 325M | 1.3 GiB | 약 2 GiB |
| CTC 1B | 975M | 3.7 GiB | 약 3 GiB |

측정 조건은 batch 1, 30초 오디오, BF16, A100입니다. [Omnilingual ASR 공식 저장소](https://github.com/facebookresearch/omnilingual-asr) — 확신도: 상

공식 코드와 모델은 Apache-2.0으로 공개되어 있습니다. [Omnilingual ASR 저장소 LICENSE](https://github.com/facebookresearch/omnilingual-asr/blob/main/LICENSE), [omniASR-CTC-300M 모델 카드](https://huggingface.co/facebook/omniASR-CTC-300M) — 확신도: 상

주의점은 다음과 같습니다.

- 현재 공식 CTC 모델의 한국어·일본어 개별 coverage를 이번 조사에서 직접 확정하지 못함
- official fairseq2 기반 추론 경로가 필요함
- Hugging Face의 제3자 변환 모델은 SentencePiece 계열 9,812 vocab을 사용한다고 설명함
- SentencePiece token은 문자 단위 정렬과 동일하지 않음
- 공식 저장소는 현재 CTC 모델 입력 길이에 제한이 있다고 설명함

[제3자 Hugging Face 변환본](https://huggingface.co/aadel4/omniASR-CTC-300M), [공식 Omnilingual ASR 문서](https://github.com/facebookresearch/omnilingual-asr) — 확신도: 중

**판정:** 상업 라이선스와 VRAM은 매력적이지만, 현재의 한국어 자모·일본어 문자 forced alignment 경로에 바로 넣을 수 있는지는 확인되지 않았습니다. 실험용 다국어 후보로만 권장합니다.

### 4.4 기타 후보

- `GigaAM-Multilingual`: MIT 라이선스이지만 공개 모델 카드의 지원 언어가 러시아어·카자흐어·키르기즈어·우즈베크어·영어 중심이라 한국어·일본어 후보로 부적합합니다. [GigaAM 모델 카드](https://huggingface.co/ai-sage/GigaAM-Multilingual) — 확신도: 상
- VoxPopuli 계열: CC-BY-NC-4.0으로 표시되어 상업 후보에서 제외해야 합니다. [VoxPopuli 저장소](https://github.com/facebookresearch/voxpopuli) — 확신도: 상
- `sadda-speech/wav2vec2-espeak-ctc`: Apache-2.0 기반의 IPA/phoneme CTC 후보이지만, 한국어·일본어 G2P와 phoneme mapping이 필요하여 현재 문자 가사 경로의 직접 후보는 아닙니다. [모델 카드](https://huggingface.co/sadda-speech/wav2vec2-espeak-ctc) — 확신도: 중

---

## 5. 모델 크기와 forced alignment timing 회귀

### 5.1 문헌에서 확인된 사실

노래 음성은 일반 음성보다 alignment가 어렵다는 실측 사례가 있습니다.

- CTC 기반 연구에서 speech 평균 오차는 22.6ms, singing 평균 오차는 29.8ms였습니다.
- 해당 연구의 모델은 4.5M 파라미터로, 비교 기준 48M 모델보다 약 10배 작았습니다.
- 그러나 이것은 “1B Wav2Vec2를 95M으로 줄였을 때 동일 품질”을 입증한 비교가 아닙니다.
- 연구 모델은 alignment 전용으로 설계되고 추가 제약 손실을 사용했습니다.

[Interspeech 2022 연구](https://www.isca-archive.org/interspeech_2022/teytaut22_interspeech.html), [논문 PDF](https://www.isca-archive.org/interspeech_2022/teytaut22_interspeech.pdf) — 확신도: 상

장시간 노래 alignment 연구에서는 speech 평균 오차 50ms, singing 평균 오차 120ms, singing median 오차 50ms 미만이 보고되었습니다. [Doras, Teytaut, Roebel 2023](https://doi.org/10.3390/app13031854) — 확신도: 상

Wav2Vec2를 singing domain에 transfer learning한 연구에서는 speech 사전학습과 singing fine-tuning이 중요했고, CTC 모델의 DSing 테스트 WER은 20.99였습니다. [Ou et al. 2022](https://arxiv.org/abs/2207.09747), [ISMIR 논문 PDF](https://archives.ismir.net/ismir2022/paper/000107.pdf) — 확신도: 상

### 5.2 확인하지 못한 내용

다음 직접 비교 자료는 확인하지 못했습니다.

- MMS-1B vs MMS-300M의 동일 singing set forced alignment timing 비교
- 300M vs 95M Wav2Vec2의 동일 singing set 비교
- 보카로·Synthesizer V·VOCALOID·SVS 데이터에서 한국어 CTC 후보 비교
- 한국어 또는 일본어 합성보컬에서 posterior collapse율 비교
- `Kkonjeong`, Reazon HuBERT, Omnilingual의 singing forced alignment benchmark

따라서 “95M으로 줄여도 timing 회귀가 없다”고 주장할 근거는 현재 없습니다.

### 5.3 모델 크기만으로 품질을 예측하기 어려운 이유

강제정렬 품질은 파라미터 수보다 다음 요소의 영향을 함께 받습니다.

- 노래 음성으로 학습 또는 fine-tuning되었는가
- source separation 이후의 음질
- pitch와 phoneme 지속시간 변화
- emission frame stride
- target vocab의 자모·음절·subword 단위
- blank posterior의 peakiness
- transcript normalization
- 긴 오디오에서의 누적 drift

따라서 1B→300M→95M의 단순 파라미터 비교만으로 회귀 여부를 판정할 수 없습니다. 이는 위 singing alignment 연구들이 모델 구조·학습 도메인·CTC 제약을 함께 조정한 사실에 근거한 설계 해석입니다. — 확신도: 중

---

## 6. 최종 추천, VRAM·속도 예상, A/B 프로토콜

### 6.1 추천 모델

#### 한국어

**추천 1순위: `Kkonjeong/wav2vec2-base-korean`**

- 94.4M
- Apache-2.0
- 자모 기반 CTC
- 기존 한국어 발음 변환 경로와 호환 가능성이 가장 높음

단, 합성보컬에서의 품질 비회귀는 검증 전까지 보장할 수 없습니다. [Kkonjeong 모델 카드](https://huggingface.co/Kkonjeong/wav2vec2-base-korean) — 확신도: 상

**비교군: `kresnik/wav2vec2-large-xlsr-korean`**

- 공개 음성 CER이 우수
- 300M급
- 음절 vocab 가능성이 높아 자모 경로와 별도 normalization 필요

[kresnik 모델 카드](https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean) — 확신도: 상~중

#### 일본어

**히라가나 정규화 가능 시: `reazon-research/japanese-hubert-base-k2-rs35kh`**

- 98.4M
- Apache-2.0
- 공개 음성 CER이 강함
- VRAM 절감 가능성이 큼

[Reazon HuBERT 모델 카드](https://huggingface.co/reazon-research/japanese-hubert-base-k2-rs35kh) — 확신도: 상

**직접 한자·가나 정렬 유지 시:**

1. 기존 `jonatasgrosman/wav2vec2-large-xlsr-53-japanese`
2. `reazon-research/japanese-wav2vec2-large-rs35kh`

[Jonatas 모델 카드](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-japanese), [Reazon Large 모델 카드](https://huggingface.co/reazon-research/japanese-wav2vec2-large-rs35kh) — 확신도: 상

### 6.2 VRAM과 속도 예상

가중치 크기만 단순 계산하면 다음과 같습니다.

| 모델 규모 | FP32 가중치 대략 | 1B 대비 감소 |
|---:|---:|---:|
| 1B | 약 3.7 GiB | 기준 |
| 300M | 약 1.1 GiB | 파라미터 약 70% 감소 |
| 95M | 약 0.35 GiB | 파라미터 약 90.5% 감소 |

이는 파라미터 수와 4-byte FP32를 이용한 계산값이며, 실제 runtime peak VRAM이 아닙니다. MMS 카드의 1B/F32 표기와 후보 모델의 파라미터 수에 근거한 산술 추정입니다. [MMS 모델 카드](https://huggingface.co/facebook/mms-1b-all), [Kkonjeong 모델 카드](https://huggingface.co/Kkonjeong/wav2vec2-base-korean), [Omnilingual 공식 사양](https://github.com/facebookresearch/omnilingual-asr) — 확신도: 상

실제 inference VRAM은 emission 저장, 긴 오디오 chunk, attention activation, CTC DP 때문에 달라집니다. 공식 Omnilingual 측정에서는 300M CTC가 약 2GiB, 1B CTC가 약 3GiB였지만, 이는 A100·BF16·30초·batch 1 조건입니다. RTX 3090 및 everyric2의 긴 곡 처리에 그대로 적용할 수 없습니다. [Omnilingual ASR 공식 저장소](https://github.com/facebookresearch/omnilingual-asr) — 확신도: 상

**속도 절감폭의 정확한 수치:** 확인하지 못했습니다.

1B에서 300M 또는 95M으로 줄이면 계산량 감소 가능성은 높지만, 모델 구조·frame stride·추론 backend·오디오 chunk 길이에 따라 실제 RTF가 달라집니다. 따라서 다음을 직접 측정해야 합니다.

- warm-up 이후 wall-clock time
- RTF
- 초당 처리 audio seconds
- peak allocated VRAM
- peak reserved VRAM
- 긴 곡에서의 emission 생성 시간
- forced-alignment DP 시간

### 6.3 A/B 검증 프로토콜

#### A. 평가 세트 구성

곡 단위로 train/dev/test를 분리하고, 같은 곡의 라인이 여러 split에 섞이지 않게 해야 합니다.

권장 층화:

- 일반 사람 보컬
- 보카로·합성보컬·SVS
- 반주가 큰 곡
- source separation 품질이 낮은 곡
- 빠른 랩·장음·짧은 자음
- 한국어·일본어·발음 변환 외국어
- 현재 MMS posterior가 붕괴한 곡
- 긴 곡과 짧은 곡

평가 설계 자체는 제안 사항입니다.

#### B. 기준 정답

baseline MMS 결과를 정답으로 사용하면 안 됩니다.

권장 방식:

- vocal stem 또는 가능한 한 깨끗한 보컬 사용
- 사람 검수자가 라인 시작·끝을 주석
- 일부 표본은 두 명 이상이 독립 주석
- 주석자 간 오차를 측정하여 달성 가능한 정확도의 하한으로 사용

#### C. 고정해야 할 조건

후보 모델 간 다음을 동일하게 유지해야 합니다.

- 동일한 오디오 파일
- 동일한 sampling rate와 mono 변환
- 동일한 vocal separation 결과
- 동일한 가사 원문
- 동일한 deterministic pronunciation conversion
- 동일한 punctuation·공백·Unicode normalization
- 동일한 chunking과 overlap
- dev set에서만 threshold·보정 파라미터 튜닝

일본어의 경우 “한자 직접 정렬”과 “히라가나 변환 후 정렬”을 별도 실험군으로 분리해야 합니다.

#### D. timing 지표

각 라인 및 가능하면 단어·자모 단위로 다음을 계산합니다.

- onset MAE
- offset MAE
- median absolute error
- P95, P99 error
- 50ms·100ms·200ms·300ms 이내 비율
- 300ms 초과 비율
- 곡 후반으로 갈수록 오차가 증가하는 drift slope

singing alignment 연구에서 MAE, median error, 장시간 오차를 사용하므로 이 지표 구성이 적절합니다. [Doras et al.](https://doi.org/10.3390/app13031854), [Teytaut et al.](https://www.isca-archive.org/interspeech_2022/teytaut22_interspeech.html) — 확신도: 상

#### E. 붕괴 지표

사용자께서 이미 관찰하신 CTC posterior 바닥 현상을 별도 hard gate로 둬야 합니다.

- forced path 생성 실패율
- 곡 단위 alignment 실패율
- 최소 token posterior
- 평균 target-token log probability
- blank posterior 비율
- target token coverage
- path가 지나치게 한 구간에 몰리는 비율
- 후보와 baseline의 라인 누락률
- posterior collapse 곡의 복구 여부

#### F. 통계적 비회귀 판정

곡 단위 paired 비교를 사용합니다.

- baseline과 후보를 같은 곡에서 비교
- line 단위 수치를 곡 단위로 집계
- bootstrap 95% 신뢰구간
- paired permutation 또는 Wilcoxon signed-rank test
- 합성보컬 subset을 별도 보고
- worst 5% 곡을 별도 보고

권장 승격 조건은 다음과 같습니다.

1. 전체 세트에서 후보의 MAE·P95가 baseline보다 악화되지 않을 것
2. 합성보컬 subset에서 후보의 collapse rate가 baseline보다 높지 않을 것
3. 300ms 초과 라인 비율이 증가하지 않을 것
4. 현재 붕괴 곡에서 후보가 최소한 baseline 수준의 coverage를 보일 것
5. VRAM peak가 9GB 예산 안에 들어올 것
6. 속도 개선은 timing 품질을 희생하지 않는 범위에서만 인정할 것

예를 들어 “MAE +20ms, P95 +50ms까지 허용” 같은 수치는 프로젝트가 주석자 간 오차를 확인한 후 정해야 하며, 문헌에서 정해진 보편 기준은 아닙니다. 위 수치는 제안값입니다.

### 6.4 운영 전략

최초 배포에서는 다음 구조를 권장합니다.

```text
후보 모델 정렬
→ posterior/coverage 품질 검사
→ 통과: 후보 결과 사용
→ 실패 또는 collapse: MMS fallback
```

다만 상업 라이선스 제거가 최우선이면 MMS fallback은 법무적으로 여전히 문제가 있으므로, 최종 상용 배포 전에는 상업적으로 허용되는 후보 또는 자체 fine-tuned 모델로 fallback까지 대체해야 합니다.

## 최종 의사결정

현재 가장 합리적인 실험 순서는 다음입니다.

1. 한국어: `Kkonjeong/wav2vec2-base-korean`
2. 일본어 히라가나 경로: `reazon-research/japanese-hubert-base-k2-rs35kh`
3. 일본어 직접 문자 경로: `reazon-research/japanese-wav2vec2-large-rs35kh`
4. 다국어 실험군: `facebook/omniASR-CTC-300M`
5. 기존 MMS와 동일 곡·동일 preprocessing으로 A/B
6. 합성보컬 collapse rate를 hard gate로 적용
7. 비회귀를 확인하기 전에는 production 기본 모델을 변경하지 않음

현재 조사만으로 “품질 회귀 없음”을 보증할 수 있는 후보는 없습니다. 다만 라이선스·VRAM·vocab 구조까지 고려하면 한국어는 Kkonjeong, 일본어는 Reazon HuBERT Base가 가장 먼저 검증할 가치가 있는 조합입니다.