# 과제 A — 보컬 분리 모델 교체 타당성 (codex gpt-5.6-luna, 2026-07-30)

> 3갈래 모델 교체 조사 중 A 트랙 보고 원문. 선행: [2026-07-26 통합 조사](../2026-07-26-alignment-papers-patents-survey.md) §4

# Everyric2 보컬 분리 모델 교체 타당성 조사

조사 기준일: **2026-07-30**  
판정 원칙: 코드 라이선스와 체크포인트 라이선스를 별도로 확인하며, 체크포인트 라이선스가 명시되지 않은 경우 상업 사용 가능으로 간주하지 않았습니다.

## 1. htdemucs 가중치의 상업 이용 가능 여부

### 결론

**htdemucs 코드의 MIT 라이선스는 확인되지만, 공식 pretrained weight의 상업 이용 가능 여부는 확인되지 않았습니다.**

| 항목 | 확인 결과 |
|---|---|
| Demucs 코드 | MIT. 상업 사용·수정·배포 가능 범위의 문구가 포함되어 있습니다. [Demucs LICENSE](https://raw.githubusercontent.com/facebookresearch/demucs/main/LICENSE) |
| htdemucs 학습 데이터 | MUSDB18-HQ와 추가 800곡으로 명시되어 있습니다. [Demucs README](https://raw.githubusercontent.com/facebookresearch/demucs/main/README.md) |
| MUSDB18 권리 | 일부 트랙은 CC BY-NC-SA이며, 전체 데이터셋은 학술 목적 사용으로 안내됩니다. [MUSDB18 권리 안내](https://sigsep.github.io/datasets/musdb.html) |
| 공식 weight 라이선스 | 공식 저장소에서 별도 상업 라이선스를 확인하지 못했습니다. |
| 공식 답변 | pretrained model의 상업 배포 라이선스를 묻는 이슈가 2022년부터 열려 있으나 답변 없이 남아 있습니다. 저장소도 2025-01-01 archive 처리되었습니다. [Issue #327](https://github.com/facebookresearch/demucs/issues/327) |

**최신 보수적 결론:** 상업 서비스에 공식 htdemucs weight를 포함하는 것은 승인받지 않은 상태로 보아야 합니다. 코드 MIT만으로 가중치까지 MIT라고 해석하면 안 됩니다.

확신도: **높음** — “상업 이용 가능하다고 공식 확인되지 않았다”는 판단.  
법적 최종 판단은 Meta 및 데이터 권리자에게 서면 확인을 받으셔야 합니다.

---

## 2. BS-RoFormer / Mel-Band RoFormer 공개 체크포인트 라이선스

### 공통 원칙

ZFTurbo 저장소의 코드는 MIT이지만, 저장소가 외부 호스트에 있는 weight까지 MIT로 부여하지는 않습니다. [ZFTurbo LICENSE](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/LICENSE)

### ZFTurbo 공개 목록

아래는 ZFTurbo가 현재 공개 목록과 Mel-RoFormer 실험 문서에 색인한 BS/Mel 계열입니다. 체크포인트 목록 자체는 [pretrained_models.md](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md)와 [Mel-RoFormer 실험 목록](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/mel_roformer_experiments.md)을 기준으로 확인했습니다.

| 체크포인트 | weight 라이선스 | 상업 사용 판정 |
|---|---|---|
| BS-RoFormer viperx vocals, `ep_317` | TRvlvr release에 별도 라이선스 확인 못 함. [model_repo](https://github.com/TRvlvr/model_repo) | **확인 못 함 — 사용 보류** |
| MelBand RoFormer viperx vocals, `ep_3005` | 별도 라이선스 확인 못 함 | **확인 못 함 — 사용 보류** |
| BS-RoFormer viperx other, `ep_937` | 별도 라이선스 확인 못 함 | **확인 못 함 — 사용 보류** |
| MelBand crowd, aufr33+viperx | 별도 라이선스 확인 못 함 | **확인 못 함 — 사용 보류** |
| MelBand dereverb, anvuew | 현재 공개 페이지에서 상업 라이선스 확인 못 함 | **확인 못 함** |
| MelBand denoise, aufr33 | 별도 라이선스 확인 못 함 | **확인 못 함** |
| MelBand denoise aggressive, aufr33 | 별도 라이선스 확인 못 함 | **확인 못 함** |
| MelBand aspiration, SUC-DriverOld | 별도 라이선스 확인 못 함 | **확인 못 함** |
| MelBand dereverb/deecho, SUC-DriverOld | 별도 라이선스 확인 못 함 | **확인 못 함** |
| BS PolarFormer | ZFTurbo 목록에는 있으나 weight 라이선스는 별도 확인 못 함 | **확인 못 함** |
| Mel-RoFormer 실험 `ep_53`, `ep_38`, `ep_166`, `ep_168`, `ep_15`, `ep_9`, `ep_7`, `ep_1`, `ep_5` | 모두 checkpoint별 라이선스 확인 못 함 | **사용 보류** |

ZFTurbo의 MIT는 inference/training 코드에 적용되는 것으로 보이며, viperx가 배포된 [TRvlvr model_repo](https://github.com/TRvlvr/model_repo)에는 체크포인트별 라이선스가 명시되어 있지 않습니다. 따라서 **viperx는 기술적으로 유망하지만 상업 배포용으로는 법적 증빙이 부족합니다.**

### Kimberley Jensen / Kim FT MelBand

현재 가장 유력한 조합입니다.

- Hugging Face 모델 페이지에 현재 `License: mit`가 표시됩니다. [KimberleyJSN/melbandroformer](https://huggingface.co/KimberleyJSN/melbandroformer)
- 최초에는 GPL-3.0으로 표시되었으나, 2026-04-22 커밋에서 MIT로 변경되었습니다. [라이선스 변경 커밋](https://huggingface.co/KimberleyJSN/melbandroformer/commit/ac9b0614ab3cd7f77219e18ba494dfd93956c348)
- inference/config 코드는 ZFTurbo MIT 또는 [lucidrains BS-RoFormer MIT](https://raw.githubusercontent.com/lucidrains/BS-RoFormer/main/LICENSE)를 사용할 수 있습니다.
- Kimberley Jensen의 별도 GitHub 저장소에는 명확한 LICENSE 파일을 확인하지 못했으므로, 배포 시에는 ZFTurbo/lucidrains 코드와 Hugging Face 체크포인트를 직접 고정하는 편이 안전합니다. [Kimberley 저장소](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model)

**코드+가중치 모두 상업 이용 가능으로 확인되는 조합**

> ZFTurbo MIT inference/config 코드 + lucidrains MIT architecture + KimberleyJSN Hugging Face checkpoint의 현재 MIT metadata

다만 학습 데이터와 모든 제3자 권리까지 완전히 정리되었다는 의미는 아닙니다. 모델 파일, commit hash, SHA-256, 현재 LICENSE 문구를 보관하고 법무 검토를 권고드립니다.

확신도: **중상** — 현재 체크포인트 MIT 표기는 확인되지만, 데이터 provenance까지 완전하게 확인된 것은 아닙니다.

### MVSep 공개분

MVSep API에는 다음과 같은 BS/Mel 계열 모델이 공개되어 있습니다.

- BS-RoFormer viperx
- BS-RoFormer 2024.02/04/08/2025.07
- MelBand RoFormer
- MelBand RoFormer XL
- MVSep Team BS-RoFormer
- MVSep Team 2026.07 모델

[MVSep Full API 모델 목록](https://www.mvsep.com/en/full_api)

그러나 이는 온라인 서비스에서 선택 가능한 모델명·성능 정보이지, 상업 재배포 가능한 checkpoint와 라이선스 문서가 아닙니다. 공개 API 문서에서 체크포인트 다운로드 권리나 weight 라이선스를 확인하지 못했습니다. [MVSep API에서 license/weights 항목 확인 결과](https://www.mvsep.com/en/full_api)

**판정:** MVSep 공개 모델은 Everyric2에 weight를 내려받아 포함할 후보가 아닙니다.

확신도: **높음** — 공개 서비스와 상업 재배포 권리는 별개라는 판단.

---

## 3. Mel-RoFormer 논문과 RMVPE 대체 가능성

논문: [Mel-RoFormer for Vocal Separation and Vocal Melody Transcription](https://arxiv.org/html/2409.04702)

### 공개 코드와 체크포인트

확인되는 공개 산출물은 다음과 같습니다.

- BS/Mel-RoFormer 구현: [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)
- 학습 설정과 커뮤니티 체크포인트: [ZFTurbo Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
- 논문이 직접 연결하는 것은 구현과 configuration이며, **논문의 melody-transcription 전용 fine-tuned checkpoint는 공식 다운로드 링크로 확인하지 못했습니다.**

따라서 현재 공개된 Kim/viperx weight는 주로 **보컬 분리용**으로 보아야 하며, 논문에서 사용한 melody transcription head와 동일한 모델이라고 가정하면 안 됩니다.

### 출력 형태

Mel-RoFormer의 분리 모델은 보컬 waveform/stem을 출력합니다.  
멜로디 전사 모델의 출력은 다음과 같습니다.

- onset
- offset
- pitch key
- 비중첩 note sequence
- monophonic melody 가정

논문은 6초 입력에 대해 50Hz frame representation을 사용하지만, 최종 출력은 연속적인 F0 곡선이 아니라 note event입니다. [논문 2장 및 5.4절](https://arxiv.org/html/2409.04702)

### RMVPE F0 → MIDI 대체 가능성

**직접 대체할 수 없습니다.**

RMVPE/FCPE 경로는:

```text
보컬 waveform → frame-level F0 → 보정/양자화 → MIDI
```

Mel-RoFormer melody head는:

```text
음원 → onset/frame predictor → note onset/offset/pitch
```

이므로 vibrato, pitch bend, 음정 흔들림, 음성 구간의 연속 F0가 사라질 수 있습니다. 또한 논문 모델은 monophonic lead melody 가정을 사용합니다.

판정:

- **보컬 분리 front-end 대체:** 가능성 높음
- **RMVPE/FCPE 대체:** 현재는 부적합
- **MIDI note 생성 전용 보조 모델:** 가능성 있음
- **기존 Everyric2의 F0 기반 경로와 완전 호환:** 불가능

확신도: **높음** — 논문이 note-level 출력과 monophonic 가정을 명시합니다.

---

## 4. RTX 3090 기준 VRAM·처리 시간

주의할 점은 **실제 Everyric2 환경에서 측정한 수치가 아닙니다.** 공개 문서의 모델 크기·chunk 설정과 간접 벤치마크를 이용한 추정입니다.

### 공개된 기준

- Demucs 공식 문서는 기본 옵션에서 약 7GB GPU memory를 안내합니다. Hybrid Transformer의 최대 segment는 7.8초입니다. [Demucs README](https://raw.githubusercontent.com/facebookresearch/demucs/main/README.md)
- Mel-RoFormer 44.1kHz stereo 모델은 8초 chunk, 50% overlap, 약 105M parameters입니다.
- 24kHz mono small 모델은 9.1M parameters, 6초 chunk입니다. [Mel-RoFormer 논문 설정 및 결과](https://arxiv.org/html/2409.04702)
- RTX 3090에서 최적화된 TensorRT Demucs는 약 5초/곡을 주장하지만, 이는 현재 PyTorch CLI와 다른 구현이며 해당 모델 카드도 CC-BY-NC입니다. [Demucs TensorRT benchmark](https://huggingface.co/MansfieldPlumbing/Demucs_v4_TRT/blob/main/README.md)

### 추정치

| 모델 | 입력/설정 | VRAM peak 추정 | 3분 곡 처리시간 추정 | 판정 |
|---|---|---:|---:|---|
| htdemucs 현재 PyTorch CLI | 44.1k stereo, 7.8초 이하, shifts=0 | 5–8GB | 15–60초 | 공식 VRAM 근거 있음, 시간은 미검증 |
| BS-RoFormer viperx | 44.1k stereo, 8초, batch=1 | 6–10GB | 40–120초 | 9GB 공유 예산에서 위험 |
| MelBand RoFormer Kim FT | 44.1k stereo, 8초, batch=1 | 5–9GB | 30–90초 | 후보로 시험할 가치 있음 |
| Mel-RoFormer 24k-small | 24k mono, 6초 | 2–4GB | 10–30초 | 저VRAM 후보, 공식 checkpoint 미확인 |
| Mel-RoFormer 24k-large | 24k mono, 6초 | 3–6GB | 15–45초 | 품질·메모리 절충안, 공식 checkpoint 미확인 |

위 표의 RoFormer VRAM과 시간은 **추정값이며 미검증**입니다. 정확한 수치는 GPU 드라이버, PyTorch allocator, flash-attention, overlap, batch size, 동시 서비스 점유량에 따라 크게 달라집니다.

RoFormer는 8초 chunk를 4초 hop으로 처리하므로, 3분 곡에서 Demucs보다 더 많은 inference window가 발생할 가능성이 있습니다. [논문 overlap 설정](https://arxiv.org/html/2409.04702)

### 권장 실측 방법

3090에서 다음 조건으로 각각 5회 이상 측정하셔야 합니다.

- 동일한 3분·5분 WAV
- 동일한 44.1kHz stereo 입력
- cold start와 warm start 분리
- batch=1
- `torch.cuda.max_memory_allocated()`
- `nvidia-smi` 50ms polling
- GPU background service가 없는 상태와 실제 prod 상태 모두 측정
- 처리시간 median/p95 기록

현재 9GB peak 예산을 고려하면 **BS-RoFormer viperx보다 Kim FT MelBand부터 측정하는 것이 합리적**입니다.

확신도: VRAM 범위 **낮음~중간**, Demucs 공식 7GB 기준 **높음**, RoFormer 시간 비교 **낮음**.

---

## 5. 2025~2026 최신 상업 라이선스 대안

### SCNet

- 공식 코드: MIT. [SCNet LICENSE](https://raw.githubusercontent.com/starrytong/SCNet/main/LICENSE)
- 공식 weight 다운로드 링크는 제공되지만, 체크포인트 자체의 별도 상업 라이선스는 확인하지 못했습니다. [SCNet README](https://github.com/starrytong/SCNet)
- 논문은 CPU inference가 HT-Demucs의 48%라고 보고합니다. [SCNet 논문](https://arxiv.org/abs/2401.13276)
- ZFTurbo에는 SCNet Small/Large/XL/IHF 등 여러 체크포인트가 색인되어 있습니다. [ZFTurbo checkpoint 목록](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md)

**판정:** 속도 후보로는 매우 좋지만, 코드+weight 모두 상업 가능하다고 확인된 상태는 아닙니다.

### TFC-TDF-UNet v3 / SDX 계열

- SDX23 공식 코드: MIT. [SDX23 LICENSE](https://raw.githubusercontent.com/kuielab/sdx23/main/LICENSE)
- TFC-TDF-UNet 계열 checkpoint와 config는 공개되어 있습니다. [SDX23 repository](https://github.com/kuielab/sdx23)
- 그러나 checkpoint별 라이선스와 학습 데이터의 상업 이용 권리는 확인하지 못했습니다.
- ZFTurbo의 MDX23C 계열도 같은 문제를 가집니다. [ZFTurbo 공개 모델 목록](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md)

**판정:** 속도는 유망하지만 상업 서비스용으로 바로 채택하기에는 weight 권리 확인이 부족합니다.

### Apollo

Apollo는 보컬 분리 모델이 아니라 **MP3 음원 복원 모델**입니다. [Apollo README](https://github.com/JusperLee/Apollo)

- 코드/프로젝트 라이선스: CC BY-SA 4.0
- 상업 사용 자체는 가능할 수 있으나 attribution/share-alike 조건이 있습니다. [Apollo LICENSE](https://raw.githubusercontent.com/JusperLee/Apollo/main/LICENSE)
- Everyric2의 보컬 분리 front-end를 대체하지 못합니다.

**판정:** 이번 교체 대상에서는 제외합니다.

### BSRNN / oBSRNN

2025~2026년에 공개된 BSRNN checkpoint도 있습니다.

- 코드: MIT로 표시됩니다. [oBSRNN repository](https://github.com/magronp/bsrnn)
- `bsrnn-large`, `bsrnn-opt`, `simo-bsrnn-opt` checkpoint가 공개되어 있습니다. [Zenodo checkpoint](https://zenodo.org/records/17516442)
- 그러나 Zenodo의 weight License 필드는 비어 있습니다.
- MUSDB18-HQ 기반이며, 데이터셋 자체도 상업 이용에 적합하지 않습니다. [MUSDB18 권리](https://sigsep.github.io/datasets/musdb.html)

**판정:** 기술 연구용 후보이지, 현재 상업 서비스용으로는 부적합합니다.

### Band-SCNet

2025년 논문 기준 2.59M parameters, 92ms latency, SDR 7.79dB를 보고합니다. [Band-SCNet 논문](https://www.isca-archive.org/interspeech_2025/yang25d_interspeech.html)

공개 코드·상업용 checkpoint를 확인하지 못했습니다.

**판정:** 확인 못 함. 연구 결과만으로는 도입 후보로 삼기 어렵습니다.

### 대안 종합

현재 공개 자료 기준으로 **코드와 가중치 모두 상업 가능하다고 확인되는 분리 모델은 사실상 Kim FT MelBand가 가장 명확합니다.** SCNet, SDX/TFC-TDF, BSRNN은 코드 라이선스는 비교적 명확하지만 weight 라이선스가 부족합니다.

---

## 6. 최종 결론 및 통합 권고

### 교체 가치

교체 가치는 있습니다.

1. htdemucs 공식 weight의 상업 이용 권리가 불명확합니다.
2. Mel-RoFormer 계열은 논문상 vocal SDR과 음질이 htdemucs보다 유망합니다. [Mel-RoFormer 결과](https://arxiv.org/html/2409.04702)
3. 다만 모든 RoFormer weight가 상업 사용 가능한 것은 아닙니다.
4. 전사·정렬 품질 회귀 금지가 최우선이므로 즉시 기본 모델을 바꾸면 안 됩니다.

### 추천 순서

1. **1순위: Kimberley Jensen Kim FT MelBand checkpoint**
   - 현재 Hugging Face metadata가 MIT입니다.
   - 44.1kHz stereo라 기존 파이프라인과 비교가 쉽습니다.
   - ZFTurbo 목록상 vocals SDR 10.98로 viperx보다 높습니다. [ZFTurbo 목록](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md)
   - 단, 법무 검토 및 checkpoint hash 고정이 필요합니다.

2. **2순위: 자체 학습한 Mel-RoFormer**
   - 상업 이용 가능한 학습 데이터만 사용합니다.
   - 코드와 weight를 직접 MIT/Apache-2.0 등으로 명확히 배포할 수 있습니다.
   - 장기적으로 가장 안전한 전략입니다.

3. **보류: viperx BS/Mel-RoFormer**
   - 품질은 좋지만 checkpoint license가 명시되지 않았습니다.
   - 9GB VRAM 예산에도 BS-RoFormer는 부담 가능성이 큽니다.

4. **실험 전용: 24k Mel-RoFormer small**
   - 저VRAM·저지연 후보입니다.
   - 다만 논문의 정확한 checkpoint가 공개되어 있지 않고, 24k mono 변환이 CTC 품질에 영향을 줄 수 있습니다.

### 통합 형태

권장 구조는 다음과 같습니다.

```text
SeparatorBackend
 ├─ DemucsSubprocessBackend       # 기존 fallback
 └─ RoFormerWorkerBackend         # 장기 실행 별도 프로세스
```

- 초기 shadow test와 rollback을 위해 기존 Demucs subprocess는 유지합니다.
- RoFormer는 매 요청마다 모델을 다시 로드하지 말고, **장기 실행 worker process**에서 한 번 로드하는 방식이 적절합니다.
- GPU 작업은 동시성 1로 제한합니다.
- worker 내부에서 `eval()`, `inference_mode()`, AMP, chunk overlap-add를 사용합니다.
- 부모 서비스 프로세스와 CUDA 메모리를 분리해 장애 격리와 rollback을 확보합니다.
- 단순한 Python in-process singleton은 VRAM 누수와 다른 서비스와의 CUDA 충돌 위험이 있어 1차 운영 형태로는 권장하지 않습니다.

### 회귀 검증 게이트

최소한 다음 데이터를 사용하셔야 합니다.

- 영어·일본어·한국어·중국어
- 남녀 보컬
- 저음·고음·랩·코러스·더블링·리버브
- 원본 음질과 압축 음질
- 3~5분 곡 중심 100~300곡

검증 지표:

- CTC/Wav2Vec2 인식률: WER/CER/PER
- 가사 라인 timestamp MAE 및 p95
- ±50ms, ±100ms 이내 라인 비율
- 누락·병합·분할된 가사 라인 수
- CTC confidence 및 alignment success rate
- RMVPE/FCPE voiced ratio
- F0 gross pitch error, octave error
- MIDI note onset/offset 차이
- stem ground truth가 있는 곡의 SDR/SIR/SAR
- ground truth가 없을 경우 vocal RMS 대비 accompaniment bleed

운영 승인 기준은 예를 들어 다음처럼 설정하실 수 있습니다.

```text
WER/CER: baseline보다 악화하지 않음
timestamp p95: baseline + 10ms 이내
alignment success rate: baseline 이상
F0 gross pitch error: baseline + 허용 오차 이내
VRAM peak: 실제 prod 여유 포함 9GB 이하
```

## 최종 판정

**htdemucs는 코드 MIT만으로 상업 서비스에 계속 사용하기에는 weight 권리 리스크가 있습니다.**  
다만 회귀 금지 조건 때문에 즉시 제거하기보다는:

> **Kim FT MelBand를 별도 worker로 도입하고, 기존 htdemucs와 shadow A/B 검증 후 교체 여부를 결정하는 방식**

을 권고드립니다.

RMVPE/FCPE F0 경로는 유지하시고, Mel-RoFormer는 우선 **보컬 분리 front-end만 대체**하는 것이 가장 안전합니다.

이번 조사는 코드 변경 없이 라이선스·논문·성능 자료만 검토한 결과입니다.