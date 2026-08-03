# 과제 D — 반주(instrumental) 분리 모델 조사 (codex gpt-5.6-luna, 2026-07-30)

> 용도: ① 커버↔원곡 링크 확정용 반주 상관 ② 향후 MR/카라오케. 정책: 명백한 NC만 제외, 미확인 라이선스도 실측 포함.
> 핵심 요지: Kim FT 여집합(mixture−vocals)은 SDR 17.32·Bleedless 46.72로 전용 inst 모델보다 "깨끗"하지만 Fullness(악기 보존)가 낮다 — SDR/Fullness/Bleedless를 구분하지 않으면 "전용 모델이 항상 우월" 통념은 부정확. 상관용은 단일 2-stem 1회가 기본, MR은 필요 시 전용 Bleedless 모델 추가 2단계.

## 1. 요약

- 현재 공개된 MVSep 기준 instrumental SDR 상위는 **BS-RoFormer 124-band 18.64 dB**, **BS PolarFormer 124-band 18.33 dB**, **BS-RoFormer 2025.07 18.20 dB**입니다. ([MVSep 리더보드](https://mirror.mvsep.com/quality_checker/multisong_leaderboard?sort=instrum), 확신도: 높음)
- Kim FT MelBand의 여집합은 SDR·Bleedless가 높지만 Fullness가 낮습니다. 전용 instrumental 모델은 악기 보존량이 더 많지만 보컬 잔류가 증가하는 경우가 많습니다. ([MVSep 품질표](https://www.mvsep.com/en/algorithms), 확신도: 높음)
- 링크 확정용 상관에는 단일 2-stem 모델이 충분하며, MR용은 SDR보다 **instrumental Bleedless와 청취 품질**을 우선해야 합니다.
- 기본 실측 우선순위는 `BS-RoFormer viperx/2025.07 → Kim FT → BS-RoFormer Inst FNO/HyperACE → Gabox Bleedless → SCNet XL IHF → MDX23C → htdemucs_ft` 순서가 적절합니다.
- 확인한 후보 중 명백한 NC 라이선스는 발견하지 못했습니다. 다만 커뮤니티 파인튜닝 모델 대부분은 체크포인트 라이선스가 명시되지 않아 "미확인"으로 분류해야 합니다.

## 2. Instrumental SDR 모델 비교표

SDR은 주로 MVSep의 Multisong 기준입니다. 데이터셋과 평가 프로토콜이 다르면 수치를 직접 비교하면 안 됩니다. 라이선스는 코드 라이선스가 아니라 가능한 한 **체크포인트 배포 라이선스** 기준입니다.

| 모델명 | 아키텍처 | instrumental SDR | 출처 | 다운로드 경로 | 라이선스 상태 | 확신도 |
|---|---|---:|---|---|---|---|
| BS-RoFormer 124-band 2026.07 | BS-RoFormer, 124 bands | **18.64 dB** | [MVSep 알고리즘표](https://www.mvsep.com/en/algorithms) | MVSep API/서비스. 로컬 체크포인트 경로는 확인 못 함 | 미확인 | 높음 |
| BS-RoFormer 124-band fullness duality | BS-RoFormer | **18.47 dB** | [MVSep 알고리즘표](https://www.mvsep.com/en/algorithms) | MVSep 서비스. 로컬 경로 확인 못 함 | 미확인 | 높음 |
| BS PolarFormer 124-band | BS-RoFormer 계열 + PoPE | **18.33 dB** | [MVSep 리더보드](https://mirror.mvsep.com/quality_checker/multisong_leaderboard?sort=instrum) | [ZFTurbo release](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.20) | 상업확인* | 중간 |
| BS-RoFormer 2025.07 | BS-RoFormer | **18.20 dB** | [MVSep 알고리즘표](https://www.mvsep.com/en/algorithms) | [audio-separator 모델 목록](https://pypi.org/project/audio-separator/0.44.1/), `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | 미확인 | 높음 |
| MVSep Ensemble 2025.06 | BS-RoFormer ×2 + MelBand + SCNet XL IHF | **18.23 dB** | [MVSep 알고리즘표](https://www.mvsep.com/en/algorithms) | MVSep 서비스/API | 미확인 | 높음 |
| BS-RoFormer v1.04 | BS-RoFormer | **17.55 dB** | [MVSep 알고리즘표](https://www.mvsep.com/en/algorithms) | ZFTurbo/MVSep 모델 저장소 계열 | 미확인 | 높음 |
| BS-RoFormer viperx | BS-RoFormer | **17.17 dB** | [MVSep 알고리즘표](https://www.mvsep.com/en/algorithms) | [audio-separator](https://pypi.org/project/audio-separator/0.44.1/) | 미확인 | 높음 |
| BS-RoFormer HyperACE v2 instrumental | BS-RoFormer + HyperACE head | **17.40 dB** | [MVSep](https://www.mvsep.com/algorithms/34?lang=en), [모델 카드](https://huggingface.co/pcunwa/BS-Roformer-HyperACE) | Hugging Face `pcunwa/BS-Roformer-HyperACE` | 미확인 | 높음 |
| BS-RoFormer Inst FNO | BS-RoFormer + FNO1d mask head | **17.60 dB** | [모델 카드](https://huggingface.co/pcunwa/BS-Roformer-Inst-FNO) | Hugging Face `pcunwa/BS-Roformer-Inst-FNO` | 미확인 | 높음 |
| BS-RoFormer SW | 6-stem BS-RoFormer | **17.50 dB** | [MVSep 알고리즘표](https://www.mvsep.com/en/algorithms) | [openmirlab registry](https://github.com/openmirlab/bs-roformer-infer) | 미확인 | 높음 |
| SCNet XL IHF | SCNet XL | **17.41 dB** | [MVSep SCNet 표](https://www.mvsep.com/en/algorithms) | [ZFTurbo releases](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases) | 미확인 | 높음 |
| SCNet XL | SCNet XL | **17.27 dB** | [MVSep SCNet 표](https://www.mvsep.com/en/algorithms) | [ZFTurbo releases](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases) | 미확인 | 높음 |
| MelBand RoFormer 2024.10 | MelBand RoFormer | **17.59 dB** | [MVSep MelBand 표](https://www.mvsep.com/en/algorithms) | [GaboxR67 모델 저장소](https://huggingface.co/GaboxR67/MelBandRoformers), UVR resources | 미확인 | 높음 |
| MelBand RoFormer becruily deux | MelBand RoFormer | **17.51 dB** | [MVSep MelBand 표](https://www.mvsep.com/en/algorithms) | 커뮤니티/UVR resources | 미확인 | 높음 |
| MelBand RoFormer Bas Curtiz | MelBand RoFormer | **17.49 dB** | [MVSep MelBand 표](https://www.mvsep.com/en/algorithms) | 커뮤니티/UVR resources | 미확인 | 높음 |
| MelBand RoFormer Kim FT | MelBand RoFormer | **17.32 dB** | [MVSep MelBand 표](https://www.mvsep.com/en/algorithms) | [KimberleyJSN/melbandroformer](https://huggingface.co/KimberleyJSN/melbandroformer) | **상업확인: MIT** | 높음 |
| Gabox Instrumental v10 | MelBand RoFormer | **16.97 dB** | [MVSep MelBand 표](https://www.mvsep.com/en/algorithms) | [GaboxR67 모델 저장소](https://huggingface.co/GaboxR67/MelBandRoformers) | 미확인 | 높음 |
| Unwa Instrumental v1e+ | MelBand RoFormer | **16.64 dB** | [MVSep MelBand 표](https://www.mvsep.com/en/algorithms) | UVR resources / 커뮤니티 저장소 | 미확인 | 높음 |
| Gabox Instrumental v7 | MelBand RoFormer | **16.63 dB** | [MVSep MelBand 표](https://www.mvsep.com/en/algorithms) | [GaboxR67 모델 저장소](https://huggingface.co/GaboxR67/MelBandRoformers) | 미확인 | 높음 |
| Unwa Instrumental v1e | MelBand RoFormer | **16.36 dB** | [MVSep MelBand 표](https://www.mvsep.com/en/algorithms) | UVR resources | 미확인 | 높음 |
| MDX23C 8K FFT v2 | MDX23C/TFC-TDF | **16.66 dB** | [MVSep MDX23C 표](https://www.mvsep.com/en/algorithms) | [ZFTurbo MSS 저장소](https://github.com/ZFTurbo/Music-Source-Separation-Training) | 미확인 | 높음 |
| MDX23C InstVoc HQ | MDX23C | **약 16.46 dB** | [MVSep quality checker](https://mvsep.com/quality_checker/queue?page=54) | [UVR/audio-separator](https://pypi.org/project/audio-separator/0.44.1/) | 미확인 | 중간 |
| htdemucs_ft | Hybrid Demucs | **14.63 dB** | [MVSep Demucs 표](https://www.mvsep.com/en/algorithms) | [facebookresearch/demucs](https://github.com/facebookresearch/demucs) | 상업확인* | 높음 |
| MVSEP MDX23 ensemble | MDX23C + Demucs/MDX 계열 | **15.82~16.60 dB** | [ZFTurbo MDX23 저장소](https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model) | GitHub 저장소 | 미확인 | 높음 |

\* `상업확인`은 저장소 또는 배포 아티팩트에 MIT가 명시된 경우입니다. 학습 데이터의 저작권·상업 이용권까지 확인했다는 뜻은 아닙니다.

2024~2026년 공개된 모든 커뮤니티 파인튜닝 체크포인트를 문자 그대로 전수 열거하는 중앙 목록은 존재하지 않습니다. 위 표는 MVSep 공식 비교표와 공개 다운로드 registry에서 SDR이 확인되는 주요 2-stem·instrumental 특화 모델을 망라한 것입니다.

## 3. 여집합 vs 전용 instrumental 모델 품질 비교

MVSep은 instrumental 품질을 SDR만이 아니라 **Fullness**와 **Bleedless**로 분리합니다.

| 출력 방식 | Fullness | Bleedless | SDR | 해석 |
|---|---:|---:|---:|---|
| Kim FT MelBand `mixture − vocals` | 27.71 | **46.72** | **17.32 dB** | 보컬 잔류 적음, 악기 일부 손실 |
| Unwa Instrumental v1e | **38.85** | 35.68 | 16.36 dB | 악기 보존량 높음, 아티팩트 증가 |
| Unwa Instrumental v1e+ | 36.20 | 38.57 | 16.64 dB | 절충 |
| Gabox Instrumental v7 | 29.34 | 45.06 | 16.63 dB | Kim FT와 유사하게 clean |
| becruily instrumental 계열 | 33.41 | 42.11 | 17.51 dB | 절충형 |

결론: Kim FT 여집합은 SDR·Bleedless가 전용 instrumental 모델보다 오히려 높지만 Fullness는 낮습니다. "전용 모델이 항상 우월하다"는 통념은 SDR/Fullness/Bleedless를 구분하지 않으면 부정확합니다.

## 4. 2-stem 운용 전략 권장

- **상관용:** 단일 모델(BS-RoFormer/MDX23C 계열)로 vocals·instrumental 동시 추출. 시간축·위상 관계 공유가 중요.
- **MR용:** vocals=Kim FT, instrumental=전용 Bleedless/Fullness 모델을 별도 실행 가능. 단, inference 시간 ~2배, `vocals+instrumental≈mixture` 재구성 보장 상실, 3090 24GB에서 두 RoFormer 동시 상주 시 OOM 위험.
- 권장 패턴: 1) 링크용은 단일 모델 1회 실행이 기본값, 2) MR 요청이 있을 때만 전용 instrumental 모델을 추가 실행하는 2단계 구조.

## 5. 용도별 최소 품질 기준

**상관용:** 문헌상 보편적 SDR 임계값은 확인 못 함. 실무 제안(문헌 기준 아님, 확신도 중간): `instrumental SDR ≥ 14 dB`(현재 htdemucs_ft 14.63 dB)를 1차 필터로 삼고, 최종 채택은 SDR이 아니라 실측 positive/negative correlation AUC와 false-link rate로 결정. beat/onset envelope·chroma correlation을 waveform correlation과 병행 권장.

**MR용(bleedless):**
- Gabox `Instrumental Bleedless V1/V2/V3` 계열 존재 확인. V1 참고치 Fullness 35.03/Bleedless 39.10/SDR 16.49 dB이나 출처가 커뮤니티 가이드라 확신도 낮음.
- BS-RoFormer 2025.07: Bleedless 49.12, Fullness 27.82 (확신도 높음)
- BS-RoFormer 124-band 2026.07: Bleedless 49.85, Fullness 29.12, SDR 18.64 dB (확신도 높음)
- HyperACE v2 instrumental: SDR 17.40, Bleedless 37.87, Fullness 38.03 (확신도 높음)
- MR용 1순위(현재 공개 수치 기준): BS-RoFormer 2025.07 또는 124-band 표준 출력.

## 6. 최종 후보 목록 + 우선순위 + 예상 VRAM

VRAM은 공식 통일 벤치마크 부재로 추정치(확신도 중간)입니다.

| 우선순위 | 후보 | 용도 | 예상 VRAM | 라이선스 |
|---|---|---|---:|---|
| P0 | BS-RoFormer 2025.07/viperx | 상관+MR baseline | 4~8GB | 미확인 |
| P0 | Kim FT MelBand 여집합 | 상관+Clean MR baseline | 4~8GB | MIT |
| P0 | htdemucs_ft | 기존 baseline | 4~8GB | 상업확인* |
| P0 | MDX23C InstVoc HQ | 상관+저비용 비교 | 2~5GB | 미확인 |
| P1 | BS-RoFormer Inst FNO | Full MR | 6~10GB | 미확인 |
| P1 | BS-RoFormer HyperACE v2 instrumental | Full MR | 6~10GB | 미확인 |
| P1 | Gabox Instrumental Bleedless V1/V2 | Clean MR | 4~8GB | 미확인 |
| P1 | SCNet XL IHF | 상관+MR 앙상블 | 4~8GB | 미확인 |
| P1 | BS PolarFormer 124-band | 상관+MR | 6~10GB | 상업확인* |
| P1 | BS-RoFormer 124-band 2026.07 | MR 최상위 후보 | 6~12GB | 미확인 |
| P2 | MelBand becruily deux | MR 절충 | 4~8GB | 미확인 |
| P2 | Gabox Instrumental v7/v10 | MR 비교 | 4~8GB | 미확인 |
| P2 | Unwa Instrumental v1e/v1e+ | Full MR | 4~8GB | 미확인 |
| P2 | BS-RoFormer SW (6-stem) | 향후 악기별 기능 | 8~14GB | 미확인 |
| P2 | MVSep Ensemble | 최고 품질 비교용 | 10~20GB+ | 미확인 |
| P2 | MVSep Mega 53-stem | 향후 악기별 MR | 16GB 이상 | 미확인 |

실측 순서 권장: `htdemucs_ft → Kim FT 여집합 → BS-RoFormer viperx/2025.07 → MDX23C InstVoc HQ → BS-RoFormer Inst FNO → HyperACE v2 → Gabox Bleedless V1 → SCNet XL IHF → BS PolarFormer 124-band → BS-RoFormer 124-band 2026.07 → Unwa v1e+ → becruily deux → BS-RoFormer SW`.
