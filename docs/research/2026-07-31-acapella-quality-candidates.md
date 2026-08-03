# 보컬 분리 모델 신규 후보 조사 — 품질파 + 경량/실시간 (2026-07-31)

이 문서는 용도가 다른 **두 축**을 다룬다.

- **Part 1 (§0–§5): 품질파(성능파) 후보** — 서버측 최고 품질 아카펠라 추출용. 기존 조사
  문서에 없는 신규 후보.
- **Part 2 (§6–§10): 실시간급/경량 후보** — 클라이언트(브라우저 WebGPU/로컬 CPU)나 스트리밍
  처리에서 경량 정렬기(NeMo Conformer ONNX급)와 짝지을 후보.

방법: 순수 웹 조사(코드 실행·모델 다운로드 없음). MVSep quality checker 리더보드, ZFTurbo
Music-Source-Separation-Training 저장소, HuggingFace 저자 프로필 API, PyPI, arXiv.

---

# Part 1 — 품질파(성능파) 후보

## 0. 먼저 알아야 할 조사 한계 (지표 신뢰도)

**MVSep multisong 리더보드의 Bleedless 컬럼은 사실상 비어 있다.** 상위 20행을 컬럼 단위로
확인했지만 Bleedless 값이 채워진 행은 `BS PolarFormer 124 bands fullness level 1 stems`
(-5.8874) 한 건뿐이었다. 나머지는 전부 공란이고, `?sort=bleedless` 정렬도 데이터가 없어
동작하지 않는다. 즉 **"Bleedless를 1차 지표로 삼아 신규 후보를 줄세운다"는 계획은 공개
리더보드만으로는 불가능하다.** 아래 표의 신규 후보들은 Bleedless를 우리가 직접 실측해야
비교가 가능하다. (Fullness 역시 다수 행에서 instrumental SDR과 같은 값이 잡혀, 별도 지표로
신뢰하기 어렵다.)

따라서 아래 판정은 **multisong vocals SDR + 아키텍처 신규성 + 가중치 접근성**을 기준으로 했고,
Bleedless는 "실측 필요"로 남긴다.

---

## 1. 신규 후보 요약표

| # | 모델 | 아키텍처 | 가중치 위치 | multisong vocals SDR | Bleedless / Fullness | 가중치 라이선스 | 코드 라이선스 | VRAM(추정) | MSST/UVR | 판정 |
|---|------|----------|-------------|----------------------|----------------------|-----------------|---------------|------------|----------|------|
| A1 | **BS-Roformer Leap / Leap Xe** (pcunwa) | BS-Roformer (61밴드, dim 256 / depth 16) | [HF pcunwa/BS-Roformer-Leap](https://huggingface.co/pcunwa/BS-Roformer-Leap) | **Xe 11.7577** / base 11.7222 / uvr5 11.7398 | 미측정 / Xe 17.5303 | **미확인**(HF 라이선스 태그 없음) | MSST = MIT | 높음 (12–16GB 추정, chunk 881559≈20s) | 표준 `bs_roformer` config, 리더보드 엔트리에 uvr5 명시 | **1순위 — 시험 가치 높음** |
| A2 | **BS PolarFormer (오픈 가중치판)** | BS + PoPE 극좌표 위치 임베딩 ([arXiv 2509.10534](https://arxiv.org/abs/2509.10534)) | [ZFTurbo v1.0.20 ckpt](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.20/model_bs_polarformer_float16.ckpt) | 11.00 (저장소 보고) | 미측정 | 미확인(저장소 릴리스 자산) | **MIT** | 중 (float16, 6–10GB 추정) | MSST 네이티브 | **2순위 — 아키텍처 신규성** |
| A3 | **anvuew/BS-RoFormer** | BS-Roformer 파인튜닝 (bascurtiz 데이터셋) | [HF anvuew/BS-RoFormer](https://huggingface.co/anvuew/BS-RoFormer) | 자체보고 12.45 / ft1 12.55 (multisong 아님) | 미측정 | **GPL-3.0** (명시) | GPL-3.0 | 중 (8–12GB 추정) | MSST 호환 | **3순위 — 라이선스 명확** |
| A4 | **anvuew/BS_RoFormer_mag** | BS-Roformer, magnitude 스펙트럼 특화 | [HF anvuew/BS_RoFormer_mag](https://huggingface.co/anvuew/BS_RoFormer_mag) | 리더보드 미등재 | 미측정 | **GPL-3.0** | GPL-3.0 | 중 (추정) | MSST 호환 | 보조 후보 |
| B1 | BS-EXP-SiameseRoformer (pcunwa) | BS-Roformer + SiameseNorm ([arXiv 2602.08064](https://arxiv.org/abs/2602.08064)) | [HF](https://huggingface.co/pcunwa/BS-EXP-SiameseRoformer) | 리더보드 미등재 | — | 미확인 | 미확인 | 미확인 | 커스텀 Python 모듈 동반 | 실험판, 후순위 |
| B2 | BS-Roformer-Large-Inst (pcunwa) | MaskEstimator에 4층 TransformerBlock 추가 | [HF](https://huggingface.co/pcunwa/BS-Roformer-Large-Inst) | inst 자체보고 17.6059 | — | 미확인 | 미확인 | 높음(추정) | 커스텀 모듈 | inst 전용, 후순위 |
| B3 | Mel-Band-Roformer-big (pcunwa) | Mel-Band Roformer 대형 | [HF](https://huggingface.co/pcunwa/Mel-Band-Roformer-big) | 리더보드 미등재 | — | 미확인 | 미확인 | 높음(추정) | MSST 호환 | beta 변종 다수, 후순위 |
| B4 | BS-Roformer-Resurrection (pcunwa) | BS-Roformer | [HF](https://huggingface.co/pcunwa/BS-Roformer-Resurrection) | 리더보드 미등재 | — | 미확인 | 미확인 | 중(추정) | MSST 호환 | 커버 목록 누락분, 낮은 우선순위 |
| B5 | BS Mamba2 (ZFTurbo) | Mamba-2 SSM 밴드분할 | [ZFTurbo v1.0.19 ckpt](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.19/bs_mamba2_vocals.ckpt) | **8.82** (MUSDB18만 학습) | — | 미확인 | MIT | 낮음~중 | MSST 네이티브 (`bs_mamba2`) | **시험 가치 낮음** (SDR 미달) |
| C1 | sami-bytedance-v.1.1 | 비공개 | 없음 (API/사내) | 11.8225 | — | 접근 불가 | 접근 불가 | — | 불가 | 조달 불가 |
| C2 | BS Roformer 124 bands (2026.07.10) | MVSep 사내 | 없음 (서비스 전용) | **12.3339** (현 1위) | — / 18.6414 | 접근 불가 | 접근 불가 | — | 불가 | 조달 불가 |
| C3 | Diff-VS (Moises, ICASSP 2026) | EDM 확산 U-Net, 복소 STFT | 미공개 | 미보고 | — | 미공개 | 미공개 | — | 불가 | 논문만 |
| C4 | TS-BSmamba2 (원본) | 2단계 밴드분할 Mamba-2 | [GitHub](https://github.com/baijinglin/TS-BSmamba2) — **가중치 미공개** | MUSDB18 chunk-SDR 9.56 | — | 가중치 없음 | **Apache-2.0** | — | 코드만 | 학습 필요, 제외 |

---

## 2. 후보별 노트

### A1. BS-Roformer Leap / Leap Xe (pcunwa) — 최우선

**이번 조사의 핵심 발견.** 2026-06-30 HF 업로드로, 기존 조사 목록(Revive 2/3e, HyperACE v2,
Inst FNO)에 전혀 없다. MVSep multisong 리더보드 vocals 정렬에서 세 개 엔트리로 잡힌다:
`unwa leap Xe` 11.7577 (15위), `BS-Roformer_Leap by pcunwa (uvr5)` 11.7398 (17위),
`unwa leap` 11.7222 (18위).

우리가 이미 후보로 잡은 `BS-Roformer Voc HyperACEv2`가 11.3957(37위)인 것과 비교하면
**약 +0.36 dB 우위**이고, pcunwa 계열 공개 가중치 중 현재 최상위다. 그 위쪽은 전부 MVSep
사내 모델·앙상블·ByteDance 등 조달 불가 항목이다. 즉 **오픈 가중치로 도달 가능한 천장에
가장 가까운 지점**이 Leap이다.

파일 구성 (HF API 확인):
- `bs_roformer_leap_voc.ckpt` + `bs_leap_voc_conf.yaml` (보컬)
- `bs_roformer_leap_inst.ckpt` + `bs_leap_inst_conf.yaml` (반주)
- `Xe/bs_leap_xe_voc.ckpt`, `Xe/bs_leap_xe_inst.ckpt` + 각 config (Xe 변종, 리더보드상 더 높음)

config 실측값: `dim: 256`, `depth: 16`, `stereo: true`, `num_stems: 1`,
`target_instrument: vocals`, 61밴드(1025 bin 분할), `chunk_size: 881559`(≈20초 @44.1kHz),
inference `batch_size: 2`, `dim_t: 1101`, `num_overlap: 2`.

**주의점 두 가지.**
1. **라이선스 미확인.** HF에 라이선스 태그가 없고 모델 카드는 imgur GIF 한 장이 전부다.
   명시적 NC 표기는 없으므로 우리 정책상 실측 후보에는 포함되지만, 채택 시점에 저자 확인이
   필요하다.
2. **chunk_size 881559는 매우 크다.** 통상 BS-Roformer(352800≈8초) 대비 2.5배다. batch 2로
   추론하면 VRAM 요구가 크게 오른다(12–16GB 추정). 3090 24GB에서 batch 1로 내려 돌리는 것을
   전제로 계획하는 편이 안전하다.

### A2. BS PolarFormer 오픈 가중치판 — "이미 커버됨" 아님

기존 목록의 `BS PolarFormer 124-band`는 **MVSep 서비스 전용 모델**(리더보드 2026.06.13
버전, vocals 12.0230)로 가중치를 받을 수 없다. 반면 ZFTurbo 저장소 v1.0.20 릴리스에는
**다운로드 가능한 `model_bs_polarformer_float16.ckpt`가 있고 multisong vocals SDR 11.00으로
보고**된다. 이 둘은 별개이며, 후자는 우리가 실제로 돌릴 수 있다. 커버 목록이 앞쪽만 잡고
있어서 실측 가능한 쪽이 누락된 상태다.

아키텍처는 극좌표 기반 위치 임베딩(PoPE)을 쓰는 BS 계열 신규 변종이고, 코드는 MSST
네이티브 지원(MIT)이다. float16 배포라 VRAM 부담도 상대적으로 낮다. **SDR 절대값은 Leap보다
낮지만(11.00 vs 11.76), 아키텍처 계열이 달라 앙상블 다양성 측면에서 값어치가 있다.**

### A3/A4. anvuew BS-RoFormer 계열 — 라이선스가 명확한 유일한 신규 후보

기존 목록은 anvuew를 De-Reverb(GPL-3.0)로만 잡고 있는데, anvuew는 그 뒤로 보컬 분리 본체
모델을 두 개 더 냈다:
- `anvuew/BS-RoFormer` (2026-04-17, GPL-3.0): `bs_roformer_anvuew_sdr_12.45.ckpt`,
  `bs_roformer_ft1_anvuew_sdr_12.55.ckpt` 두 체크포인트. 데이터셋은 bascurtiz 제공.
- `anvuew/BS_RoFormer_mag` (2026-03-12, GPL-3.0): magnitude 스펙트럼 정확도 특화.

파일명의 12.45/12.55는 **저자 자체 테스트셋 기준이지 MVSep multisong이 아니다.** MVSep
리더보드에서 해당 엔트리를 찾지 못했으므로 Leap과 직접 비교하려면 우리가 같은 세트로
돌려야 한다. 다만 **GPL-3.0으로 명시된 유일한 신규 후보**라 라이선스 리스크가 없다는 점이
실무적으로 크다. anvuew가 karaoke_bs_roformer(2025-10, GPL-3.0, MVSep 뉴스 기준 SDR 10.22)도
내놓은 만큼 계열 전체가 우리 정책과 잘 맞는다.

### B계열 — 시험 가치 제한적

pcunwa의 나머지 2026 릴리스들은 리더보드 등재가 없어 성능을 검증할 방법이 없다.
`BS-EXP-SiameseRoformer`는 모델 카드가 SiameseNorm 논문(arXiv 2602.08064) 링크뿐인데,
그 논문은 **음원 분리 논문이 아니라 Transformer의 Pre/Post-Norm 조화에 관한 일반
정규화 기법 논문**이다. 즉 이 모델은 "새 정규화를 BS-Roformer에 붙여본 실험판"이지 분리
성능을 노린 릴리스가 아니다. `BS-Roformer-Large-Inst`는 반주 전용(자체보고 17.6059)이고
"FNO보다 SDR이 약간 높고 번거로운 설치 절차가 불필요"하다는 저자 노트가 있어, 이미 후보인
Inst FNO의 상위 호환 성격이다 — 반주 쪽을 손볼 때만 의미가 있다.

`BS Mamba2`(ZFTurbo)는 SSM 계열이라 아키텍처 신규성은 있으나 **MUSDB18 100곡만으로 학습되어
vocals SDR 8.82**에 그친다. 원본 TS-BSmamba2는 Apache-2.0이지만 가중치를 공개하지 않아
직접 학습 없이는 쓸 수 없다. **품질파 후보로는 둘 다 탈락.**

### C계열 — 조달 불가 / 참고용

리더보드 상위권을 실제로 점유한 것은 MVSep 사내 모델(`BS Roformer 124 bands 2026.07.10`
12.3339, `BS PolarFormer 124 bands` 12.0230)과 `sami-bytedance-v.1.1`(11.8225)인데 전부
가중치 비공개다. **오픈 가중치 최고점(Leap 11.76)과 리더보드 1위(12.33) 사이의 약 0.57 dB
격차는 현재 조달 수단이 없다.**

리더보드에는 `MLSLABS WCJ`(11.3882), `BS²`(11.3865), `bsrhfi`(11.5277),
`dilettante v2024.10.28hh`(Hunter Hogan, 11.5165), `1245`/`1255`/`1261`, `test`/`test 2x`
같은 익명·내부 테스트 엔트리도 다수 있으나, 검색으로 배포처를 찾지 못했다. 공개 가중치가
없는 개인 실험 제출로 판단된다.

`Diff-VS`(Moises 팀, ICASSP 2026, arXiv 2604.01120)는 확산 모델 기반 보컬 분리로 계열이
완전히 다르지만 가중치·코드 공개 언급이 없다. 논문상으로도 "판별 모델 베이스라인과 대등"
수준이라 SDR 도약은 아니다. 향후 공개되면 재검토 대상.

### 참고: MVSep "Synth Vocals 2026" 리더보드

조사 중 발견한 것으로, MVSep에 `/quality_checker/leaderboard/synth_vocals_2026/`
전용 리더보드가 있다. **합성 보컬 대상 분리 평가**라 이 프로젝트의 합성보컬 정렬 붕괴
문제와 직결되는 벤치마크다. 다만 현재 등재 엔트리가 MVSep 사내 모델 2건
(`BS Roformer 124 bands 2026.07.10` vocals 17.4025, `BS Roformer 2025.07.20` 17.2056)뿐이라
모델 후보 발굴에는 쓸 수 없다. **우리 합성보컬 셋을 만들 때 지표 설계 참고자료로는 유용하다**
(SI-SDR, L1 freq, log WMSE, AURA-STFT, AURA-MRSTFT, bleedless, fullness 컬럼 구성).

---

## 3. "이미 커버됨" 판정 근거

다음은 리더보드/저장소에서 확인했으나 기존 목록에 이미 있어 신규로 세지 않았다.

- `Deux + SCNet (Regular/High Fullness)` 11.6312/11.5844 → becruily deux 커버됨
- `BS Roformer SW (6 stems)` 11.3019 → BS-RoFormer SW 53-stem 계열로 커버됨
- `MVSep Ensemble` 각 버전 (11.93 등) → MVSep 앙상블/Mega 커버됨
- `SCNet XL IHF` 11.11 → 커버됨
- `MDX23C` 10.36 → 커버됨
- `MelBand Roformer (KimberleyJensen)` 10.98 → Kim MelBand RoFormer 커버됨
- `BS Roformer (viperx)` 10.87, `MelBand Roformer (viperx)` 9.67 → ZFTurbo 릴리스 체크포인트 커버됨
- `htdemucs`, `Demucs4 Vocals 2023` 9.04 → 커버됨
- `BS-Roformer Voc HyperACEv2` 11.3957 → 커버됨 (신규 후보 비교 기준선으로 사용)
- anvuew De-Reverb 계열 → 커버됨 (단 anvuew **보컬 본체** 모델은 미커버, A3/A4 참조)

ZFTurbo `pretrained_models.md` 전체 vocals 표를 확인한 결과, 위 목록과 BS PolarFormer(A2),
BS Mamba2(B5)를 제외하면 **미커버 상위 vocals 모델은 없다.** 아키텍처 지원 목록에 있는
`conformer`, `bs_conformer`, `scnet_tran`, `scnet_masked`는 공개 체크포인트가 MUSDB18HQ 소형
모델(SCNet Tran Small avg 8.92, SCNet Masked Small avg 8.81)뿐이라 품질파 후보가 아니다.
논문상 vocals 12.2 dB로 보고된 `SCNet6(L)`은 공개 가중치를 찾지 못했다.

---

## 4. 출처

- [MVSep Multisong Leaderboard (vocals 정렬)](https://mvsep.com/quality_checker/multisong_leaderboard?sort=vocals) — 1~2페이지 확인
- [MVSep 전체 리더보드 목록](https://mvsep.com/quality_checker/other_leaderboards)
- [MVSep Synth Vocals 2026 리더보드](https://mvsep.com/quality_checker/leaderboard/synth_vocals_2026/)
- [MVSep 알고리즘 목록](https://mvsep.com/en/algorithms) · [MVSep 뉴스](https://mvsep.com/en/news)
- [ZFTurbo Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) (MIT) · [pretrained_models.md](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) · [릴리스](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases)
- [HF pcunwa 프로필](https://huggingface.co/pcunwa) · [BS-Roformer-Leap](https://huggingface.co/pcunwa/BS-Roformer-Leap)
- [HF anvuew/BS-RoFormer](https://huggingface.co/anvuew/BS-RoFormer) · [BS_RoFormer_mag](https://huggingface.co/anvuew/BS_RoFormer_mag)
- [TS-BSmamba2 (Apache-2.0, 가중치 미공개)](https://github.com/baijinglin/TS-BSmamba2) · [arXiv 2409.06245](https://arxiv.org/abs/2409.06245)
- [Diff-VS, arXiv 2604.01120](https://arxiv.org/abs/2604.01120)
- [SiameseNorm, arXiv 2602.08064](https://arxiv.org/abs/2602.08064)
- [Apollo, arXiv 2409.08514](https://arxiv.org/abs/2409.08514)

---

## 5. Part 1 3줄 요약 — 품질파 (시험 가치 순)

1. **BS-Roformer Leap Xe (pcunwa, 2026-06)** — multisong vocals 11.7577로 오픈 가중치 중
   사실상 최상위이며 우리 기준선 HyperACE v2 대비 +0.36 dB. 라이선스는 미확인이나 NC 표기가
   없어 실측 대상. chunk 881559(≈20초)라 VRAM 여유를 두고 batch 1로 돌릴 것.
2. **BS PolarFormer 오픈 가중치판 (ZFTurbo v1.0.20, float16)** — 기존 목록이 잡은 MVSep
   서비스 전용 124밴드판과 **별개로 다운로드 가능한** 체크포인트(vocals 11.00). PoPE 극좌표
   임베딩으로 계열이 달라 앙상블 다양성 확보에 유리하고 코드가 MIT다.
3. **anvuew/BS-RoFormer ft1 (GPL-3.0, 2026-04)** — 신규 후보 중 유일하게 라이선스가 명시된
   보컬 본체 모델. 자체보고 12.55는 multisong 기준이 아니라 직접 실측이 필요하지만,
   라이선스 리스크가 없어 채택 경로가 가장 짧다.

**공통 전제:** 공개 리더보드에 Bleedless 데이터가 사실상 없으므로, 아카펠라 순도 판정은
위 3종을 우리 셋으로 직접 돌려 Bleedless를 실측해야 한다. SDR 순위만으로는 순도를 대리할 수 없다.

---

# Part 2 — 실시간급/경량 분리 후보

용도: 클라이언트(브라우저 WebGPU / 로컬 CPU) 또는 스트리밍 처리에서 경량 정렬기
(NeMo Conformer ONNX급)와 짝지을 분리 모델.

## 6. 결론 먼저 — "진짜 스트리밍 분리"는 정렬 입력으로 못 쓴다

품질 하한 기준선(우리 실측): **htdemucs(SDR ~9급)으로 owsm 정렬은 버티고, omniasr는
kimft(MelBand)급이 필요했다.**

그런데 인과(causal)·저지연 스트리밍 분리의 현재 SOTA는 이렇다.

| 모델 | 지연 | 파라미터 | MUSDB18-HQ vocals SDR | 전체 SDR |
|------|------|----------|----------------------|----------|
| HS-TasNet (L-Acoustics, ICASSP 2024) | 23 ms | 42 M | — | 4.65 (추가 데이터 시 5.55) |
| RT-STT (arXiv 2511.13146, 2025-11) | 23 ms | **383 K** | 5.56 | 5.17 |

**하한선 대비 3~4 dB 아래다.** htdemucs(~9)조차 owsm 기준 겨우 버티는 선인데, 스트리밍
계열은 그 절반 수준이다. 정렬 입력으로 투입하면 보컬 잔향·반주 누출이 CTC posterior를
직접 무너뜨릴 구간이 나온다. **따라서 "저지연 인과 분리" 축은 이 프로젝트에서 탈락이다.**

실질적으로 유효한 경로는 **비인과(non-causal) 청크 처리를 유지하되 모델·런타임을 경량화**하는
쪽이다 — 지연은 곡 단위로 감수하고, CPU/브라우저에서 돌아가게 만드는 방향. 아래 후보들은
그 기준으로 정렬했다.

## 7. 경량 후보 요약표

| # | 후보 | 계열/런타임 | RTF / 속도 | 품질 (vocals SDR) | 가중치 라이선스 | 코드 라이선스 | CPU-only | 브라우저 이식 | 판정 |
|---|------|-------------|-----------|-------------------|-----------------|---------------|----------|---------------|------|
| R1 | **demucs-onnx / StemSplit HT-Demucs ONNX** | ONNX Runtime | **0.20**(단일 스템) ~ 0.49(full bag) @ M4 Pro CPU; GPU EP ≥5× | **9.19** | **MIT** | MIT | **가능** | **검증됨** (onnxruntime-web 스캐폴딩) | **1순위** |
| R2 | **demucs-web** (timcsy) | onnxruntime-web, WebGPU/WASM | 미공개 | HTDemucs급 (~9) | **MIT** | MIT | 가능(WASM) | **실사례 확인** (라이브 데모) | 브라우저 PoC 레퍼런스 |
| R3 | **Mini-BS-RoFormer-18M** (HiDolen) | transformers custom_code | 미공개 (17.9M, 10,115 GFLOPs/30초) | **10.03** (MUSDB18HQ val) | **MIT** | MIT | 가능(추정) | ONNX 익스포트 선행 필요 | **2순위** |
| R4 | **BSRoformer.cpp** (chenmozhijin) | GGML C++ | **벤치마크 미공개** | 원본 체크포인트 따라감 | 체크포인트별 | **MIT** | 가능 | WASM 빌드 별도 작업 | 실측 필요 |
| R5 | bs-roformer-infer (PyPI) | PyTorch 추론 툴킷 | 미공개 | 원본 체크포인트 따라감 | 체크포인트별 | MIT | 가능(`map_location="cpu"`) | 불가 | 배포 편의 도구 |
| R6 | Mini-BS-RoFormer-V2-46.8M | transformers custom_code | 8,343 GFLOPs/30초 (18M보다 **낮음**) | 10.86 (MUSDB18HQ val) | **CC-BY-NC-4.0** | — | — | — | **NC — 정책상 제외** |
| R7 | HS-TasNet | PyTorch (lucidrains/temismink) | 23 ms 지연, 실시간 | 4.65 | **사전학습 가중치 없음** | MIT | — | — | **탈락** (품질+가중치) |
| R8 | RT-STT | 논문만 | 23 ms 지연, GPU 5.80→1.01 ms(양자화) | 5.56 | **미공개** | 미공개 | — | — | **탈락** (조달 불가+품질) |
| R9 | Open-Unmix (UMX-L) | PyTorch / ONNX | 경량 | ~7급 (구세대) | MIT | MIT | 가능 | ONNX 사례 있음 | 하한 미달, 베이스라인용 |
| R10 | Spleeter | TensorFlow | 빠름 | ~6급 | MIT | MIT | 가능 | — | **유지보수 중단(2022), 제외** |

## 8. 후보별 노트

### R1. demucs-onnx (StemSplit) — 1순위, 경량축의 실질 정답

2026-05 공개. **HT-Demucs FT의 최초 성공적 ONNX 익스포트**로, 그동안 아무도 못 뚫던 4가지
블로커를 해결했다: 복소 STFT 텐서(`torch.stft` → sin/cos 커널 Conv1d), `fractions.Fraction`
산술, `random.randrange`(추론은 결정적이므로 하드코딩), fused attention 커널
(`_native_multi_head_attention` → 표준 ONNX 연산). PyTorch FP32 대비 최대 절대오차
1.71×10⁻⁴로 패리티 검증됐다.

수치가 우리 요구에 정확히 맞는다.
- **vocals SDR 9.19** — 하한 기준선 htdemucs와 같은 급. owsm 정렬은 버티는 선.
- **RTF 0.20** (단일 보컬 스페셜리스트, M4 Pro CPU). 3분 곡을 CPU만으로 약 36초.
  full bag은 0.49. ONNX CPU가 PyTorch CPU보다 1.31× 빠르고, GPU EP(CUDA/CoreML/DirectML)는
  5배 이상.
- **모델 크기**: 보컬 스페셜리스트 316 MB, 6스템 258 MB, full bag 1.26 GB. **FP16 변종은
  다운로드 크기 절반이고 런타임 성능은 동일**.
- **MIT** (원본 HT-Demucs와 동일). 7개 모델 저장소 전부 패리티 검증 + MIT.
- iOS(CoreML) / Android(NNAPI) / 브라우저(WASM) / CUDA 검증 완료. `onnxruntime-web` 데모
  스캐폴딩을 CLI로 생성해준다.

**한계 하나는 분명히 해둘 것.** SDR 9.19는 우리 스펙트럼에서 "owsm은 OK, omniasr는 위험"
지점이다. omniasr 경로에 이걸 물리면 품질 회귀가 난다.

### R2. demucs-web (timcsy) — 브라우저에서 실제로 돌아간 사례

MIT, HTDemucs ONNX(~172 MB)를 onnxruntime-web으로 브라우저에서 4스템 분리한다.
[라이브 데모](https://timcsy.github.io/demucs-web/)가 있다. WebGPU 우선, 미지원 시 WASM
폴백.

**운영상 걸림돌 하나**: SharedArrayBuffer 때문에 `Cross-Origin-Opener-Policy: same-origin` +
`Cross-Origin-Embedder-Policy: require-corp` 헤더가 필요하다. 크롬 확장/웹앱에서 이 헤더를
붙이면 서드파티 리소스 로딩이 같이 깨질 수 있으니, 도입 시 격리 컨텍스트 설계를 먼저 봐야
한다. 성능 수치는 README에 없다 — 우리가 재보는 수밖에 없다.

### R3. Mini-BS-RoFormer-18M — 파라미터 대비 품질이 가장 좋음

MIT, **17.9M 파라미터**(depth 8, dim 256, intermediate 768)로 MUSDB18HQ validation 평균
SDR 9.01, **vocals 10.03**. 파라미터가 htdemucs(~40M+)의 절반 이하인데 vocals가 더 높다.
safetensors + `custom_code`(transformers 4.55.4, `modeling_bs_roformer.py` 동봉),
청크 264,600 샘플(6초) / overlap 3초.

**비교 주의**: 이 10.03은 MUSDB18HQ validation 기준이고 Part 1의 multisong SDR과 척도가
다르다. 직접 줄세우면 안 된다. 다만 같은 MUSDB18HQ 척도에서 htdemucs를 상회할 가능성이
있으므로, **경량축에서 품질 상한을 올릴 유일한 MIT 후보**다.

브라우저로 가려면 ONNX 익스포트를 우리가 해야 한다. BS-Roformer 계열의 onnxruntime-web
실사례는 찾지 못했다.

### R6. Mini-BS-RoFormer-V2-46.8M — 아깝지만 NC로 탈락

후속 V2는 **vocals 10.86**으로 18M판보다 확실히 낫고, 시간축 stride 4 다운샘플링 덕에
파라미터가 늘었는데도 **연산량은 오히려 적다**(30초 오디오 기준 8,343 GFLOPs vs 18M판의
10,115). 경량축 관점에서 가장 매력적인 수치인데, 라이선스가 **CC-BY-NC-4.0**이다.
우리 정책은 "명백한 NC만 제외"이므로 **이건 명백한 NC라 제외 대상**이다. 18M판(MIT)으로
가야 한다.

### R4. BSRoformer.cpp — 요청하신 "실측 속도 보고"는 존재하지 않는다

이미 기존 목록에 있는 항목이라 신규 후보는 아니지만, **속도 실측 보고를 찾아달라는 주문에
대한 답은 "공개된 것이 없다"** 이다. README에 벤치마크 표가 없고, 검색으로도 RTF·곡당
처리시간 보고를 찾지 못했다. 확인된 것은 스펙뿐이다.

- 양자화: FP32 / FP16(50%) / **Q8_0(25%, 권장)** / Q5_1(18%) / Q4_0(12.5%).
  K-Quant는 gguf-py 제약으로 미지원.
- 백엔드: CPU / CUDA / Vulkan(q8·fp16에서 `NV_coopmat2` 기본 활성). **Metal 없음.**
- BS Roformer + Mel-Band Roformer 자동 판별. MIT.

즉 "Q8_0으로 몇 배 빨라지는가"는 우리가 직접 재야 하는 미지수다. 경량축 계획을 여기에
의존시키려면 실측이 선행 조건이다.

### R5. bs-roformer-infer — 경량화 수단이 아니라 배포 편의 도구

2026-07-12 공개(0.1.5), MIT. 추론 전용 툴킷으로 10개 모델 레지스트리(BS-RoFormer-SW 권장),
SHA256 검증 자동 다운로드, CLI + Python API. **ONNX·양자화 지원이 없고 속도 수치도 없다.**
PyTorch 그대로이고 CPU는 `map_location="cpu"`로 가능한 수준이다. 기본 모델이 ~700 MB.

경량화에는 기여하지 않지만, **서버측 파이프라인의 체크포인트 관리·다운로드 검증을 단순화**
하는 용도로는 값어치가 있다.

### R7/R8. 스트리밍 계열 — 조달과 품질 양쪽에서 막힘

**HS-TasNet**: L-Acoustics 논문(ICASSP 2024, arXiv 2402.17701). 23 ms 지연에 SDR 4.65,
추가 학습 데이터로 5.55. lucidrains 구현이 MIT이고 `sounddevice_stream()`으로 라이브 추론까지
지원하지만 **사전학습 가중치가 없다 — 직접 학습해야 한다.** 42M 파라미터를 우리가 학습할
이유가 없다.

**RT-STT**: arXiv 2511.13146(2025-11). **383 K 파라미터**로 HS-TasNet(42M)의 1/100인데 SDR은
5.17로 더 높다. GPU 추론 5.80 ms이고 양자화 후 1.01 ms(82.6% 감소). 아키텍처적으로는 매우
인상적이지만 **코드·가중치 공개 언급이 없다.**

둘 다 §6에서 정리한 대로 품질 하한을 크게 밑돈다.

### R9/R10. 구세대 — 참고용

**Open-Unmix**: MIT, 2024-04에 torch 2.0 대응까지 갔고 UMX-L의 ONNX 익스포트 사례가 있다.
다만 품질이 구세대(~7급)라 하한 미달. Sony의 X-UMX(CrossNet-Open-Unmix) 파생이 있으나
역시 세대가 다르다. 레퍼런스 베이스라인으로만.

**Spleeter**: **Deezer가 2022년에 유지보수를 중단했다.** 저장소가 아카이브되진 않았지만
inactive이고 distutils 등 의존성 문제로 최신 Python에서 설치가 깨진다. **2025–26 후속작은
없다.** 제외.

### 인접 참고: 음악 향상(enhancement) 계열

arXiv 2607.12872 "Low-Latency Neural Models for Real-Time Music Enhancement"(2026-07-14,
Widmer 그룹)는 노이즈·리버브·아티팩트 제거용 인과 모델 벤치마크로 MusicFilterNet-MS를
제안한다. 전 모델이 실시간보다 빠르게 동작한다. **분리가 아니라 향상이라 범위 밖**이지만,
분리 후 보컬 스템 정리(de-reverb 대안)로는 인접 관련이 있다. 가중치 공개 언급 없음.

## 9. Part 2 3줄 요약 — 경량/실시간

1. **저지연 스트리밍 축은 접어라.** HS-TasNet 4.65 / RT-STT 5.17로 우리 품질 하한
   (htdemucs ~9, omniasr는 kimft급)보다 3~4 dB 아래다. 게다가 HS-TasNet은 사전학습 가중치가
   없고 RT-STT는 코드·가중치가 미공개다. 유효한 경로는 "비인과 청크 처리 + 런타임 경량화"뿐이다.
2. **demucs-onnx(StemSplit, MIT, 2026-05)가 경량축 1순위다.** vocals SDR 9.19로 하한을
   정확히 충족하고, M4 Pro CPU에서 RTF 0.20(단일 보컬 스템), FP16 변종은 크기 절반에 성능
   동일, iOS·Android·브라우저·CUDA 검증 완료다. **owsm 정렬에는 충분하고 omniasr에는 위험**
   하다는 점만 못박아 둘 것.
3. **품질을 더 올리려면 Mini-BS-RoFormer-18M(MIT, 17.9M, MUSDB18HQ vocals 10.03)이 유일한
   후보다.** 단 척도가 multisong과 달라 직접 비교 불가라 실측이 필요하고, 브라우저로 가려면
   ONNX 익스포트를 우리가 뚫어야 한다. 성능이 더 좋은 후속 V2-46.8M은 **CC-BY-NC-4.0이라
   정책상 제외**다.

**브라우저 이식성 실무 함의:** 현재 브라우저에서 돌아간 것이 확인된 계열은 **Demucs뿐**이다
(demucs-web 라이브 데모, demucs-onnx의 onnxruntime-web 스캐폴딩). Roformer 계열은
onnxruntime-web 실사례를 찾지 못했고, BSRoformer.cpp는 네이티브 GGML이라 WASM 빌드가 별도
작업이다. HT-Demucs의 ONNX 익스포트가 2026-05에야 처음 성공했다는 사실(4가지 블로커)이
"Roformer를 브라우저로 옮기는" 작업의 난이도를 가늠하게 해준다 — 가볍게 잡을 일이 아니다.

## 10. Part 2 출처

- [HS-TasNet, arXiv 2402.17701](https://arxiv.org/abs/2402.17701) · [L-Acoustics PDF](https://www.l-acoustics.com/wp-content/uploads/2024/04/real_time_demixer_2024_04_19.pdf) · [lucidrains 구현(MIT, 가중치 없음)](https://github.com/lucidrains/HS-TasNet)
- [RT-STT, arXiv 2511.13146](https://arxiv.org/abs/2511.13146) · [HTML 전문](https://arxiv.org/html/2511.13146v1)
- [demucs-onnx (StemSplit, MIT)](https://github.com/StemSplit/demucs-onnx) · [PyPI](https://pypi.org/project/demucs-onnx/) · [익스포트 기술 노트(2026-05)](https://stemsplit.io/blog/htdemucs-ft-onnx-export) · [HF htdemucs-6s-onnx](https://huggingface.co/StemSplitio/htdemucs-6s-onnx)
- [demucs-web (timcsy, MIT)](https://github.com/timcsy/demucs-web) · [라이브 데모](https://timcsy.github.io/demucs-web/)
- [Mini-BS-RoFormer-18M (MIT)](https://huggingface.co/HiDolen/Mini-BS-RoFormer-18M) · [V2-46.8M (CC-BY-NC-4.0)](https://huggingface.co/HiDolen/Mini-BS-RoFormer-V2-46.8M)
- [BSRoformer.cpp (MIT, GGML)](https://github.com/chenmozhijin/BSRoformer.cpp)
- [bs-roformer-infer (PyPI, MIT, 2026-07-12)](https://pypi.org/project/bs-roformer-infer/)
- [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) · [Spleeter (유지보수 중단)](https://github.com/deezer/spleeter)
- [Low-Latency Neural Models for Real-Time Music Enhancement, arXiv 2607.12872](https://arxiv.org/abs/2607.12872)
