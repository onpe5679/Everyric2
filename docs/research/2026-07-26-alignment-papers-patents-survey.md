# 가사 정렬 논문·특허 조사 통합 정리 (2026-07-26 조사 팀)

> **출처**: 2026-07-26 밤에 돌린 12-에이전트 조사 팀(팀 세션 `session-cbde2c10`)의 보고 전문을 세션 기록에서 회수해 정리한 문서다. 원문 12건은 [`2026-07-26-survey/`](2026-07-26-survey/)에 그대로 보존했다. 별도 자료로 [음악 분리·피치 모델 조사 보고서](../music_separation_and_pitch_models_report.md)(RoFormer 계열·RMVPE/FCPE 일반론)가 있다.
>
> **당시 맥락**: CTC 정렬의 3대 문제 — ① 간주 구간 가사 오배치, ② 실행 간 비결정성(최대 21.74초 편차 실측), ③ CTC 손실의 구조적 한계(star 축퇴) — 를 문헌으로 검증하는 조사였다. 2026-07-30 시작한 **라이선스 프리 모델 교체 + 최적화 이니셔티브**(과제 A 분리 교체 / B 정렬 음향모델 교체 / C MFA GPU)의 선행 자료로 재정리한다.

---

## 1. 특허 조사 — 상업화 관점 최우선 자료 ([원문](2026-07-26-survey/patents.md))

특허번호는 전부 Google Patents 원문 대조 완료. **법적 침해 판단이 아니라 청구항 문언 대조**이며, 실제 리스크 판단은 변호사 검토 필요.

### 주의가 필요한 등록·유효 특허

| 특허 | 권리자 | 상태 | 핵심 청구 | 우리와의 거리 |
|---|---|---|---|---|
| US20240135974A1 / **EP4362007B1** | Spotify (Durand·Stoller) | US 심사 중 / **EP 등록(2026-02-04)** | 텍스트·오디오 인코더 대조학습 → 유사도 행렬 모노토닉 경로 정렬 (CLIP류) | CTC forced alignment와 구조가 다름 → 문언상 겹침 낮음. **단, Durand식 대조학습 정렬로 갈아타면 정면 충돌** |
| US11245950B1 | Amazon | **Active** | 보컬분리→ASR→기존 가사와 비교해 오프셋 보정 + **누락 단어 자동 삽입** | "missing words 식별·삽입"이 필수 요소 — 우리는 가사 텍스트를 고치지 않으므로 벗어날 여지 |
| US9305530B1 | Amazon | Active (~2034) | ML로 노래/비노래 구간 판별 + **줄 단위 통째 배치** | 음소/단어 단위 정렬이 아니라 다름. 단 "보컬 유무 검출로 줄 타이밍 배치" 로직을 넣으면 가까워짐 |
| US9251796B2, US8686271B2 | Apple(구 Shazam) | Active | 앰비언트 마이크 캡처 + 오디오 지문 → 가사 동기화 | 시나리오 자체가 다름(우리는 원본 오디오 직접 처리) — 낮음 |
| CN106649644B | Tencent Music | Active (~2036) | 크라우드소싱 녹음 다수를 HMM 음소 강제정렬로 비교·선별 | "다수 녹음 비교"가 제한 요소 — 단일 트랙 정렬인 우리와 갈림 |

### 만료·저위험 (기술 아이디어 참고 자유)
- Philips US7915511B2, Sony US8604327B2 — Expired.
- **AIST US9595256B2 계열**(나카노·고토, VocaRefiner): 이미 주어진 요미가나를 Viterbi로 음절 동기화 — Expired. 발음 자동 생성을 청구하지 않음.

### 못 찾은 것 (정직 보고)
NetEase·LINE/NAVER 자체 특허, "일본어→한글/로마자 발음 자동 생성"을 직접 청구하는 특허(가장 근접이 위 AIST, 그마저 발음은 주어진 전제), Apple "Lyrics and Karaoke UI" 정확한 번호. KIPRIS·J-PlatPat는 JS UI라 직접 열람 불가 — Google Patents 인덱스 경유의 한계 있음.

**이니셔티브 함의**: 현행 CTC forced alignment 노선은 조사된 등록 특허들과 문언상 거리가 있다. **Durand식 대조학습 정렬로의 전환은 Spotify EP 등록특허와 정면 충돌 소지**가 있어, 유럽 서비스 계획이 있다면 노선 선택에서 제외하거나 변호사 검토가 선행돼야 한다.

---

## 2. 핵심 논문 6편 정독 ([원문](2026-07-26-survey/core-papers.md) — 25k자, 가장 상세)

세 문제 ①②③의 뿌리는 하나: torchaudio `<star>`가 `log p = 0`(= p 1.0, 모든 프레임 최우선·선호 0)이라 DP가 간주·가사 배치에 아무 선호를 갖지 않는 축퇴.

| 논문 | 핵심 | 우리에게 남은 것 |
|---|---|---|
| Stoller 2019 (ICASSP) | Wave-U-Net 문자 CTC, Mauch AE 0.35s | emission 균등노이즈 플로어, 상수 지연(180ms) 캘리브레이션 — 추론 전용 |
| Gupta 2019 (Interspeech) | Kaldi CNN-TDNN-F + 폴리포닉 도메인 적응 | 간주 전용 기법 **없음**(정정). median 좋고 mean 터지는 실패 프로파일이 분야 표준 미해결임을 확증 |
| Vaglio 2020 (ISMIR) | BLSTM CTC + 보편 IPA 62음소, **blank와 별개의 학습된 instrumental 토큰** | star를 프레임별 점수로 성형하라는 개념적 처방. 음소 전환은 같은 언어 내 이득 없음 → 한국어 자모 분해 반대 근거 |
| Teytaut 2022 (Interspeech) | guided monotony 가우시안 대각 사전 D (학습 전용) | D 수식을 앵커 기반 시간매핑으로 바꾸면 추론 사전확률로 이식 가능(제안 수준) |
| **Durand 2023 (ICASSP)** | 대조학습 정렬 + **2패스 soft line-mask 디코딩(§2.4, 학습 불필요)** | **최고 소득**: 같은 CTC 모델이 마스크 품질만으로 AAE 0.90→0.20. 자막 앵커 보유한 우리에게 최적. 단 모델 자체는 Spotify 특허(§1) |
| Cheng 2025 (DAFx) | masked CE로 onset의 blank 부재 강제, MedAE 0.041s SOTA | 앞뒤 무음 트리밍(에너지 0.05, ±1s 패딩) 즉시 이식 가능. 학습은 3090 1대로 현실적인 유일 후보(단 영어 CMU 음소라 ko/ja 직접 불가) |

**당시 도입 순서 추천**(전부 추론 전용, 학습 0): ① star 점수 성형(앵커·f0 기반, 반나절) ② 앞뒤 무음 트리밍(반나절) ③ emission forward 결정화(1일) ④ Durand 2패스 soft line-mask(2~3일, 효과 최대) ⑤ 노이즈 플로어+지연 캘리브레이션 ⑥ Teytaut D 앵커 일반화.

---

## 3. CTC 한계·대안 논문 ([원문](2026-07-26-survey/align-methods.md))

- **arXiv:2406.02560 "Less peaky and more accurate CTC forced alignment by label priors"** — ③에 가장 실전적. 디코딩 시 label prior로 나누는 보정(α≈0.3), torchaudio 생태계와 같은 라인이라 **재학습 없이 적용 가능성**. Buckeye PBE 44→38ms.
- **Demirel 2021 MSTRE-Net**: `<silence>`/`<music>` **이중 토큰** — 폴리포닉에서만 WER 6.86%p 개선(간주 처리가 폴리포닉에서만 유효함을 실측). 단 vocab 확장 = 파인튜닝 필요.
- **Kick 2025 (ICASSP)**: Durand 직접 후속(box loss + JamendoLyrics++) — 원문 인증장벽, 확인 못 함.
- **HCLAS-X (Kang 2023, arXiv:2307.04377)**: **한국어+영어 6.7만곡 in-house 학습 MAE 0.16** — 한국어 가창 정렬의 몇 안 되는 실측(재현 불가).
- 비결정성 전용 선행연구 없음 — 우리 21.74초 편차 관찰이 드문 실측 사례(논문화 가치 언급).

## 4. 보컬 분리가 정렬에 미치는 영향 ([원문](2026-07-26-survey/separation.md)) — ★과제 A 직결

**"분리기를 더 좋은 걸로 바꾸면 정렬이 나아진다"를 직접 증명하는 문헌은 없다.**
- Gao 1909.10200: 분리 후 정렬이 폴리포닉 전용 음향모델 직접 정렬보다 **훨씬 나쁨**(Mauch 3.62s vs 0.21s — 단 2018년식 Wave-U-Net 기준).
- 2506.15514 / 2506.02339 (Whisper 전사): 분리 효과의 부호가 데이터셋·분리기 조합마다 뒤집힘. **HTDemucs vanilla가 무분리보다 나쁜 사례 실측**(36.80→37.50). 정답 stem과의 격차(14~15 vs 20~24 WER)는 상한 개선 여지.
- 합성보컬 posterior 붕괴를 직접 다룬 논문 없음(문헌 공백) — SVS 명료도 연구의 "자음 조음 단서 약화"가 인접 근거.

**모델 비교 당시 스냅샷**: htdemucs_ft(품질↑ 속도 4배↓), BSRNN(0.7×RT, 부적합), BS-RoFormer(SDX23 1위 11.99dB, 공식 가중치 비공개), Mel-RoFormer 24kHz 경량판(9.1M params 11.01dB — 가중치 공개 불확실), **TFC-TDF-UNet v3(15×RT 최속, vocals 9.2~9.4dB)**.

**당시 권고**: 교체 전에 **붕괴 곡(quality_score≈0.001 보카로) 소수로 "분리 유/무" A/B 진단**부터 — 분리 아티팩트가 범인인지 합성음색 도메인 불일치가 범인인지 가르는 게 먼저.

## 5. 멜로디·노트 ([원문](2026-07-26-survey/melody-notes.md))

- **f0 벤치마크**(lars76/pitch-benchmark): 가창 기준 RMVPE 최고(87.2%), FCPE는 벤치마크 미포함. **SwiftF0**(2508.18440, 96k params, CPU서 CREPE 42배 속도)이 속도 후보.
- 노트 온셋을 정렬 제약으로: Dzhambazov 2016(HMM+온셋, +5.5%p), **jhuang448 MTL**(ICASSP 2022, phone-CTC+pitch 멀티태스크, AAE 0.31→0.23, 학습 필요, 코드 공개).
- Wav2Karaoke(NLP2026): cascade 파이프라인 실측 성능 낮음 — "음절=1노트" 가정 실패. 노트 최고 방법도 매칭률 30%대(Nishikimi 2017) → **줄 단위 노트 부여가 옳았다는 실측 근거**.
- 값싼 제안: **FCPE voicing confidence를 CTC 디코딩 prior로 재사용**(간주 프레임에 비-blank 페널티) — 추가 모델·학습 0.

## 6. 벤치마크·평가 인프라 ([원문](2026-07-26-survey/benchmarks.md)) — ★회귀 검증 기반

- **mir_eval.alignment**(`pip install mir_eval`, MIT): AAE/MAE·PCO(0.3s)·PCS·카라오케 지각 지표까지 구현 완료 — 자체 재구현 불필요.
- 데이터셋: JamendoLyrics(79곡 en/fr/de/es, 단어 단위 — ja/ko 없음), **CSD(한국어 50곡, CC BY-NC-SA — 한국어 정렬 붙은 사실상 유일 공개셋)**, Kiritan(일본어 50곡 음소 경계), DALI(CC BY-NC — 학습용, 오디오는 유튜브 의존).
- 지각 연구(Deezer ISMIR 2021): 0.3s 관용창은 심리 실험 근거가 없었고, 실제 지각은 **비대칭**(가사 선행 −0.3s vs 지연 +0.2s).
- 반복 실행 평가 관행 없음 → 곡당 5~10회 반복 + 평균·표준편차 보고를 자체 표준으로 제안.

## 7. 비결정성 ([원문](2026-07-26-survey/determinism.md))

- `forced_align` CUDA 커널 자체는 **결정적**(compute.cu 확인). 흔들리는 건 **emission** — `cli.py:22-26`이 TF32를 전역 활성화하고 결정적 모드는 미설정인 조합이 원인 유력.
- CTCLoss 비결정성 통념(#17798)은 backward(학습) 얘기 — 추론 전용인 우리와 무관(정정).
- (후속: Demucs shifts→0으로 분리 비결정성은 이미 해결 — `ctc-alignment-failure-modes` 메모리 참조.)

## 8. 음소 표현·표기·조달 ([원문](2026-07-26-survey/phonemes.md), [transliteration](2026-07-26-survey/transliteration.md), [lyrics-sourcing](2026-07-26-survey/lyrics-sourcing.md))

- **MMS kor 어댑터 vocab 실측: 한글 1,261자 완성형 음절**(자모 아님) — ctc_engine.py 주석 실측. `docs/GPU_ALIGNMENT_ENGINES.md`는 구버전 설계라 현행과 불일치.
- 자모 분해 반대 근거: Wang 2019 — 한국어 ASR 음절 WER 2.6% vs 자모 19.9%. Vaglio·Durand 모두 "같은 언어 내 음소 전환 이득 없음".
- 표기: 우리 일→한 독음은 국립국어원 표준과 2지점(어두/어중 청탁, 장음)에서 의도적으로 다름(팬 표기 관례) — 근거 주석 존재. 평가는 음절 LCS F-score + 오류 3분류(음절밀림/독음오류/표기차) 제안.
- 가사 조달: LRCLIB(MIT)이 최선 무료 소스, Genius 스크레이핑은 ToS 금지(상용 리스크), 크레딧 줄·반복 펼치기는 연구 공백(우리 규칙 기반이 뒤처진 게 아님).

## 9. 장문 정렬·산업 도구 ([원문](2026-07-26-survey/longform-tools.md), [longform2](2026-07-26-survey/longform2.md)) — ★과제 C 직결

- **MFA 정확도 우위는 speech 기준 사실**: Buckeye PBE/WBE 30/41ms vs 개선된 CTC 44/58ms(2406.02560 Table 1, **ms 단위** — 이전에 돌던 "63.0/70.0 일치율" 수치는 그 논문에 존재하지 않음, 오귀속 정정).
- **MFA 한국어·일본어 사전학습 모델 존재, 라이선스 CC BY 4.0**(상업 가능) — 반면 **torchaudio MMS_FA는 CC-BY-NC**.
- 단 MFA는 30초 미만 세그먼트 요구 → "교체"가 아니라 "앵커 분할 + 세그먼트별 MFA"로만 성립. 가창 포함 학습 여부는 미확인(사이트 429).
- 노래에 그대로 쓸 기성 도구는 사실상 없음: aeneas 스스로 노래 부적합 명시, **SOFA는 코드에 이미 있으나 영어 모델뿐**(일본어 모델 URL 깨짐 — factory.py의 "English/Japanese" 광고와 불일치).
- CTC-Segmentation의 장문 처리는 선형 대각선 밴드 — 우리 `_align_in_blocks`가 옳은 방향이라는 외부 근거. 신뢰도는 평균 대신 **최솟값**으로.

---

## 10. 이번 이니셔티브(2026-07-30~)에 주는 시사점

### 과제 A — Demucs 대체 (분리 모델)
- 선행 조사 결론은 "**정렬 개선 목적의 분리기 교체는 근거 약함**" — 교체 동기를 ①라이선스 ②속도/VRAM ③멜로디 f0 품질로 명확히 하고, 정렬 회귀는 A/B로만 판정할 것.
- Mel-RoFormer 논문(2409.04702)의 "분리+멜로디 전사 동시"는 당시 가중치 공개 불확실 — 신규 조사 결과 전사 헤드는 노트 이벤트 출력·체크포인트 미공개로 f0 경로 대체 불가 확정([track-A](2026-07-30-model-replacement/track-A-separation.md)).
- 속도 최우선이면 TFC-TDF-UNet v3 계보 재확인 가치.

### 과제 B — MMS-1b-all 대체 (정렬 음향모델)
- vocab 제약이 핵심: 후보는 **한글 완성형 음절(또는 자모라도 결합 가능한) 문자 단위 CTC 헤드**여야 하고, 자모 분해 강제는 반대 근거 실측 있음.
- MMS_FA·MMS 계열의 CC-BY-NC가 라이선스 문제의 진원 — 신규 조사에서 모델카드 기준 CC-BY-NC-4.0 재확인, 대체 후보 확정([track-B](2026-07-30-model-replacement/track-B-ctc-model.md)).
- 합성보컬 posterior 붕괴는 어느 모델로 가도 따라오는 도메인 문제(문헌 공백) — 교체 검증 세트에 보카로 붕괴 곡을 반드시 포함.

### 과제 C — MFA 한국어 GPU 개조
- 정확도 전제는 speech에선 사실, singing에선 미검증. 라이선스는 오히려 MFA가 유리(CC BY 4.0).
- GMM-HMM GPU 개조의 현실성 + SOFA류 가창 특화 정렬기(한국어 모델 유무) — 신규 조사 결과 GPU 개조 비추천·SOFA 한국어 모델 없음 확정([track-C](2026-07-30-model-replacement/track-C-mfa-gpu.md)).
- 도입 형태는 "전면 교체"가 아니라 "앵커 분할 + 세그먼트 정렬" 또는 "고정밀 검증 심판" 역할이 현실적.

### 공통 — 회귀 없는 검증 (사용자 최우선 제약)
- mir_eval 표준 지표 + 실곡 세트(DB 실곡 + 보카로 붕괴 곡 + CSD 한국어 50곡 검토) + 곡당 반복 실행으로 A/B. `real-song-only-defects`·`release-verification-standard` 메모리 원칙 적용.
