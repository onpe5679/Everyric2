# benchmarks — 벤치마크·데이터셋 조사 (SendMessage 전문 1건)

조사 완료했습니다. 항목별로 정리합니다. URL·라이선스는 확인된 것만 적었고, 확인 못 한 부분은 명시했습니다.

---

## 1. 반드시 확인 요청하신 4가지

### JamendoLyrics (원본, 단어 단위)
- URL: https://huggingface.co/datasets/jamendolyrics/jamendolyrics (신규, 활성) / https://github.com/f90/jamendolyrics (구버전, 2025-04-30부터 미갱신·deprecated 표시)
- 라이선스: 곡마다 다른 CC 계열 혼합 (CC BY, BY-SA, BY-NC, BY-NC-SA, BY-NC-ND, BY-ND) — 데이터셋 전체 단일 라이선스 아님, 곡별 확인 필요
- 규모: **79곡** (구 README는 "80곡"으로 표기했으나 현재 HF 페이지 기준 79곡 — 곡 하나가 빠진 것으로 보임)
- **언어: 영어·프랑스어·독일어·스페인어 4개뿐. 일본어·한국어 없음.** (Durand et al., ICASSP 2023 "Similarity-based Audio-Lyrics Alignment of Multiple Languages"에서 공개)
- 어노테이션: **단어 단위** 시작/끝 타임스탬프 + 줄 단위도 함께 제공
- 오디오: HF `datasets` 라이브러리로 오디오 포함 로드 가능(`audio` 컬럼), 또는 git clone(Git LFS)
- **바로 쓸 수 있는지**: 포맷·다운로드 면에서는 예. 다만 일본어·한국어가 없어 저희 타겟을 대표하지 못함 → "영어/유럽어 기준선과 비교해 우리가 얼마나 떨어져 있나"를 보는 참고용으로만 유용.

주의: **Jam-ALT** (https://huggingface.co/datasets/jamendolyrics/jam-alt, 79곡, en/fr/de/es 각 20/19/20/20)는 같은 Jamendo 곡을 쓰지만 **가사 전사(ASR) 포맷팅 평가용 벤치마크**이고 **줄 단위 타이밍만** 있음(단어 단위 아님). 정렬 정밀도 평가 목적이 아니라서 혼동하지 않는 게 좋습니다.

### DALI (v1/v2)
- URL: https://github.com/gabolsgabs/DALI (어노테이션 + `dali_code` 파이썬 패키지, `pip install dali-dataset`)
- 라이선스: **CC BY-NC-SA 4.0 — 비상업적 목적만**
- 규모: v1 5,358곡(344.9시간) / v2 7,756곡(488.1시간)
- 언어: 30개 이상 언어 수록, **영어가 80% 이상으로 압도적**. 200곡 이상인 언어는 영어·프랑스어·독일어·스페인어·이탈리아어뿐. **일본어·한국어의 정확한 곡수는 확인 못 함**(원 논문 TISMIR 2020 Table 2를 직접 열람해야 함 — 검색으로는 못 찾음. 존재 자체는 가능성 있으나 소수일 것으로 추정)
- 어노테이션: notes/words/lines/paragraphs 4단계 (단어 단위 포함). **단, 자동 생성(teacher-student 방식)이라 수작업 골드 스탠다드보다 노이즈가 있는 것으로 알려짐** — 평가용 그라운드트루스로 쓰기엔 주의 필요
- 오디오 획득: **오디오 자체는 배포 안 됨.** `dali_code.get_audio()`가 각 주석에 연결된 유튜브 영상에서 오디오를 자동 다운로드하는 방식 → yt-dlp 계열 문제(링크 삭제/지역제한/403)를 그대로 겪을 수 있음 ([[vocaro-pronunciation-e2e]]에서 이미 겪으신 이슈와 동일 계열 리스크)
- **바로 쓸 수 있는지**: 부분적. 언어 구성이 타겟과 안 맞고, 오디오 확보가 유튜브 가용성에 의존하는 불안정한 경로. 학습용 대규모 데이터로는 좋지만 "정확한 평가"용 골드스탠다드로는 이상적이지 않음.

### AutoLyrixAlign (MIREX 2019)
- URL: https://github.com/chitralekha18/AutoLyrixAlign
- 라이선스: **GPL v3** (코드·모델), 상업적 이용은 별도 라이선스 필요
- 사전학습 모델·스크립트는 저장소가 아니라 **Google Drive 링크**로 배포되고, 실제 요구사항(OS/RAM/GPU/Kaldi 버전 등)은 그 안의 별도 README에만 있어 **원격 조사로는 확인 못 함**(실제 다운로드해야 확인 가능). 확인된 것: 약 20GB RAM 권장, 자동 다운로드·압축해제에 ~16GB 여유 디스크 필요
- 개발: HLT-NUS (Chitralekha Gupta, Emre Yilmaz, Haizhou Li). ASR 기반 시스템으로 보이나 Kaldi 사용 여부는 원격으로 명확히 확인 못 함
- **언어**: 영어 전용으로 강하게 추정됨(HLT-NUS의 영어 ASR 학습 이력, 논문들이 전부 영어 벤치마크로만 평가) — **일본어·한국어 지원은 확인 못 함, 사실상 없다고 보는 게 안전**
- MIREX 2019 "우승" 여부: 검색으로 명확한 공식 문구를 못 찾음(비공식 언급만 있었음) — MIREX 결과 페이지(`2019:MIREX2019_Results`)를 직접 봐야 확정 가능, **확인 못 함**
- **같은 곡으로 비교 가능한지**: 영어 곡이면 가능. 일본어·한국어 곡에는 적용 불가할 가능성이 높음 → 저희 시스템의 "영어 트랙 처리 성능"을 곁다리로 점검하는 용도로만 의미 있음.

### MIREX 평가 지표 정확한 계산식
원 출처: **Mauch, M., Fujihara, H., & Goto, M. (2012). "Integrating Additional Chord Information Into HMM-Based Lyrics-to-Audio Alignment." IEEE TASLP, 20(1), 200-210.** (PCS는 이 논문이 Fujihara et al. 2011을 인용해 제안)

- **AAE (Average Absolute Error)**: 각 어노테이션 단위(단어/줄 등)의 예측 타임스탬프와 정답 타임스탬프 간 절대 오차를 모든 이벤트에 대해 평균. `AAE = mean(|t̂ᵢ - tᵢ|)`
- **PCS (Percentage of Correct Segments)**: 정답으로 "올바르게 라벨된" 구간 길이의 합 / 곡 전체 길이. 템포에 대한 지각적 의존성을 완화하기 위해 절대오차 대신 채택.
- **PCO (Percentage of Correct Onsets, 관용창 0.3초)**: 정확한 식을 Deezer ISMIR 2021 논문(User-centered evaluation of lyrics-to-audio alignment, Lizé Masclef et al.)에서 확인함:
  ```
  ρ_τ^k = (1/N_k) · Σ_word_i  1[|t̂ᵢ - tᵢ| < τ] × 100      (τ=0.3s가 MIREX 표준)
  ```
  이 논문은 흥미로운 사실도 보고합니다: **0.3초 관용창에는 실제 심리학 실험으로 검증된 근거가 없었고**, 저자들이 처음으로 카라오케 실험을 통해 검증함 → 지각적으로는 **비대칭**(가사가 오디오보다 "먼저" 나오는 것에 더 관대함, lyrics-ahead −0.3s vs lyrics-lagging +0.2s가 50% 지각 임계값)이 확인됨. 저희가 나중에 지표를 다듬을 때 참고할 만합니다.

---

## 2. mir_eval — 지표를 직접 구현하지 않아도 되는 공개 평가 코드 (강력 추천)

- URL: https://github.com/craffel/mir_eval (모듈: `mir_eval/alignment.py`) / 문서: https://mir-eval.readthedocs.io/latest/api/alignment.html
- 설치: `pip install mir_eval` — MIT 라이선스, MIR 분야 표준 평가 라이브러리(Raffel et al., ISMIR 2014)
- 구현된 함수 (소스 직접 확인함):
  - `absolute_error()` — 절대오차의 **중앙값(MAE)**과 **평균(AAE)**을 함께 반환
  - `percentage_correct()` — 관용창(기본 0.3초) 내 정확도 비율 (PCO와 동일 개념)
  - `percentage_correct_segments()` — 구간 중첩 길이 / 전체 길이 (PCS)
  - `karaoke_perceptual_metric()` — 위 Deezer 논문의 비대칭 지각 가중치(skew-normal) 기반 지표까지 구현되어 있음
  - `evaluate()` — 위 전부를 dict로 반환하는 원스톱 함수
- **저희 상황에 대한 함의**: 자체 지표(자막 시각과의 평균 거리)를 대체하거나 병행해서 이 라이브러리로 MIREX 표준 지표를 바로 계산할 수 있습니다. 재구현 리스크(공식 오류) 없이 바로 씀직합니다.

참고로 `georgid/AlignmentEvaluation` (https://github.com/georgid/AlignmentEvaluation)도 MIREX 대회 실사용 평가 스크립트로 `.lab`/TextGrid 입력을 지원하지만, 유지보수 상태가 mir_eval보다 못해 **mir_eval을 우선 권장**합니다.

---

## 3. 일본어·한국어 가창 데이터셋 (정렬 있음)

| 이름 | URL | 라이선스 | 규모 | 어노테이션 |
|---|---|---|---|---|
| **Tohoku Kiritan Singing DB** (일본어) | 배포 페이지: https://zunko.jp/kiridev/login.php (Facebook 로그인 필요) / 최신 라벨: https://github.com/mmorise/kiritan_singing | 확인 못 함 | 프로 여성 가수 1인, 50곡, 약 57분, 일본 팝송 | **음소 경계 라벨 있음** (a cappella, MusicXML 변환) |
| **JVS-MuSiC** (일본어) | https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music | 개인적 사용만 허용, 재배포 금지 명시(정확한 CC 여부는 태그만 BY-SA 4.0, 오디오 자체는 별도 조건) — **상세 조건 원문 확인 필요** | 100명 화자가 동일곡("かたつむり") + 화자별 상이곡 1곡씩 | 다중 화자 비교용, 정렬 어노테이션 형태는 확인 못 함 |
| **Ofuton-P DB** (일본어) | 배포처("おふとんP歌声DB配布所") — 정확한 URL은 확인 못 함(일본어 이용약관 사이트, 자동 링크 제공 안 됨). NNSVS 레시피: https://github.com/taroushirani/nnsvs_ofuton_p_utagoe_db | 확인 못 함 | 남성 보컬 1인, 56곡, 약 61분 | 확인 못 함 (NNSVS 레시피 존재로 보아 음소 정렬 있을 가능성 높음) |
| **PJS** (일본어, 음소 밸런스) | https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus | **CC BY-SA 4.0** | 짧은 노래 100곡 + 낭독 100문장, 48kHz | "음소 밸런스"는 음소 커버리지 균형을 뜻하며, 시간정렬 라벨 존재 여부는 확인 못 함(추가 확인 필요) |
| **CSD** (한국어+영어) | https://github.com/equal-singer/CSD (KAIST MAC Lab, Zenodo 배포) | **CC BY-NC-SA 4.0**, 원 가수를 비방하는 용도 명시적 금지 | 한국어 50곡 + 영어 50곡(동일 가수), 2개 조옮김씩 = 오디오 200개 | **MIDI 노트 단위 정렬** + 문자/음소 단위 가사. 단, "음소별 정렬 라벨은 미포함 — MIDI 노트로 음절 타이밍을 추정하는 방식"이라고 명시되어 있어, 완전한 강제정렬 골드스탠다드는 아님 |

**한국어 쪽은 CSD가 사실상 유일하게 찾은, 정렬이 붙은 공개 데이터셋입니다.** K-pop 관련 정렬 어노테이션 데이터셋은 검색으로 찾지 못했습니다(확인 못 함 — 존재하지 않거나 비공개일 가능성).

---

## 4. 그 외 가창 데이터셋 (일반, 참고용)

| 이름 | 언어 | 라이선스 | 규모 | 정렬/음소 라벨 |
|---|---|---|---|---|
| **NUS-48E** | 영어 | NUS 비상업 연구/시범 라이선스(신청 필요), https://www.comp.nus.edu.sg/~nlp/corpora.html | 12인, 48곡(20곡 유니크), 169분 | **음소 단위 전사+지속시간** (25,474 phone instances), 노래+말 병렬 |
| **Opencpop** | 중국어(만다린) | **CC BY 4.0**(비상업), 상업 이용은 이메일 문의 | 프로 여성 가수 1인, 100곡 | **TextGrid 음소/노트/발화 경계 정렬 있음** — https://github.com/wenet-e2e/opencpop |
| **DSing** | 영어 | Smule DAMP 데이터 이용약관에 따름(신청 필요) — 확인 필요 | 약 4천 개 카라오케 녹음, ~8만 발화, 3,205명, ~150시간 | 전사 텍스트만, 정렬 골드스탠다드 없음(강제정렬로 자체 생성해야 함). Kaldi 레시피: https://github.com/groadabike/Kaldi-Dsing-task |

이들은 전부 일본어/한국어가 아니라서 저희 핵심 타겟에는 직접 안 맞지만, Opencpop은 "TextGrid 강제정렬 파이프라인 검증용"으로, NUS-48E는 "영어 음소 정렬 정확도 참고선"으로 쓸 수 있습니다.

---

## 5. 반복 실행(정렬 비결정성) 평가 관행

**명시적으로 확립된 관행은 찾지 못했습니다(확인 못 함).** 가사 정렬 분야에서 "같은 입력을 여러 번 돌려 분포로 평가"하는 논문이나 MIREX 프로토콜은 검색으로 발견하지 못했습니다. 다만 참고할 만한 근접 사례는 있습니다:
- Deezer ISMIR 2021 논문(위 3번 언급)은 3개 시스템을 20곡 Jamendo 데이터셋에 평가하며 **곡 간 표준오차(standard error)를 함께 보고**합니다(Table 1: `94.47 (1.52)` 형태). 이건 "곡마다의 편차"를 보고하는 관행이지, "같은 곡 반복실행 편차"는 아닙니다 — 저희가 마주친 21.74초짜리 비결정성 문제와는 결이 다릅니다.
- LLM 평가 분야에서는 "반복 실행 후 평균±표준편차/분산 분해" 관행이 최근 정착되고 있지만(예: repeated trial 재평가), 이는 가사 정렬이 아닌 다른 분야 사례입니다.

**결론**: 저희가 이 관행을 새로 세워야 할 가능성이 높습니다. 곡당 42초로 계산하면 반복 비용이 크지 않으니, 곡별로 N회 반복 후 평균+표준편차(또는 이상치에 강건한 median+IQR)를 함께 보고하는 방식을 저희 쪽에서 표준화하시는 걸 제안합니다.

---

## 「저희 시스템 평가를 위한 최소 셋업」 제안

**데이터셋 조합**:
1. **JamendoLyrics** (HF, 79곡, en/fr/de/es, 단어 단위) — 다국어 시스템으로서 국제 기준선과 비교하는 용도. 일본어·한국어가 없다는 한계를 명시하고 사용.
2. **CSD** 한국어 50곡 — 저희 핵심 타겟(한국어) 평가. 단, MIDI 노트 기반 정렬이라 저희 줄/단어 단위 형식으로 변환하는 전처리가 필요.
3. **Kiritan Singing DB** 일본어 50곡 (음소 경계 라벨) — 일본어 평가. Facebook 로그인 절차가 있어 접근성 확인 필요.

3개 합쳐 179곡, RTX 3090 곡당 42초 기준 약 2시간 — 무리 없는 규모.

**지표**: 자체 지표 대신 **mir_eval.alignment**(`pip install mir_eval`)의 `absolute_error`(AAE/MAE), `percentage_correct`(PCO, 0.3초), `percentage_correct_segments`(PCS)를 표준으로 채택. 저희 자막 줄 단위 그라운드트루스를 그대로 쓰되, JamendoLyrics·Kiritan처럼 단어/음소 단위 정답이 있는 셋에서는 단어 단위로도 비교해서 "줄 단위 대비 단어 단위에서 성능이 얼마나 갈리는지"까지 볼 수 있습니다.

**반복 횟수**: 실측된 최대 21.74초 편차를 감안하면 **곡당 최소 5회, 가능하면 10회** 반복 실행 후 평균+표준편차 보고를 제안합니다(확립된 외부 기준은 없으므로 저희가 정하는 값입니다 — 5회면 평균의 표준오차가 어느 정도 안정화되고, 179곡×5회×42초 ≈ 10.4시간으로 하룻밤 배치가 가능합니다).

**바로 시작 가능한 것**: JamendoLyrics 다운로드 + mir_eval 설치는 지금 바로 가능합니다. CSD·Kiritan은 라이선스/접근 절차(CSD는 비상업 연구 조건 동의, Kiritan은 로그인) 확인이 먼저 필요합니다.

---

전체적으로 "확인 못 함"이라고 표시한 항목들(DALI의 일본어/한국어 정확한 곡수, AutoLyrixAlign의 상세 요구사항과 MIREX 우승 공식 확정, Ofuton-P/JVS-MuSiC의 정확한 라이선스 조건, PJS의 시간정렬 라벨 유무)은 원 논문/사이트를 직접 열람해야 확정됩니다. 데이터셋을 실제로 받지 않고 조사만 한 결과라 이 정도가 한계입니다.
