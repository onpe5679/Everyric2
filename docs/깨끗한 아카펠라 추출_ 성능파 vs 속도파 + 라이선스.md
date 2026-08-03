<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# 깨끗한 아카펠라 추출: 성능파 vs 속도파 + 라이선스

## Where Models Agree

| Finding | GPT-5.6 Sol Thinking | Claude Opus 5 Thinking | Gemini 3.1 Pro Thinking | Evidence |
| :-- | :-- | :-- | :-- | :-- |
| 아카펠라 목적에선 SDR보다 Bleedless가 핵심 지표 | ✓ | ✓ | ✓ | 반주 잔재 억제도가 아카펠라 순도를 결정[^1][^2] |
| BS-Roformer Revive 2가 Bleedless 1위 = 성능파 1순위 | ✓ | ✓ | ✓ | Viperx 1297 파인튜닝, 현존 보컬 모델 중 최고 Bleedless[^3][^4] |
| HyperACE v2는 SDR 11.39로 단일 모델 최고급 | ✓ | ✓ | ✓ | Multisong SDR 11.3957, aura_mrstft 최고[^5][^2] |
| Kim Mel-Band RoFormer가 속도·품질 균형 기본값 | ✓ | ✓ | ✓ | 공식 추론 패키지의 권장 기본 모델[^6][^7] |
| 추론 코드(MSST·Demucs·UVR·lucidrains)는 MIT | ✓ | ✓ | ✓ | 네 저장소 모두 MIT 명시[^8][^9][^10][^11] |
| becruily Deux는 위상보정 불필요한 올라운더 | ✓ | ✓ | ✓ | SDR 보컬 11.35 / 인스트 17.66[^12][^1] |

## Where Models Disagree

| Topic | GPT-5.6 Sol Thinking | Claude Opus 5 Thinking | Gemini 3.1 Pro Thinking | Why They Differ |
| :-- | :-- | :-- | :-- | :-- |
| unwa/becruily 가중치 라이선스 | ⚠️ 명시 없음 → 상업 비추천 | 저장소별 확인 필요 (추정) | MIT 체계 내 자유 사용 가능 | Gemini는 상위 구현체 MIT를 가중치에 상속 적용, GPT-5.6은 모델 카드에 YAML/LICENSE 부재를 근거로 상속 부정[^5][^3] |
| Kim 가중치 라이선스 | MIT (재라이선스 완료) | **GPL-3.0** → 조건부 | 언급 없음 | Claude는 2025-06 GPL-3.0 부여 시점, GPT-5.6은 2026-04-22 MIT 재라이선스 커밋 반영[^13][^14] |
| 속도파 최우선 도구 | HTDemucs `--two-stems` | **BSRoformer.cpp (GGML)** | Kim Mel-Band | Claude만 C++/GGML 구현체 존재를 파악, 나머지는 파이썬 생태계 내에서만 판단[^15] |
| 속도 수치 제시 여부 | 통합 벤치마크 없음 → 수치 거부 | 튜닝 원리로 대체 | VRAM 8~24GB 범위 제시 | GPT-5.6은 동일조건 벤치마크 부재를 엄격 적용, Gemini는 실무 경험적 범위 제시 |
| 최종 워크플로우 | 2단계 선별 재처리 | Revive2+HyperACE 앙상블+TTA | 3단계 파이프라인(분리→코러스→디리버브) | Gemini만 De-Reverb를 필수 단계로 포함[^16] |

## Unique Discoveries

| Model | Unique Finding | Why It Matters |
| :-- | :-- | :-- |
| Claude Opus 5 Thinking | Kim 모델 + TTA 시 보컬 SDR 12.76 / 인스트 12.46[^17] | 처리시간 2배로 품질 최상단 도달 가능한 숨은 레버 |
| Claude Opus 5 Thinking | BSRoformer.cpp: `--chunk-size` 기본 352800(~8초), `--overlap` 권장 2~4[^15] | 파이썬 오버헤드 없는 최속 경로 + 정확한 튜닝 파라미터 |
| Claude Opus 5 Thinking | BS Roformer 기반 **53스템** 분리 모델 공개[^18] | 아카펠라 외 세부 악기 stem 확장 가능 |
| Claude Opus 5 Thinking | MLX 변환판(ZFTurbo vocals v1) 존재, MIT 상속[^19] | Apple Silicon 가속 경로 |
| GPT-5.6 Sol Thinking | HyperACE v2 vocals의 Bleedless는 34.08로 Revive 2보다 낮음[^5] | "SDR 1위 = 가장 깨끗함"이 아님을 증명 |
| Gemini 3.1 Pro Thinking | Revive 2의 Bleedless 실측 수치 **40.07**[^4] | 최고치의 정량적 근거 |
| Gemini 3.1 Pro Thinking | Gabox Karaoke V2 + De-Reverb 3단계 파이프라인[^16] | Dry 아카펠라 완성을 위한 실전 체인 |

## Comprehensive Analysis

### 고신뢰 결론

세 모델이 완전히 일치한 가장 중요한 지점은, 아카펠라 추출에서 **SDR을 1차 기준으로 삼으면 잘못된 선택을 하게 된다**는 것입니다. GPT-5.6 Sol Thinking은 평가 우선순위를 "Bleedless → 청감상 보컬 손상 → Fullness → SDR" 순으로 명시했고, Claude Opus 5 Thinking과 Gemini 3.1 Pro Thinking 역시 Bleedless를 핵심 지표로 지목했습니다. 이 합의를 뒷받침하는 결정적 데이터가 GPT-5.6 Sol Thinking이 찾아낸 수치입니다. HyperACE v2 vocals는 SDR 11.3957로 단일 모델 최고급이지만 Bleedless는 34.0758에 불과하며, Fullness가 19.0952로 매우 높습니다. 반면 Gemini 3.1 Pro Thinking이 확인한 Revive 2의 Bleedless는 **40.07**로 압도적입니다. 즉 HyperACE v2는 "보컬을 최대한 온전히 담는" 모델이고, Revive 2는 "반주를 최대한 배제하는" 모델로, 사용자님의 목적(깨끗한 아카펠라)에는 **Revive 2가 정확히 부합**합니다.[^4][^2][^3][^5][^1]

또한 세 모델 모두 **Revive 3e를 선택하면 안 된다**는 점을 직간접적으로 시사합니다. 제작자 본인이 Revive 3e는 Revive 2와 정반대로 Fullness를 극한까지 밀어붙인 모델이라고 명시했기 때문입니다. 동일한 논리로 becruily의 max fullness 계열(fullness 20.72, bleedless 31.25)도 Claude Opus 5 Thinking이 명확히 배제 대상으로 지목했습니다.[^3][^1]

추론 코드 라이선스 계층은 완전히 확정된 사실입니다. ZFTurbo의 Music-Source-Separation-Training, Meta의 Demucs, Anjok07의 UVR GUI, lucidrains의 BS-RoFormer 구현체 모두 MIT입니다. Demucs는 2020년 4월 13일 MIT로 전환되었고 상업적 이용에 제한이 없습니다.[^11][^8][^9][^20][^21][^10]

### 쟁점 1 — Kim 가중치 라이선스: 이 불일치는 시점 문제입니다

이 항목이 이번 조사에서 가장 실무적으로 중요한 분기점입니다. Claude Opus 5 Thinking은 Kim Mel-Band-Roformer 가중치를 **GPL-3.0**으로 판정하고 제품 통합 시 소스 공개 의무를 검토해야 한다고 경고했습니다. GPT-5.6 Sol Thinking은 동일 체크포인트를 **MIT**로 판정했습니다.[^13][^14]

두 판정 모두 각자의 근거 시점에서는 정확합니다. 실제 이력을 보면 2025년 6월 17일 원저자가 GPL-3.0을 부여했고, **2026년 4월 22일 커밋 `ac9b061`로 MIT로 재라이선스**되었으며, 2026년 4월 25일 리포지토리 `license: mit` 배지가 확인되었습니다. 즉 Claude Opus 5 Thinking은 2025년 시점 자료를, GPT-5.6 Sol Thinking은 2026년 재라이선스 이후 자료를 근거로 삼은 것입니다. **현재 기준으로는 MIT가 유효한 판정**이며, 이는 상업 프로젝트에서 Kim 모델을 쓰실 때 결정적으로 유리한 사실입니다. 다만 Claude Opus 5 Thinking이 함께 지적한 anvuew dereverb 모델의 GPL-3.0은 별개 사안이므로, Gemini 3.1 Pro Thinking이 제안한 De-Reverb 단계를 상업 파이프라인에 넣으실 경우 해당 체크포인트를 따로 확인하셔야 합니다.[^6][^14][^13]

### 쟁점 2 — unwa/becruily 가중치: 보수적 해석이 안전합니다

Gemini 3.1 Pro Thinking은 unwa 가중치가 MIT 체계 내에서 동작하므로 상용 편입도 비교적 자유롭다고 판단했습니다. GPT-5.6 Sol Thinking은 반대로 Revive 2와 HyperACE v2 모델 카드에 YAML 메타데이터와 라이선스 선언이 아예 없다는 점을 근거로 상업 사용을 보류하라고 권고했습니다. Claude Opus 5 Thinking은 중간 입장으로 저장소별 개별 확인을 요구했습니다.[^5][^3]

여기서는 **GPT-5.6 Sol Thinking의 보수적 해석이 더 타당합니다**. 이유는 HyperACE 모델 카드 자체가 "이 가중치는 anvuew의 BS-RoFormer 가중치에 기반한다"고 밝히고 있어, 파생 체인상 상위 가중치의 권리관계까지 추적해야 하기 때문입니다. 코드가 MIT라는 사실이 제3자가 별도로 학습시킨 체크포인트에 자동 상속되지는 않습니다. 다만 becruily의 경우 Hugging Face 프로필에 "License / Commercial Use of This Model"이라는 제목의 모델 카드 항목이 존재하는 것이 확인되므로, 해당 페이지를 직접 열어 확인하시면 명확한 답을 얻으실 수 있습니다.[^22][^5]

### 쟁점 3 — 속도: 수치보다 파라미터가 실질 레버입니다

속도 항목에서 GPT-5.6 Sol Thinking은 동일 GPU·동일 segment·동일 overlap·동일 정밀도로 전 후보를 측정한 통합 벤치마크가 존재하지 않으므로 "몇 배 빠르다"는 수치 제시를 거부했습니다. 이 판단은 방법론적으로 정확합니다. 대신 Claude Opus 5 Thinking이 제시한 튜닝 원리가 실용적으로 더 유용합니다. Segment size는 클수록 RAM↔VRAM 스왑이 줄어 빨라지고 overlap 의존도가 낮아져 결과도 좋아지므로 **VRAM 허용 최대치로 설정**해야 하며, overlap은 청크 재조합 아티팩트를 줄여 품질을 올리지만 추론 시간을 늘리므로 **권장값 2~4** 범위에서 조절하는 것이 속도·품질의 실질적 레버입니다.[^15][^23]

Claude Opus 5 Thinking만 발견한 **BSRoformer.cpp**는 속도파에서 가장 주목할 만한 발견입니다. GGML 기반 C++ 구현으로 파이썬 오버헤드를 제거하며, `--chunk-size` 기본값 352800(44.1kHz 기준 약 8초)과 `--overlap` 파라미터를 직접 제어할 수 있고 MIT 라이선스입니다. 사용자님처럼 C 기반 최적화 코드에 익숙하고 로컬 배포를 중시하는 환경에서는 파이썬 경로보다 이 쪽이 더 적합할 가능성이 높습니다.[^15]

### 주목할 단일 발견

Claude Opus 5 Thinking이 찾은 **TTA(Test-Time Augmentation) 수치**는 이번 조사에서 가장 실용적 가치가 큰 단일 발견입니다. Kim 모델에 TTA를 적용한 MVSep 측정치가 보컬 SDR 12.76 / 인스트 12.46으로 나타나, 일반 추론 대비 상당한 향상을 보입니다. 처리 시간이 배로 늘어나는 대가가 있지만, 최종 마스터 1~2곡만 처리하실 때는 최고 품질 도달 경로가 됩니다. 또한 BS Roformer 기반 **53스템 분리 모델**이 공개되어 있다는 사실은, 아카펠라를 넘어 세부 악기 stem 작업으로 확장하실 때 유용한 정보입니다.[^17][^18]

Gemini 3.1 Pro Thinking이 제시한 **3단계 파이프라인**도 개념적으로 중요합니다. 1차로 BS Roformer SW로 보컬을 분리하고, Gabox Karaoke V2로 리드/백킹을 나눈 뒤, BS-Roformer-De-Reverb(SDR 22.5050)로 잔향을 제거해 Dry 아카펠라를 완성하는 구조입니다. "깨끗한"의 정의에 리버브 제거까지 포함하신다면 이 3단계가 반드시 필요하며, 단일 모델 선택만으로는 도달할 수 없는 결과입니다.[^16]

### 권장 세팅

사용자님 목적 기준으로는 **BS-Roformer Revive 2를 기본 모델로 확정**하시고, overlap 2로 시작해 아티팩트가 보이는 곡만 overlap 4로 재처리하시는 것이 가장 효율적입니다. 상업적 배포 계획이 있으시다면 라이선스가 MIT로 확정된 **Kim Mel-Band RoFormer 또는 ZFTurbo 릴리스 체크포인트**를 우선 검토하시고, unwa 계열은 개인 작업에 한정하시는 것이 안전합니다. 속도가 병목이라면 BSRoformer.cpp + Kim 모델 조합에 chunk-size를 VRAM 최대치로 설정하시고, 반대로 최종 마스터 1~2곡은 Revive 2 결과와 HyperACE v2 결과를 청감 비교한 뒤 TTA까지 적용하시면 현재 도달 가능한 최상단입니다. 리버브 제거까지 필요하시면 De-Reverb 단계를 추가하되, 해당 체크포인트의 GPL-3.0 여부를 상업 사용 전 확인하셔야 합니다.[^19][^23][^8][^14][^16][^3][^5][^13][^17][^15]
<span style="display:none">[^24][^25][^26][^27][^28][^29][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^60][^61][^62][^63][^64][^65]</span>

<div align="center">⁂</div>

[^1]: https://lyruno.com/blog/remove-vocals/ai-vocal-separation-guide-latest

[^2]: https://mvsep.com/algorithms/34

[^3]: https://huggingface.co/pcunwa/BS-Roformer-Revive

[^4]: https://zenn.dev/unwa/articles/9a1d6de5ed5d13

[^5]: https://huggingface.co/pcunwa/BS-Roformer-HyperACE

[^6]: https://pypi.org/project/melband-roformer-infer/

[^7]: https://github.com/openmirlab/melband-roformer-infer

[^8]: https://github.com/ZFTurbo/Music-Source-Separation-Training

[^9]: https://github.com/facebookresearch/demucs/blob/main/LICENSE

[^10]: https://github.com/anjok07/ultimatevocalremovergui

[^11]: https://github.com/lucidrains/BS-RoFormer

[^12]: https://mvsep.com/en/news

[^13]: https://huggingface.co/KimberleyJSN/melbandroformer/discussions/2

[^14]: https://huggingface.co/mlx-community/mel-roformer-kim-vocal-2-mlx

[^15]: https://github.com/chenmozhijin/BSRoformer.cpp

[^16]: https://blog.yexca.net/en/archives/280/

[^17]: https://mvsep.com/quality_checker/entry/7025

[^18]: https://github.com/ZFTurbo/Music-Source-Separation-Training/releases

[^19]: https://huggingface.co/mlx-community/mel-roformer-zfturbo-vocals-v1-mlx

[^20]: https://pypi.org/project/demucs/

[^21]: https://stemsplit.io/blog/demucs-local-setup-guide

[^22]: https://huggingface.co/becruily

[^23]: https://github.com/Anjok07/ultimatevocalremovergui/discussions/831

[^24]: https://pypi.org/project/bs-roformer-infer/

[^25]: https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model

[^26]: https://mvsep.com/algorithms/49

[^27]: https://arxiv.org/html/2305.07489v2

[^28]: https://mvsep.com/quality_checker

[^29]: https://github.com/openmirlab/bs-roformer-infer

[^30]: https://mvsep.com/quality_checker/multisong_leaderboard

[^31]: https://mvsep.com/news/60

[^32]: https://neuralanalog.com/stems/best-ai-stem-separation-model-vocals

[^33]: https://ultimate-vocal-remover.en.uptodown.com/windows

[^34]: https://github.com/lucidrains/BS-RoFormer/blob/main/LICENSE

[^35]: https://huggingface.co/AEmotionStudio/roformer-models

[^36]: https://huggingface.co/pcunwa

[^37]: https://huggingface.co/pcunwa/BS-Roformer-Inst-FNO

[^38]: https://github.com/facebookresearch/demucs

[^39]: https://github.com/Atm4x/Mel-Band-Roformer-Vocal-Model-GUI

[^40]: https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md

[^41]: https://onegen.ai/project/demucs-advanced-music-source-separation-with-hybrid-transformer-technology/

[^42]: https://mvsep.com/quality_checker/leaderboard2.php?sort=vocals

[^43]: https://www.bluebash.co/blog/understanding-hugging-face-ai-model-licensing-commercial-use/

[^44]: https://github.com/KimberleyJensen

[^45]: https://huggingface.co/pcunwa/models

[^46]: https://mvsep.com/quality_checker/multisong_leaderboard?page=46

[^47]: https://huggingface.co/models?p=2\&sort=trending\&search=roformer

[^48]: https://ismir2023program.ismir.net/lbd_353.html

[^49]: https://arxiv.org/html/2603.04032v1

[^50]: https://daglo.ai/license

[^51]: https://github.com/huggingface/transformers/blob/main/LICENSE

[^52]: https://www.olis.or.kr/license/guide/softwareGuide.do

[^53]: https://www.reddit.com/r/MachineLearning/comments/15c89r7/d_huggingface_changed_the_license_of_one_of_its/

[^54]: https://stemsplitter.github.io/research/

[^55]: https://github.com/yaml/pyyaml-future/blob/main/License

[^56]: https://help.arena.ai/articles/4171906875-model_licenses

[^57]: https://www.kaggle.com/code/steubk/music-source-separation-train-on-kaggle

[^58]: https://ultimatevocalremover.com/

[^59]: https://www.reddit.com/r/audioengineering/comments/12iws99/ultimate_vocal_remover_is_holy_sht_level_good/

[^60]: https://mvsep.com/quality_checker/entry/7762

[^61]: https://sourceforge.net/projects/demucs.mirror/

[^62]: https://www.reddit.com/r/LocalLLaMA/comments/1u8eq0g/can_i_get_a_reality_check_on_this_inference_speed/

[^63]: https://arxiv.org/abs/2409.04702

[^64]: https://arxiv.org/html/2505.22758v1

[^65]: https://www.aibase.com/repos/project/bs-roformer

