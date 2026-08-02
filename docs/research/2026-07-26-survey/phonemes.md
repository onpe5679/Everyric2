# phonemes — 음소 중간표현·다국어 조사 (SendMessage 전문 1건)

조사 완료. PDF 원문(Vaglio 2020, CLAP-IPA)을 직접 읽었고, 우리 코드베이스(`ctc_engine.py`)의 실측 주석도 근거로 반영했습니다. 파일은 수정하지 않았습니다.

## 1. Vaglio et al. 2020 (ISMIR), Multilingual Lyrics-to-Audio Alignment
**서지**: A. Vaglio, R. Hennequin, M. Moussallam, G. Richard, F. d'Alché-Buc, "Multilingual Lyrics-to-Audio Alignment," ISMIR 2020. PDF: https://program.ismir2020.net/static/final_papers/101.pdf (CC BY 4.0)

**핵심 내용** (원문 직접 확인, Table 1/2 수치):
- 아키텍처: RNN(BiLSTM 3층)+CTC, **처음부터 학습**하는 소형 모델(우리처럼 사전학습 파운데이션 모델을 그대로 쓰는 게 아님).
- 언어: en/de/fr/es/it (source, DALI 데이터셋) + pt/pl/fi/nl (zero-resource, 평가 전용). **전부 라틴 알파벳 유럽어. 한국어·일본어는 없음.**
- Universal phoneme set = **62개 음소** (Phonemizer로 IPA 생성 → 9개 언어의 음소를 합집합). Language sharing factor 5.35 (평균 5~6개 언어가 음소 1개를 공유). 이건 PHOIBLE 전체(2000+ 언어)가 아니라 **이 9개 언어만으로 만든 태스크 특정 집합**입니다.
- 문자 기반 출력 = 라틴 알파벳+공백+악기토큰+blank = 30개 클래스. 음소 기반 = 62음소+같은 부가토큰 = 65개 클래스.
- 평가지표: AAE(초, 작을수록 좋음), PCO(0.3초 이내 정확도 %, 클수록 좋음). **정확한 수치표는 본문에 없고 Fig.2 박스플롯 + "정확한 수치는 supplementary materials" 라고만 되어 있어, 저희도 정확한 AAE/PCO 격차 숫자는 확인 못 함.** 결론 텍스트는 명확: "다국어 학습 + universal phoneme 표현이 모든 경우 최상", "문자 대비 음소가 거의 항상 낫고, 다국어 학습셋에서 격차가 더 크다", "저자원/제로자원 언어에서 특히 효과".
- 흥미로운 부수 정보: 비교 대상 GYL1 시스템은 "extended lexicon to cope with **long vowels duration in singing**"을 씀 — 가창용 발음사전을 따로 확장한 선례입니다.

**적용 가능성**: **제한적**. (1) 학습 전제 — 저희는 "학습 안 함" 제약, 이 논문은 다국어 학습 자체가 이득의 핵심 메커니즘. (2) 언어 커버리지 — 한국어/일본어 없음, 전부 같은 문자체계(라틴) 안에서의 문자 vs 음소 비교라 "다른 문자체계 자체를 못 다룬다"는 문자 표현의 약점이 두드러진 셋업. 저희는 애초에 언어별 다른 어댑터(vocab)를 쓰므로 이 약점이 이미 없음.
**구현 비용**: 그대로 적용 불가(학습 필요, 저자원 언어용 GTP 도구·발음사전 구축 필요).

## 2. 한국어 자모 분해 — 이득이 아니라 반증 근거를 찾음
**핵심 발견**: 자모(초성/중성/종성) 분해 단위가 "정렬"에 도움된다는 논문은 못 찾았습니다. 대신 **인접 태스크(ASR 전사)에서 자모 단위가 음절 단위보다 명백히 나쁘다는 실측 반증**을 찾았습니다.

- **서지**: J. Wang, J. Kim, S. Kim, Y. Lee (VUNO Inc.), "Exploring Lexicon-Free Modeling Units for End-to-End Korean and Korean-English Code-Switching Speech Recognition," arXiv:1910.11590 (2019). https://arxiv.org/abs/1910.11590
- **수치** (hybrid CTC/Attention, Zeroth-Korean 51.6h): 음절 단위 **WER 2.6%**, 자모 단위 **WER 19.9%** (7배 이상 악화), 자모+서브워드 그룹핑해도 **WER 4.3%**로 음절보다 여전히 나쁨. CER은 자모-서브워드(2k)가 75.3%까지 치솟음.
- **원인**(저자 설명, Fig.3): 자모 단위는 같은 단어("학교")가 여러 다른 자모 시퀀스 표현으로 붕괴(collapse)할 수 있어 시퀀스투시퀀스 디코더가 헷갈림.
- **주의**: 이건 **자유 전사(free decoding) ASR**이지 **강제정렬(forced alignment)**이 아닙니다. 강제정렬은 정답 토큰열이 고정이므로 이 "표현 붕괴" 문제 자체는 안 생길 수 있습니다. 그래도 "잘게 쪼갤수록 무조건 좋다"는 가정에 대한 직접 반증 사례로서 가치가 있습니다.
- **결론**: "한국어 받침을 음소로 분해하면 정렬 정확도가 실제로 오릅니까?" → **근거 없음. 오히려 인접 태스크에서는 반대 방향(자모가 더 나쁨) 증거가 있습니다.**

**G2P 도구**:
- **g2pK** (https://github.com/Kyubyong/g2pK) — Apache License 2.0. jamo, python-mecab-ko, konlpy 의존. `to_syl=False`로 자모 출력 가능. 연음·비음화·경음화 등 규칙 기반 처리.
- **KoG2P** (https://github.com/scarletcho/KoG2P, Yejin Cho 2017) — 규칙 기반, 라이선스는 저장소에서 직접 확인 못 함(리드미에 명시 안 보임 — "확인 못 함").
- 저희 코드베이스(`everyric2/alignment/ctc_engine.py:203-234`, `_oov_substitute`)에 **이미 자모 분해 로직이 존재**합니다: 된소리→예사소리(ㅃ→ㅂ), 활음 제거(ㅛ→ㅗ), 종성 제거를 초성/중성/종성 인덱스 연산으로 점진 적용해 OOV 한글 음절을 vocab에 있는 가까운 음절로 치환하는 용도입니다. 이건 "완전한 음소 표현 전환"이 아니라 "국소적 자모 연산을 문자 표현 안에서 쓰는" 절충안이고, 이미 실전에서 쓰이고 있습니다.

## 3. 일본어 G2P와 가창 특수성
- **pyopenjtalk** (https://github.com/r9y9/pyopenjtalk) — OpenJTalk 파이썬 래퍼, ESPnet에서 사용. Open JTalk 사전(open_jtalk_dic_utf_8) 자동 다운로드. 라이선스는 OpenJTalk/HTS 라이선스를 별도 확인해야 함(패키지 자체 README에 "두 소프트웨어 라이선스를 확인하라"고만 되어 있음 — 정확한 라이선스 문구는 "확인 못 함").
- **특수 모라**: 促音(っ, 폐쇄 지속시간이 단자음의 2~3배로 늘어나는 게르미네이트), 撥音(ん, 모라 비음), 長音(ー, 장모음) — 셋 다 "1모라"로 세지만 음소적으로는 독립 세그먼트가 필요합니다. OpenJTalk의 full-context label 체계에서는 이들을 별도 음소 기호(cl=촉음, N=발음, 장음은 모음 중복/장음기호)로 표현하는 것이 표준 관행입니다. **다만 "CTC 정렬에서 촉음이 구체적으로 어떻게 실패/성공하는지"를 다룬 논문은 검색으로 못 찾았습니다 — 확인 못 함.**

## 4. MMS 어댑터 실제 토큰 집합 — 코드베이스 실측으로 확정
**결정적 사실**: 저희가 실제로 쓰는 것은 `torchaudio.pipelines.MMS_FA`가 아니라 **`facebook/mms-1b-all`** (HuggingFace `Wav2Vec2ForCTC` + 언어별 어댑터)입니다. `everyric2/alignment/ctc_engine.py:56-77`에 2026-07-25 실측 vocab이 이미 주석으로 박혀 있습니다:

| 어댑터 | 총 토큰 | 한글 | 가나 | 한자 | 라틴소문자 |
|---|---|---|---|---|---|
| kor | 1330 | **1261** | 0 | 2 | 26 |
| jpn | 2268 | 0 | **158** | 2048 | 26 |
| cmn | 4495 | 0 | 4 | **4419** | 26 |
| eng | 154 | 0 | 0 | 8 | 26 |

→ **kor 어댑터의 한글 1261자는 완성형 음절 그대로이지 자모(초성/중성/종성) 분해가 아닙니다.** MMS 어댑터의 vocab은 언어를 막론하고 **문자(음절/한자/가나/알파벳) 단위**이지 음소가 아닙니다. 이건 torchaudio 튜토리얼(https://docs.pytorch.org/audio/2.8/tutorials/forced_alignment_for_multilingual_data_tutorial.html)이 설명하는 별도 경로(`MMS_FA`, uroman 로마자화 후 라틴 문자로만 정렬 — 이것도 문자 단위)와는 다른 모델입니다. 저희 `docs/GPU_ALIGNMENT_ENGINES.md`는 구버전 설계(`kresnik/wav2vec2-large-xlsr-korean`, "한글 자모"라고 적힘)를 기술하고 있는데 **현재 코드(`ctc_engine.py`)와 불일치 — 이 문서는 outdated**로 보입니다.

**결론**: MMS 어댑터는 음소를 받지 않습니다. 음소 표현으로 가려면 어댑터 교체가 아니라 **모델 자체를 통째로 바꿔야** 합니다.

## 5. 학습 없이 음소 표현으로 갈 수 있는가 — CLAP-IPA / IPA-ALIGNER
**서지**: J. Zhu, C. Yang, F. Samir, J. Islam (UBC), "The taste of IPA: Towards open-vocabulary keyword spotting and forced alignment in any language," arXiv:2311.08323 (2023, v2 2024). 코드/체크포인트: https://github.com/lingjzhu/clap-ipa

**핵심 내용** (원문 전체 확인):
- 진짜 **IPA 음소 기반** 사전학습 모델(CLAP-IPA, contrastive) + 미세조정판(IPA-ALIGNER, ForwardSum loss). **사전학습 체크포인트가 공개되어 있어 학습 없이(zero-shot) 쓸 수 있습니다.**
- 문자 기반과 직접 비교(Table 2/3, 같은 데이터로 학습해 통제): 95개 미확인 언어에서 **음소 기반이 모든 개별 언어에서 문자 기반을 능가**(Fig.3). 텍스트 인용: "text-based model generalizes well to Hausa (Latin alphabet) but significantly underperforms in languages with non-Latin alphabet, such as **Tamil, Vietnamese, Japanese**, Arabic, and Cantonese." → **일본어가 명시적으로 문자기반이 특히 나쁜 사례로 언급됨.**
- **한국어/일본어 포함 여부**: 학습 데이터(FLEURS-IPA, Table 12)에 **kor, jpn 둘 다 포함**(kor 5.68h, jpn 5.06h, "seen" 언어). 하지만 **forced alignment 평가셋(TIMIT=영어, DoReCo=14개 저자원 언어)에는 한국어·일본어가 없습니다.** 따라서 "한국어 정렬 정확도가 몇 %다"라는 직접 수치는 **확인 못 함**.
- **forced alignment 수치**(Table 4, TIMIT-Phone F1, 20ms 허용): zero-shot CLAP-IPA-tiny 40.46 / finetuned IPA-ALIGNER-tiny 57.31 vs 기존 HMM 정렬기 MFA 63.0, WebMAUS **70.0**. → **zero-shot도 finetuned도 HMM 기반 정렬기(MFA/WebMAUS)를 못 넘습니다.** 영어 기준으로도 이 정도라, 노래 도메인(반주 잡음, 발음 변형)에서는 이보다 더 나빠질 가능성이 높습니다.
- **가창 도메인 평가는 전무**: 이 논문의 모든 데이터는 발화(speech) 코퍼스(TIMIT, FLEURS, DoReCo, MSWC)이고 노래는 한 번도 다루지 않습니다.

**적용 가능성**: 학습 없이 쓸 수 있다는 점은 저희 제약과 맞지만, (1) 한국어 정렬 정확도가 검증 안 됨, (2) 노래 도메인 검증이 전무, (3) 영어 기준으로도 기존 HMM 정렬기보다 약함 — **지금 프로덕션에 바로 붙이기엔 위험이 큽니다.**
**구현 비용**: 어댑터 교체가 아니라 **정렬 엔진 전체 재작성**(현재 `ctc_engine.py`의 앵커 블록·referee·OOV 치환 로직이 전부 MMS의 `F.forced_align`/`F.merge_tokens` API에 묶여 있음 — CLAP-IPA는 DTW/ForwardSum 기반이라 이 인프라를 재사용 못 함). 학습은 불필요하지만 **소규모 파일럿 검증(한국어 곡 몇 곡으로 A/B)이 선행되어야** 합니다.

## 6. 음소→글자 역매핑 (카라오케 하이라이트 직결)
- CLAP-IPA 논문의 "adaptive average pooling" (Fig.1, §4.2): 음소 시퀀스 길이 N을 문자/단어 길이 N'로 매핑하는 풀링 마스크를 만들어, 한 글자에 속하는 여러 음소의 hidden state를 평균해 그 글자 하나의 벡터로 만듭니다. **이건 표현 학습 단계의 해법이지, "정렬된 음소 구간(시작/끝 시각)을 글자 구간으로 합치는" 문제에 대한 표준 기법은 별도로 명시되어 있지 않습니다.**
- 실무적으로 흔히 쓰이는 방식(TTS/카라오케 파이프라인 일반)은 "한 글자에 속하는 음소들의 duration을 이어붙여 그 글자의 [첫 음소 시작, 마지막 음소 끝]을 글자 구간으로 삼는" 단순 결합입니다만, 이를 학술적으로 검증한 논문은 검색으로 **찾지 못했습니다 — 확인 못 함**. 이 부분은 저희가 직접 설계해야 할 가능성이 높습니다(음소 경계 → 글자 경계로 groupby하는 것 자체는 구현이 단순하지만, 정확도 검증 사례가 없다는 뜻).

## 7. PHOIBLE / IPA / X-SAMPA
- PHOIBLE (https://phoible.org): 2000+ 언어·방언의 음소 인벤토리 데이터베이스. Universal SAMPA는 92개 음(폴리/모노폰 포함)이라는 언급을 찾았으나 1차 출처 확인은 못 했습니다.
- 관련 선행연구: Li et al. 2020 "Universal Phone Recognition with a Multilingual Allophone System" (arXiv:2002.11800) — PHOIBLE과 결합해 2000개 언어별 인식기로 커스터마이즈 가능하다고 주장. CLAP-IPA와 같은 계보(저자원 언어 음소 인식)의 선행연구입니다.

## 8. 가창에서 음소 실현 차이 (모음연장·자음약화)
- Vaglio 2020 자체가 "singing voice exhibits more variety than speech with potentially large phonemes pronunciation variations" 언급, GYL1이 이를 "extended lexicon"으로 대응.
- Y. Teytaut, A. Roebel, "Phoneme-to-Audio Alignment with Recurrent Neural Networks for Speaking and Singing Voice," Interspeech 2021 (https://www.isca-archive.org/interspeech_2021/teytaut21_interspeech.html) — **가창 특화 음소 정렬 연구**. 수치: 최고 모델(PHATT)의 평균 오차가 **발화 16.3ms vs 가창 29.8ms** — 가창이 발화보다 정렬이 거의 2배 어렵다는 정량적 근거.
- PMC 논문(모음연장 배율): 영어 3.0배, 힌디어 2.7배, 페르시아어 2.8배로 노래에서 모음이 늘어난다는 수치(정확한 서지 재확인은 못 했음 — PMC7438159, "Speech variability: A cross-language study on acoustic variations of speaking versus untrained singing").

## 총평: 한국어 처리를 음소(자모 분해)로 바꾸는 것의 이득/비용 판정

**지금 시점에는 권장하지 않습니다.** 이득 쪽 근거(Vaglio 2020)는 학습을 전제로 하고 전부 유럽 라틴어권 실험이라 저희 상황(사전학습 모델만 추론, 한국어·일본어, 노래 도메인)에 그대로 이식되지 않으며, 오히려 인접 태스크(한국어 ASR 전사, Wang et al. 2019)에서는 자모 단위가 음절 단위보다 뚜렷이 나쁘다는 반증 수치(WER 2.6%→19.9%)가 있습니다. 저희가 실제 쓰는 MMS 어댑터는 애초에 음소가 아니라 문자(음절) 단위이므로 "음소로 바꾸기"는 어댑터 스위치가 아니라 **모델 전체 교체**를 뜻하며, 학습 없이 쓸 수 있는 유일한 진짜 음소 기반 후보(CLAP-IPA/IPA-ALIGNER)는 한국어 정렬 정확도가 미검증이고 노래 도메인 평가가 전무하며 영어 기준으로도 기존 HMM 정렬기보다 약합니다(TIMIT-Phone F1 57.31 vs WebMAUS 70.0). 반면 저희가 실측한 kor 어댑터의 실전 커버리지(한글+라틴 혼용곡 0.994)는 이미 준수합니다. 따라서 "완전한 음소 전환"보다는, 이미 코드에 있는 자모 기반 국소 치환(`_oov_substitute`)을 g2pK 같은 규칙 기반 G2P(연음·비음화·경음화 처리)로 확장하는 저비용 개선이 더 합리적이고, 완전 전환을 검토한다면 CLAP-IPA/IPA-ALIGNER를 한국어 곡 소수로 오프라인 파일럿 검증부터 시작할 것을 권합니다.

---
확인 못 한 항목 목록: KoG2P 라이선스, pyopenjtalk 기반 OpenJTalk 정확한 라이선스 문구, 촉음의 CTC 정렬 실패 사례를 직접 다룬 논문, PHOIBLE의 정확한 총 음소 수, Universal SAMPA 92음소의 1차 출처, 음소→글자 duration 비례분배 기법을 검증한 논문, 한국어 대상 CLAP-IPA/IPA-ALIGNER의 정렬 정확도 수치.
