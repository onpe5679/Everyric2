# melody-notes — 음표·멜로디 동시정렬 조사 (SendMessage 전문 1건)

멜로디·음표(노트) 쪽 조사 결과입니다. (제 담당은 f0추출/노트분절/Wav2Karaoke/노트-가사 결합이며, 순수 가사 정렬 방법론은 다른 에이전트가 더 깊이 다룰 것으로 보입니다.)

## 1. Wav2Karaoke — 원문 확보, 전문 읽음

**서지**: Hong Hai Ngo, Yunmeng Li, Steven Coyne, Kamila Szelag-Levine, Keisuke Sakaguchi (Tohoku University / RIKEN). "Wav2Karaoke: A Pipeline to Align Notes and Lyrics from a Song". 言語処理学会 第32回年次大会 (NLP2026), 2026년 3월, pp.687-692. **동료심사 없는 워크숍 발표(non-peer-reviewed, CC BY 4.0)**입니다 — 결과 해석에 주의 필요.
URL(원문 PDF): https://www.anlp.jp/proceedings/annual_meeting/2026/pdf_dir/C2-17.pdf
**코드**: https://github.com/NyokoKei/Wav2Karaoke (공개)

**핵심 방법 — 중요: "동시 정렬"이 아니라 순차(cascade) 파이프라인입니다.**
1. HT-Demucs로 보컬 분리
2. **CREPE**로 f0 추출 (프레임당 10ms, confidence 포함) — 멜로디와 가사를 **독립적으로** 처리
3. **MFA(Montreal Forced Aligner)**로 단어/음소 정렬 (원문 가사 사용, Whisper는 록 장르에서 부정확해 포기)
4. Maximal Onset Principle + Pyphen으로 음소→음절 그룹화 (음절=1노트 가정)
5. **마지막 단계에서만 결합**: 각 음절의 시간구간 내 고신뢰도(>99th percentile) f0 값들의 중앙값을 그 음절의 피치로 할당. 즉 정렬은 따로따로 하고, 사후에 "음절 구간 안의 f0를 모아 대표값 하나 뽑기"만 함 — 저희가 논의한 "노트 온셋을 정렬 제약으로 쓰기"와는 다릅니다.

**결과 (DALI v1 영어 80곡, 72명 가수)**: WER 20.0%, MER 19.8%, CER 6.8%, RPA 16.7%, RCA 45.6%, MFA alignment score **-5.19(음수=나쁨)**. 논문 스스로 "개별 벤치마크 대비 현저히 낮다"고 인정하며, 여성/저속곡이 상대적으로 낫다고 보고. **시사점**: 순차 cascade + "음절=1노트" 단순 가정은 실전에서 잘 안 됩니다 — 저희가 노트를 줄 단위로만 붙이기로 한 결정이 오히려 안전했던 선택으로 보입니다.

**적용 가능성**: 추론만으로 파이프라인 자체는 재현 가능하나, 성능이 낮아 그대로 가져올 가치는 낮음. 다만 "고신뢰도 프레임만 median" 트릭과 MFA beam width(200)/retry beam(800) 세팅은 참고할 만합니다.

## 2. 노트 온셋을 가사 정렬의 양성/음성 제약으로 쓰는 연구 — 정확히 질문하신 아이디어를 다룬 논문 있음

**G. Dzhambazov, A. Srinivasamurthy, S. Şentürk, X. Serra. "On the Use of Note Onsets for Improved Lyrics-to-Audio Alignment in Turkish Makam Music". ISMIR 2016.**
URL: https://www.researchgate.net/publication/315547464 (PDF: https://bpb-us-e1.wpmucdn.com/wp.nyu.edu/dist/2/2294/files/2016/07/243_Paper.pdf)
- 핵심: "다음 가사 음절로의 전이는 대개 새 음표로의 전이를 동반한다"는 가정 하에, 기존 HMM 기반 정렬기의 디코딩에 노트 온셋 정보를 **추가 항**으로 결합. 튀르크 마캄(전통 성악) 독창 구절 정렬에서 **절대 5.5%p 개선**. 저자들 표현으로 "이 유형의 정렬에 보컬 멜로디 온셋을 추론 과정에 포함시킨 첫 시도".
- **적용 가능성/비용**: 추론 시 결합(디코딩 단계 prior 추가)이라 재학습 불필요해 보이나, 원 논문은 HMM 기반 정렬기 대상이라 CTC 기반인 저희 파이프라인엔 직접 이식이 아니라 **원리를 차용**하는 형태가 됨.

**J. Huang et al. "Improving Lyrics Alignment through Joint Pitch Detection". ICASSP 2022.**
arXiv: https://arxiv.org/pdf/2202.01646 / 코드: https://github.com/jhuang448/LyricsAlignment-MTL
- 핵심: 멀티태스크 학습으로 phone-CTC와 pitch(frame-level cross-entropy)를 **하나의 출력 텐서(N_time × N_phone × N_pitch)**로 결합. 손실 = L_phone + 0.5·L_pitch. 디코딩 시 경계탐지(boundary) 확률을 Viterbi argmax에 가중치(α=0.8)로 추가.
- **데이터**: DALI v2(영어, 학습 4224곡/검증 1056곡), 평가 Jamendo/Mauch, 피치 평가 RWC.
- **성능**: Jamendo 단어수준 AAE(평균절대오차) baseline 0.31 → MTL 0.23 (약 26% 개선).
- **적용 가능성/비용**: **학습이 필요**합니다(DALI v2로 처음부터 학습, 사전학습 모델 없음). 다만 구조상 저희의 CTC forced alignment 모델에 pitch head를 하나 더 붙이는 정도라 구현 비용은 중간.

**STARS: Guo, Zhang, Pan et al. (Zhejiang Univ). arXiv 2507.06670 (2025).**
데모: https://gwx314.github.io/stars-demo/ — GitHub 저장소 URL은 확인 못함(논문에 명시 안 됨).
- 5단계 계층(Frame/Word/Phone/Note/Sentence)을 **하나의 공유 인코더**로 처리하는 통합 프레임워크. CTC(음소 정렬) + CE(피치) + BCE(경계) + VQ commitment loss 결합. Lyric Alignment BER/IOU 18.6/80.9, Note Transcription COnPOff(F)/RPA 71.0/86.7. GTSinger/OpenCPop/VocalSet 사용, NVIDIA 4090 1대로 15만 스텝 학습.
- **적용 가능성**: 코드 미공개로 보여 재현 어려움. 다만 "공유 인코더 + 멀티태스크 헤드"라는 설계 자체는 참고 가치 있음.

## 3. f0 추출기 비교 — 수치 있으나 출처가 2차 요약이라 신뢰도 주의

**lars76/pitch-benchmark** (https://github.com/lars76/pitch-benchmark): 14개 알고리즘(SwiftF0, CREPE, TorchCREPE, PENN, BasicPitch, SPICE, RMVPE, Praat, pYIN, YAAPT, RAPT, SWIPE, DIO, Harvest)을 Bach10Synth/MDBStemSynth/MIR1K/NSynth/PTDB(Noisy)/SpeechSynth/Vocadito 8개 데이터셋에서 비교.
- **가창(Vocadito, MIR-1K) 기준 최고**: 저장소 저자 표현으로 "RMVPE — Best Human singing" (RMVPE 87.2%, CREPE 85.3%, pYIN 78.7%, TorchCREPE 80.6% — 이 숫자들은 AI 요약 도구로 README를 파싱한 값이라 **1차 검증 권장**. 직접 `git clone`해서 표 원문을 확인하시는 게 안전합니다.)
- **FCPE는 이 벤치마크에 포함되지 않음** — README 기준으로 FCPE vs 나머지 5종의 직접 비교표는 존재하지 않습니다.
- **SwiftF0** (Nieradzik, arXiv:2508.18440, 2025년 8월, 코드 https://github.com/lars76/swift-f0): CREPE보다 42배 빠름(CPU), 10dB SNR에서 조화평균 91.8%(CREPE 대비 +12%p), 파라미터 9.6만 개. "전체 최고" 성능으로 소개되나 저자 자신의 벤치마크라 독립 검증 아직 부족.
- 별도 논문 RMVPE(Wei et al., Interspeech 2023, arXiv:2306.15412): MIR-1K RPA 97.77%(FCPE 96.79%, CREPE 97.90% — 이 숫자는 RMVPE 논문 자체 수치로 보이며 위 pitch-benchmark 수치와 스케일이 다름, 데이터셋/전처리 차이로 추정).

**결론(FCPE 교체 근거)**: 확실한 근거로 교체를 권하긴 이릅니다. RMVPE가 가창 특화로 가장 자주 "베스트"로 언급되지만, FCPE 자체가 애초에 RMVPE 계열에서 파생된 경량화 모델(빠르지만 컨텍스트가 짧아 정확도 일부 희생)이라는 응답이 검색에서 나왔습니다 — 즉 속도-정확도 트레이드오프이지 FCPE가 열등하다는 명확한 근거는 아닙니다. **SwiftF0은 속도가 필요하면 시도해볼 가치가 있고(42배 빠름), RMVPE는 정확도가 아쉬우면 시도해볼 가치가 있습니다.** 둘 다 추론만 필요(사전학습 가중치 존재), GPU 재학습 불요.

## 4. 노트 분절(f0→이산 노트) — 비브라토/포르타멘토 처리

**Nishikimi, Nakamura, Goto, Itoyama, Yoshii. "Scale- and Rhythm-Aware Musical Note Estimation for Vocal F0 Trajectories Based on a Semi-Tatum-Synchronous Hierarchical Hidden Semi-Markov Model". ISMIR 2017.** (원문 전체 확인)
- **핵심**: f0 궤적을 "악보 모델(상위 HMM: 스케일→음높이, 템포 그리드 상의 온셋)" + "F0 모델(하위 HSMM: 온셋 편차·전이 구간(포르타멘토)·주파수 편차를 코시분포로 생성)"의 계층 생성모델로 결합. 온셋 편차(vocal-onset과 note-onset 사이 갭)와 F0 전이시간(포르타멘토 구간 길이)을 **별도 잠재변수로 명시적으로 모델링**해 자연스럽게 흡수. 비브라토는 코시분포의 frequency deviation으로 흡수(명시적 비브라토 모델은 없음).
- **성능**: RWC 63곡, note-level matching rate 30.7%(제안법) vs 14.8%(선행 HMM, Nishikimi 2016) vs 22.0%(단순 majority-vote). **스케일+리듬 제약이 없으면 12.9%로 급락** — 음악 이론적 사전확률(스케일/리듬)의 효과가 큼을 실측으로 보여줌.
- 코드 공개 확인 못함(데모만 http://sap.ist.i.kyoto-u.ac.jp/members/nishikimi/demo/ismir2017/).
- **관련**: Ryynänen & Klapuri (2008, Computer Music Journal)의 계층 HMM은 상위=음높이 전이, 하위=비브라토/포르타멘토 등 "음 내부 요동"을 별도 레이어로 처리 — Tony 소프트웨어(Mauch et al.)가 이 방식의 수정판을 사용.
- **시사점**: 노트 레벨 정확도가 최선의 방법으로도 30% 남짓이라는 건, **글자 단위 노트는 지금 기술 수준에서 신뢰하기 매우 어렵다**는 저희 기존 판단(줄 단위만 노트 부여)을 뒷받침하는 실측 근거입니다.

## 5. 노트 전사(singing transcription) 벤치마크

- **MIR-ST500**: 중국어 대중가요 500곡(학습 400/테스트 100), 사람이 라벨링한 보컬 멜로디 노트. AMT(자동악보전사) 표준 벤치마크. 평가지표 COn(onset)/COnP(+pitch)/COnPOff(+offset).
- **Opencpop** (wenet-e2e, https://github.com/wenet-e2e/opencpop): 만다린 여성 가수 1인, 100곡, 3756 발화, 5.2시간, 44.1kHz. **음소·음절(노트)·발화 경계와 피치 타입이 모두 어노테이션됨** — SVS(가창합성)용이지만 노트-음절 정합 학습/평가 데이터로 그대로 쓸 수 있습니다.
- **VOCANO** (ISMIR 2021, https://archives.ismir.net/ismir2021/paper/000036.pdf): pYIN 기반 f0 + 에너지·f0궤적 결합 온셋/오프셋 검출. MIR-ST500 사용. 코드 공개 여부는 원문에서 확인 못함(2차 요약 결과, 신뢰도 낮음 — 필요하시면 PDF 직접 재확인 권함).

## 6. 카라오케 채점 공개 구현 (마이크 기능 관련)

- **UltraSinger** (https://github.com/rakuri255/UltraSinger): Demucs+CREPE류+faster-whisper로 Ultrastar Deluxe/MIDI 자동 생성.
- **UltrastarCreatorTool** (https://github.com/retotito/UltrastarCreatorTool): Demucs 4.0.1 + WhisperX 정렬 + 피치검출 + 피아노롤 편집기.
- **Vocaluxe / UltraStar Deluxe / AllKaraoke**: SingStar류 오픈소스 노래방 게임, 실시간 마이크 f0 대 레퍼런스 멜로디 비교·채점 로직 참고 가능(라이선스는 프로젝트별 확인 필요).

## 확인 못한 것
- Wav2Karaoke 논문에서 "노트 온셋을 가사 정렬 제약으로 명시적으로 사용"하는 실험은 없었습니다(순차 파이프라인이라 이 아이디어를 검증한 논문이 아님) — 이 아이디어의 실측 근거는 위 Dzhambazov(2016)와 jhuang448 MTL(2022) 논문입니다.
- VOCANO 코드 공개 여부, STARS GitHub URL — 확인 못함.
- pitch-benchmark 저장소의 정확 수치(RMVPE/CREPE/pYIN/TorchCREPE %) — AI 요약 경유라 원문 표를 직접 보시는 걸 권합니다.

## 멜로디로 가사 정렬을 개선하는 가장 값싼 방법 (제안)

**이미 계산해 둔 FCPE의 프레임별 voicing confidence(또는 f0 존재 여부)를, 재학습 없이 CTC forced-alignment 디코딩 단계에서 negative prior(마스크)로 곱하는 것.** 구체적으로: 간주/무음 구간(FCPE confidence가 낮은 프레임)에서는 CTC의 blank 토큰 로그확률에 보너스를(또는 비-blank 토큰에 페널티를) 더해 해당 프레임에 음절이 배치될 가능성을 낮춥니다. Dzhambazov(2016)가 쓴 "boundary term을 argmax에 가중치로 더하는" 디코딩 방식과 원리가 같지만, 노트 온셋 검출 대신 **이미 갖고 계신 FCPE confidence를 그대로 재사용**하므로 추가 모델도 학습도 필요 없습니다. 구현은 CTC 디코더 루프에 한 줄(프레임별 로그확률 가중합) 추가하는 정도로, 지금 겪고 계신 "간주 구간에 가사 8줄이 몰리는" 증상(양성 방향: 노래가 있는 곳에 가사가 있어야 한다)에 정확히 대응합니다. jhuang448 MTL(음성 방향까지 포함해 학습 재구성)보다 훨씬 저렴한 1차 시도로 권합니다 — 효과가 부족하면 그때 MTL 학습 투자를 고려하시면 됩니다.
