# longform2 — 장문 정렬 도구 재조사 (SendMessage 전문 1건)

**Q1. CTC-Segmentation**
논문: Kürzinger et al., "CTC-Segmentation of Large Corpora for German End-to-End Speech Recognition," Speech and Computer (SPECOM) 2020. arXiv:2007.09127
동작: 긴 오디오/텍스트를 `ctc_segmentation.get_partitions`로 파티션 분할 후 CTC posteriorgram에 DP 탐색을 적용. `forced_align`과 달리 텍스트가 오디오 전체와 정확히 1:1 대응한다고 가정하지 않고, 부분/미확정 텍스트 구간을 허용하며 구간별 신뢰도 점수를 냄.
결론: `forced_align`을 대체하기보다 **보완**함 — 내부 DP는 동일 계열(Viterbi over CTC posterior)이지만, 긴 오디오를 신뢰도 기반으로 분할/검증하는 전처리·후처리 레이어로 기능.
[github.com/lumaku/ctc-segmentation](https://github.com/lumaku/ctc-segmentation)

**Q2. 앵커 기반 재귀 분할**
확인된 원 논문: Moreno, Joerg, Van Thong, Glickman, "A recursive algorithm for the forced alignment of very long audio segments," ICSLP 1998. DOI: 10.21437/ICSLP.1998-603ㅡ [isca-archive.org/icslp_1998/moreno98b_icslp.html](https://www.isca-archive.org/icslp_1998/moreno98b_icslp.html)
방법(초록 확인): 정렬 문제를 "점점 좁아지는 사전/언어모델을 쓰는 재귀적 음성인식 문제"로 치환 — 청크를 인식해 신뢰도 높은 지점을 앵커로 잡고 그 지점에서 오디오·텍스트를 분할, 재귀적으로 반복.
**주의**: 이 앵커는 ASR 자체 신뢰도에서 자기유도(self-derived)된 것이지, 유튜브 자막 같은 **외부 시각을 양성 제약으로 주입**하는 정식 절차는 아님. 외부 앵커를 하드 제약으로 쓰는 명명된 정립 절차는 **확인 못 함**(SailAlign을 후보로 시도했으나 DBLP에서 서지 확인 실패 — 제가 갖고 있던 기억은 검증 안 됨이므로 인용하지 않음). Moreno et al.의 구조가 외부 앵커로 대체 가능한 가장 가까운 틀.

**Q3. MFA vs CTC 경계 정확도**
arXiv:2406.02560 원문(PDF Table 1)을 직접 읽었습니다. **"MFA 63.0 · WebMAUS 70.0"이라는 수치는 이 논문에 없습니다** — WebMAUS는 본문·참고문헌 어디에도 등장하지 않습니다. 실제 표(전부 ms 단위 오차, 낮을수록 좋음, %정확도 아님):
- MFA: Buckeye PBE 30/WBE 41(파인튜닝 27/36), TIMIT PBE 17/WBE 23(파인튜닝 16/22)
- 표준 CTC: Buckeye PBE 44/WBE 58, TIMIT PBE 32/WBE 42
- 논문의 label-prior CTC(제안): Buckeye PBE 38/WBE 43(파인튜닝 30/34, WBE는 MFA보다 근소 우위), TIMIT PBE 28/WBE 29 — 단 TIMIT에서는 여전히 MFA에 뒤짐
즉 "63.0/70.0"은 잘못 기억되었거나 다른 논문/다른 지표(threshold 내 정확도 %)와 혼동된 것으로 보입니다 — **확인 못 함, 이 논문 출처로는 오답**.
**왜 GMM-HMM이 표준 CTC보다 경계가 정확한가(논문 설명)**: blank 토큰이 CTC의 유효 정렬 경로에서 가장 빈번·범용적이라 학습 중 blank 많은 경로가 자기강화되어 posterior가 "peaky"해짐 → 비-blank 심볼이 프레임 하나에서만 발화 → onset은 근사하나 offset/길이(PDUR)가 모델 프레임 크기로 수렴해 부정확. GMM-HMM(MFA)은 프레임 단위 상태 정렬을 직접 지도학습 목표로 삼으므로 이 병리가 없음.
**일본어/한국어 사전학습 MFA 모델**: japanese_mfa, korean_mfa 존재를 [mfa-models 저장소](https://github.com/MontrealCorpusTools/mfa-models)에서 확인(acoustic/japanese/mfa, acoustic/korean/mfa 폴더). 단, 학습 데이터에 가창이 포함되는지는 mfa-models.readthedocs.io가 계속 429를 반환해 **확인 못 함** — 다른 언어 MFA 모델군 관례상 낭독/방송/대화 음성 코퍼스일 가능성이 높으나 확언은 못 함.

**Q4. 장문 정렬 도구의 가창 실전 사례**
GitHub 이슈 검색(공식 논문/문서 아님, 커뮤니티 시도 수준):
- **aeneas**: "Aeneas for music?" 이슈에서 사용자가 시도 후 "결과가 좋지 않다"고 보고, 해결 없이 종료. [#209](https://github.com/readbeyond/aeneas/issues/209)
- **WhisperX**: 베트남어 노래 자막([#1009](https://github.com/m-bain/whisperX/issues/1009)), 일본어 가사 정렬 시도([#1308](https://github.com/m-bain/whisperX/issues/1308)) — 둘 다 부정확/환각 문제 보고. 카라오케용 .ass 출력 요청([#579](https://github.com/m-bain/whisperX/issues/579))도 있음.
- **stable-ts**: 가사 정렬 시 단어 반복/누락([#454](https://github.com/jianfch/stable-ts/issues/454)), 10분 넘는 가사 정렬 붕괴([#385](https://github.com/jianfch/stable-ts/issues/385)).
- **Gentle**: 가창 관련 이슈를 검색했으나 **찾지 못함**(검색 결과 0건) — 확인 못 함.
결론: 넷 다 공식적으로 검증된 가창 적용 사례는 **없음**. 커뮤니티가 비공식으로 시도한 흔적은 있으나 전부 정확도 저하를 보고하며, 동료심사 논문에서 가창에 채택된 사례는 못 찾았습니다.

제약 준수: 조사만 수행, 파일 수정/커밋/원격 서버 접속/무거운 로컬 연산 없음.
