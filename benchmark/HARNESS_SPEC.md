# 벤치 하네스 스펙 (2026-07-30 모델 교체 이니셔티브 P1)

목적: 분리·정렬 후보 모델의 광역 스윕 A/B. 로컬 RTX 5090에서 실행(사용자 승인). 프로드 서버 접근 금지 — 완전 로컬.

## 입력 (이미 준비됨)
- `benchmark/eval_set.json` — 73곡: video_id, title, artist, language(ja/ko/en/zh + `_mms` 접미 변형), stratum, quality_score(프로드 당시), duration_est, lyrics(원 가사 전문), baseline_timestamps(프로드 싱크 — segments 리스트: text/start/end/confidence/words/pron 등)
- `benchmark/audio/<video_id>.m4a` — 원본 오디오 (서버에서 수집 중, 곧 도착)

## 아키텍처 — 어댑터 2축
```python
class SeparatorAdapter:   # name: "htdemucs" (P1은 이것만 구현)
    def separate(self, audio_path: Path, work_dir: Path) -> SeparationOut:  # vocals_path, inst_path, elapsed_sec, vram_peak_mb

class AlignerAdapter:     # name: "mms-baseline" (P1은 이것만 구현)
    def align(self, vocals_path: Path, lyrics: str, language: str) -> AlignOut:  # lines: [{text,start,end,confidence}], elapsed_sec, vram_peak_mb, quality_score
```
- 레지스트리 dict + CLI `--separators htdemucs --aligners mms-baseline --songs <vid,vid|all> --repeat 1 --strata ja,ko`
- 분리 결과는 `(separator, video_id)` 키로 디스크 캐시(`benchmark/stems/<separator>/<vid>/vocals.wav|inst.wav`) — 정렬 후보들이 재사용
- 구현은 기존 everyric2 내부 재사용: `everyric2.audio.separator.VocalSeparator`(demucs), `everyric2.alignment.ctc_engine`(MMS CTC — 엔진 생성·호출 방법은 코드 탐색해서 파악; `everyric2.audio.loader.AudioLoader`로 로드). language의 `_mms` 접미는 벗겨서 기본 언어로 전달.
- VRAM 측정: `torch.cuda.reset_peak_memory_stats()` / `max_memory_allocated()` 스테이지별.

## 실행 결과 저장 (재개 가능)
- `benchmark/runs/<separator>__<aligner>/<video_id>__r<run_idx>.json` — AlignOut 전문 + 스테이지 시간·VRAM. 존재하면 스킵(resume).

## 지표 (mir_eval 사용, 이미 .venv에 설치됨)
기준(reference)은 **프로드 baseline_timestamps의 라인 start들** — 절대 정답이 아니라 공통 비교 기준임을 리포트에 명시.
- 곡별: 라인 온셋 AAE/MAE(mir_eval.alignment.absolute_error), PCO(percentage_correct, window 0.3), P95 |delta|, 최대 |delta|, 300ms 초과 라인 비율, 라인 수 불일치 여부
- 붕괴 게이트: aligner가 주는 quality_score(현행 ctc_engine의 산출 방식 재사용), 정렬 실패/예외 여부
- 집계: stratum별 + 전체, 곡 단위 paired(같은 곡의 baseline 대비)

## 사용자 검수 산출물 (필수 — 수치만큼 중요)
```
benchmark/results/<stratum>/<video_id>__<제목슬러그>/
  audio/original.m4a            # 원본 (복사 또는 하드링크)
  audio/<separator>/vocals.wav  # 분리 후보별 — 그대로 재생 가능해야 함
  audio/<separator>/inst.wav
  align/<aligner>.srt           # 정렬 후보별 SRT
  align/baseline-prod.srt       # 프로드 기준선도 SRT로
  align/diff__<aligner>__vs__baseline.html   # 라인 표: text | baseline start | candidate start | Δ(초) — |Δ|>0.3s 행 배경 하이라이트, 곡 요약(MAE/P95) 헤더
  summary.md                    # 곡별 지표 + 특이사항 (붕괴·라인수 불일치 등)
```
- 최상위 `benchmark/REPORT.md` — 후보 조합별 집계표(stratum × 지표), 갱신형(재실행 시 덮어씀)
- 제목슬러그: 파일명 안전화(한글·일본어 유지하되 금지문자 제거, 40자 절단)

## 제약
- `.gitignore`에 `benchmark/` 전체 추가(스펙 문서인 이 파일과 REPORT.md만 예외: `!benchmark/HARNESS_SPEC.md`, `!benchmark/REPORT.md`) — 가사·오디오는 절대 커밋 금지
- 테스트: pytest 스위트에 넣지 말 것(무거움). 검증은 스모크 실행으로 — 오디오 도착한 곡 중 ja 1곡·ko 1곡으로 `--songs` 지정 실행해 전 산출물(스템 wav, SRT, diff html, runs json) 생성 확인
- Windows 경로·인코딩(utf-8) 주의. 실행 인터프리터: `./.venv/Scripts/python.exe`
- 서버(100.76.4.47)·프로드 DB 접근 금지. 예외/실패 곡은 기록하고 계속(전체 중단 금지)
```
./.venv/Scripts/python.exe scripts/benchmark_alignment.py --separators htdemucs --aligners mms-baseline --songs <vid1>,<vid2>
```
