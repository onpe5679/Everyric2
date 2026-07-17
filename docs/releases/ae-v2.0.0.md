# Everyric Studio for After Effects v2.0.0

음악 가사 싱크 데이터를 **편집 가능한 AE 텍스트 레이어**로 변환하는 CEP 패널입니다. 일반 텍스트 레이어·정적 트랜스폼·in/out 시간·참고 마커만 생성하며, Text Animator나 모션 키프레임은 만들지 않습니다 — 후반 디자인은 여러분의 몫입니다.

## 주요 기능

- **선택 레이어 채우기** — 레이어의 시간·디자인은 그대로 두고 Source Text만 가사로 교체
- **타이포그래피 생성** — 싱크 원자를 읽기 단위 블록/카드로 묶어 텍스트 레이어 자동 배치
  - 읽기 리듬 프리셋 (Readable / Balanced / Rhythmic) + 실제 가사 분할 미리보기
  - 읽기 속도·최단 노출 기반 위험 구간 경고
  - 생성 전 교체 범위 확인, Undo 1회로 전체 복원
- **로컬 싱크 생성** — 컴포지션 오디오와 가사로 Everyric2 정렬 실행 (엔진 자동 설치 지원)
- 라인 타이밍 참고 마커, Everyric 생성물만 정리하는 안전한 클린업

## 설치

1. 릴리스 에셋에서 `Everyric-Studio-2.0.0.zxp` 다운로드
2. [aescripts ZXP Installer](https://aescripts.com/learn/zxp-installer/)로 열기
3. After Effects 재시작 → **Window → Extensions → Everyric Studio**

ZXP Installer를 쓸 수 없는 경우 `Everyric-Studio-2.0.0-manual.zip`을 받아 `install.bat`을 실행하세요.

## 요구 사항

| 용도 | 요구 사항 |
|---|---|
| 패널 (JSON 불러오기 · 타이포 생성) | After Effects 2024(24.0) 이상 · Windows |
| 로컬 싱크 생성 (선택) | 패널 내 "엔진 설치" 버튼으로 자동 설치 (CPU 약 300MB / CUDA 약 2.5GB) |

## 알려진 제한

- Windows 전용입니다. macOS는 지원하지 않습니다.
- 자가 서명 인증서로 서명되어 있어 일부 환경에서 설치 시 경고가 표시될 수 있습니다.
