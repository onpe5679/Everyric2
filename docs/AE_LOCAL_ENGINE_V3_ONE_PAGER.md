# Everyric Studio 3.0 로컬 엔진 개편 — Problem 1-Pager

## Background

Everyric Studio 2.x는 CEP 패널이 임의의 Python 실행 파일과 구 CTC 엔진을 직접 호출하는
구조다. 현재 Everyric2의 제품 정렬 경로는 무분리 omniASR 고속 단계와
BS-PolarFormer+omniASR/OWSM 정밀 단계를 조합하는 적응형 스택이다. 패널 설치와 모델
조달도 분리되어 있어 ZXP만 설치한 일반 사용자는 실행 가능한 상태를 만들기 어렵다.

## Problem

- 패널이 엔진 내부 이름과 CLI 구현에 결합되어 있다.
- Python wheel 업데이트와 모델 조달·검증·복구가 하나의 상태로 관리되지 않는다.
- CUDA·드라이버·모델·OWSM 격리 환경이 빠져도 설치 완료처럼 보일 수 있다.
- 설치 경로와 CUDA 인덱스가 Windows에 하드코딩되어 향후 플랫폼 확장이 어렵다.
- 구 엔진 공개 선택지가 남아 신 스택 실패가 잘못된 폴백으로 가려질 수 있다.

## Goal

- Everyric Studio 3.0은 신 적응형 엔진만 실행한다.
- 사용자는 CEP 안에서 런타임·모델 설치, 진행률, 상태, 업데이트, 복구를 관리한다.
- ZXP는 로컬 Runtime Manager를 부트스트랩하며 어떤 오디오도 서버로 전송하지 않는다.
- 로컬 Bridge v1이 기능 탐색, 진행률, 취소 가능한 프로세스 수명, 결과 스키마를 고정한다.
- 플랫폼별 차이는 코드 분기가 아니라 Runtime Manifest와 capability 결과로 표현한다.
- 업데이트는 스테이징→무결성 검사→health check→활성 버전 전환 순으로 수행한다.

## Non-goals

- AE 오디오 업로드 API 또는 클라우드 폴백
- macOS MPS·AMD DirectML/ROCm 모델 포팅 자체
- 자동 모션 디자인 및 UXP 마이그레이션
- 기존 구 엔진 실행 호환

기존 `.everyric.json`과 AE 프로젝트의 타이밍 데이터는 사용자 작업물이므로 계속 읽는다.
이는 구 엔진 실행 호환이 아니다.

## Constraints

- 초기 완전 로컬 실행 대상은 Windows x64 + NVIDIA CUDA 12.8이다.
- 지원하지 않는 환경은 설치 전에 명시적으로 차단하며 다른 엔진으로 폴백하지 않는다.
- 모델과 런타임은 CEP 확장 폴더 밖의 사용자 데이터 디렉터리에 둔다.
- 다운로드 파일은 고정 revision/URL과 SHA-256으로 검증한다.
- CEP가 닫혀도 기존 정상 버전은 손상되지 않아야 한다.
- OWSM은 메인 런타임과 별도 환경을 유지한다.

## Public contracts

- Engine version: `1.0.0`
- Everyric Studio version: `3.0.0`
- Local bridge protocol: `everyric-local-bridge/v1`
- Result schema: `sync-document/v3`

## Release gates

- 깨끗한 Windows 사용자 계정에서 ZXP 설치→엔진/모델 설치→health check 성공
- CUDA 실제 텐서 연산, omniASR/OWSM/PolarFormer 자산 확인
- 한국어·일본어·영어 실곡 정렬 및 AE 레이어 생성
- 다운로드 중단/재시작, 체크섬 실패, 설치 실패, 이전 정상 버전 복구
- AE 종료·재실행 후 설치 상태와 프로젝트 싱크 복원
- 패널 typecheck/unit/build와 Python 계약 테스트 통과
