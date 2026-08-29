# Everyric Engine 1.0.0

- 공개 엔진 계약을 `adaptive` 하나로 통합했습니다.
- fast는 원곡+omniASR, medium은 BS-PolarFormer+omniASR, heavy는
  BS-PolarFormer+OWSM 경로를 사용합니다.
- 로컬 런타임·모델 프로비저너와 상태/복구 명령을 추가했습니다.
- 모델 리비전과 BS-PolarFormer 파일은 매니페스트에 고정되며 직접 다운로드 파일은
  SHA-256과 크기를 검증합니다.
- 구 CTC, NeMo, GPU hybrid, SOFA 엔진 모듈은 배포 wheel에서 제외됩니다.

OWSM 체크포인트 라이선스와 학습 데이터 권리는 별도 검토 대상입니다.
